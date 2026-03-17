"""HeteroFloodGNN: 1D-2Dヘテロジニアスグラフニューラルネットワーク.

Encode-Process-Decode architecture:
  Encoder: ノード/エッジタイプ別MLP → 共通hidden次元
  Processor: HeteroConv × L layers with residual
  Decoder: ノードタイプ別MLP → delta_water_level
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv, LayerNorm, MessagePassing
from typing import Dict, Optional, Tuple, Union


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_layers: int = 2,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """汎用MLP."""
    act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
    layers = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # No activation/dropout on last layer
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class HeteroFloodGNN(nn.Module):
    """1D-2D結合都市洪水予測GNN.

    Input: HeteroData with node types '1d', '2d' and edge types
           ('1d','pipe','1d'), ('2d','surface','2d'),
           ('1d','coupling','2d'), ('2d','coupling','1d')
    Output: dict {'1d': [N_1d, 1], '2d': [N_2d, 1]} (delta_water_level)
    """

    # 各ノード/エッジタイプの入力次元
    NODE_DIMS = {"1d": 10, "2d": 15}   # static + dynamic*2
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 9,       # 7 static + 2 dynamic
        ("2d", "surface", "2d"): 7,    # 5 static + 2 dynamic
    }

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        processor_mlp_layers: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.0,  # GraphCast式ノイズ注入
        node_dim_1d: int = 10,   # 1Dノード入力次元 (inject_rainfall=True → 12, extra_features → 14)
        node_dim_2d: int = 15,   # 2Dノード入力次元 (extra_features=True → 19)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        node_dims = {"1d": node_dim_1d, "2d": node_dim_2d}

        # === Encoders ===
        self.node_encoders = nn.ModuleDict({
            ntype: make_mlp(dim, hidden_dim, hidden_dim, encoder_layers, activation)
            for ntype, dim in node_dims.items()
        })

        self.edge_encoders = nn.ModuleDict()
        for etype, dim in self.EDGE_DIMS.items():
            key = "__".join(etype)  # PyGのModuleDictキー制約
            self.edge_encoders[key] = make_mlp(dim, hidden_dim, hidden_dim, encoder_layers, activation)

        # === Processor (Message Passing) ===
        self.processors = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None

        for _ in range(num_processor_layers):
            conv_dict = {}
            # 1D pipe edges
            conv_dict[("1d", "pipe", "1d")] = SAGEConv(
                hidden_dim, hidden_dim, aggr="mean",
            )
            # 2D surface edges
            conv_dict[("2d", "surface", "2d")] = SAGEConv(
                hidden_dim, hidden_dim, aggr="mean",
            )
            # Coupling edges (1D↔2D) — no edge features
            conv_dict[("1d", "coupling", "2d")] = SAGEConv(
                hidden_dim, hidden_dim, aggr="mean",
            )
            conv_dict[("2d", "coupling", "1d")] = SAGEConv(
                hidden_dim, hidden_dim, aggr="mean",
            )
            self.processors.append(HeteroConv(conv_dict, aggr="sum"))

            if use_layer_norm:
                self.norms.append(nn.ModuleDict({
                    "1d": LayerNorm(hidden_dim),
                    "2d": LayerNorm(hidden_dim),
                }))

        # === Decoders ===
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        # === Encode ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            # Training時にノイズ注入 (rollout安定化)
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        edge_index_dict = {}
        for etype in data.edge_types:
            edge_index_dict[etype] = data[etype].edge_index

        # === Process ===
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict)
            # Residual + LayerNorm
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])

        # === Decode ===
        out = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])  # [N, 1]

        return out

    def predict_delta(self, data: HeteroData) -> Dict[str, Tensor]:
        """delta_water_levelを予測. forward()のエイリアス."""
        return self.forward(data)

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NodeEdgeConv(MessagePassing):
    """MLP-based message passing: MLP(x_j || edge_attr || x_i).

    ベースラインのNodeEdgeConvを参考。SAGEConv/GATv2Convと異なり、
    エッジ特徴量をメッセージ計算に直接使用する。
    - pipe/surface: MLP(x_src || encoded_edge || x_tgt) — 水理情報を直接反映
    - coupling: MLP(x_src || x_tgt) — edge featなし
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        edge_dim: int = 0,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        aggr: str = "mean",  # "sum" for flow conservation (DUALFloodGNN)
    ):
        super().__init__(aggr=aggr)
        hidden = hidden_dim or out_dim
        in_dim = node_dim * 2 + edge_dim
        self.mlp = make_mlp(in_dim, out_dim, hidden, 2, activation, dropout)
        self.has_edge_feat = edge_dim > 0

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        if self.has_edge_feat and edge_attr is not None:
            return self.mlp(torch.cat([x_j, edge_attr, x_i], dim=-1))
        return self.mlp(torch.cat([x_j, x_i], dim=-1))


class HeteroFloodGNNv3(nn.Module):
    """v3: NodeEdgeConvでエッジ特徴量をメッセージ計算に直接使用.

    v1 (SAGEConv) との差分:
    - pipe/surface: NodeEdgeConv — MLP(x_j || edge_attr || x_i) (水理情報直接使用)
    - coupling: NodeEdgeConv — MLP(x_j || x_i) (edge featなし)
    - エッジ特徴量は各レイヤーで更新 (ベースラインの手法)
    - noise_std=0.02 (保守的delta予測を防ぐ)

    EDA根拠:
    - SAGEConvはedge_attrを無視 → パイプ径/勾配/流速の情報が失われる
    - deg=1 (葉ノード)が最悪のエラー → エッジ属性がないと隣接1ノードのmeanだけ
    - GATv2Conv (v2)はattentionにのみedge使用、メッセージ内容には不使用 → 効果なし
    """

    NODE_DIMS = {"1d": 10, "2d": 15}
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 9,
        ("2d", "surface", "2d"): 7,
    }

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        # === Node Encoders (v1と同一構造 → weight transfer可能) ===
        self.node_encoders = nn.ModuleDict({
            ntype: make_mlp(dim, hidden_dim, hidden_dim, encoder_layers, activation)
            for ntype, dim in self.NODE_DIMS.items()
        })

        # === Edge Encoders (raw features → hidden_dim) ===
        self.edge_encoders = nn.ModuleDict()
        for etype, dim in self.EDGE_DIMS.items():
            key = "__".join(etype)
            self.edge_encoders[key] = make_mlp(
                dim, hidden_dim, hidden_dim, encoder_layers, activation,
            )

        # === Processor: NodeEdgeConv (MLP-based message passing) ===
        self.processors = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None

        # Edge update MLPs: residual update of edge features each layer
        self.edge_updates = nn.ModuleList()

        for _ in range(num_processor_layers):
            conv_dict = {}
            # pipe/surface: NodeEdgeConv with edge features
            conv_dict[("1d", "pipe", "1d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
            )
            conv_dict[("2d", "surface", "2d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
            )
            # coupling: NodeEdgeConv without edge features
            conv_dict[("1d", "coupling", "2d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=0,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
            )
            conv_dict[("2d", "coupling", "1d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=0,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
            )
            self.processors.append(HeteroConv(conv_dict, aggr="sum"))

            if use_layer_norm:
                self.norms.append(nn.ModuleDict({
                    "1d": LayerNorm(hidden_dim),
                    "2d": LayerNorm(hidden_dim),
                }))

            # Edge update: MLP(x_src || edge_attr || x_tgt) → residual update
            updates = nn.ModuleDict({
                "1d__pipe__1d": make_mlp(
                    hidden_dim * 3, hidden_dim, hidden_dim, 2, activation, dropout,
                ),
                "2d__surface__2d": make_mlp(
                    hidden_dim * 3, hidden_dim, hidden_dim, 2, activation, dropout,
                ),
            })
            self.edge_updates.append(updates)

        # === Decoders (v1と同一構造 → weight transfer可能) ===
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges → edge_attr_dict ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process ===
        for i, conv in enumerate(self.processors):
            # NodeEdgeConv message passing (uses edge features in message MLP)
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

            # Residual + LayerNorm for nodes
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])

            # Edge feature update (residual): MLP(x_src || edge || x_tgt)
            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

        # === Decode ===
        out = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])
        return out

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HeteroFloodGNNv4(nn.Module):
    """v4: aggr="sum" + future rainfall features.

    v3からの差分:
    - NodeEdgeConv aggr="sum" (DUALFloodGNN準拠, 流量保存に物理的意味)
    - NODE_DIMS: 1D=13 (+3 future rain), 2D=18 (+3 future rain)
    - future_rain_steps=10 で未来10step降雨統計を入力

    DUALFloodGNN (arxiv:2512.23964) の知見:
    - sum aggregation は流量保存に重要 (mean → 隣接数で薄まる)
    - 未来降雨は全テストタイムステップで既知 → 合法的な強い特徴量

    coupling_edge_dim: >0でcoupling edgeに水頭差featureを使用 (0=互換モード)
    """

    NODE_DIMS = {"1d": 13, "2d": 18}  # +3 each: fr_max, fr_sum, rem_sum
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 9,
        ("2d", "surface", "2d"): 7,
    }

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
        coupling_edge_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.coupling_edge_dim = coupling_edge_dim

        # === Node Encoders ===
        self.node_encoders = nn.ModuleDict({
            ntype: make_mlp(dim, hidden_dim, hidden_dim, encoder_layers, activation)
            for ntype, dim in self.NODE_DIMS.items()
        })

        # === Edge Encoders ===
        self.edge_encoders = nn.ModuleDict()
        for etype, dim in self.EDGE_DIMS.items():
            key = "__".join(etype)
            self.edge_encoders[key] = make_mlp(
                dim, hidden_dim, hidden_dim, encoder_layers, activation,
            )

        # Coupling edge encoder: 水頭差 → hidden_dim
        if coupling_edge_dim > 0:
            self.edge_encoders["1d__coupling__2d"] = make_mlp(
                coupling_edge_dim, hidden_dim, hidden_dim, encoder_layers, activation,
            )
            self.edge_encoders["2d__coupling__1d"] = make_mlp(
                coupling_edge_dim, hidden_dim, hidden_dim, encoder_layers, activation,
            )

        coupling_conv_edim = hidden_dim if coupling_edge_dim > 0 else 0

        # === Processor: NodeEdgeConv with aggr="sum" ===
        self.processors = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        self.edge_updates = nn.ModuleList()

        for _ in range(num_processor_layers):
            conv_dict = {}
            conv_dict[("1d", "pipe", "1d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("2d", "surface", "2d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("1d", "coupling", "2d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=coupling_conv_edim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("2d", "coupling", "1d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=coupling_conv_edim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            self.processors.append(HeteroConv(conv_dict, aggr="sum"))

            if use_layer_norm:
                self.norms.append(nn.ModuleDict({
                    "1d": LayerNorm(hidden_dim),
                    "2d": LayerNorm(hidden_dim),
                }))

            updates = nn.ModuleDict({
                "1d__pipe__1d": make_mlp(
                    hidden_dim * 3, hidden_dim, hidden_dim, 2, activation, dropout,
                ),
                "2d__surface__2d": make_mlp(
                    hidden_dim * 3, hidden_dim, hidden_dim, 2, activation, dropout,
                ),
            })
            self.edge_updates.append(updates)

        # === Decoders ===
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        # Coupling edge features (水頭差)
        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process ===
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])

            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

        # === Decode ===
        out = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])
        return out

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HeteroFloodGNNv5(nn.Module):
    """v5: Dual Prediction (Node + Edge) + Mass Conservation Loss.

    v4からの差分:
    - Edge Decoders 追加: flow/velocity delta予測
    - Dual Output: {"nodes": {...}, "edges": {...}}
    - Mass Conservation Loss対応 (訓練スクリプト側で適用)

    DUALFloodGNN論文のablation study:
    - 物理Loss なし: baseline
    - 物理Loss あり: ~15%改善 (実証済み)

    期待効果: LB 0.0903 → 0.077 (-15%)
    """

    NODE_DIMS = {"1d": 13, "2d": 18}  # +3 each: fr_max, fr_sum, rem_sum
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 9,
        ("2d", "surface", "2d"): 7,
    }

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        # === Node Encoders ===
        self.node_encoders = nn.ModuleDict({
            ntype: make_mlp(dim, hidden_dim, hidden_dim, encoder_layers, activation)
            for ntype, dim in self.NODE_DIMS.items()
        })

        # === Edge Encoders ===
        self.edge_encoders = nn.ModuleDict()
        for etype, dim in self.EDGE_DIMS.items():
            key = "__".join(etype)
            self.edge_encoders[key] = make_mlp(
                dim, hidden_dim, hidden_dim, encoder_layers, activation,
            )

        # === Processor: NodeEdgeConv with aggr="sum" ===
        self.processors = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        self.edge_updates = nn.ModuleList()

        for _ in range(num_processor_layers):
            conv_dict = {}
            conv_dict[("1d", "pipe", "1d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("2d", "surface", "2d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("1d", "coupling", "2d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=0,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("2d", "coupling", "1d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=0,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            self.processors.append(HeteroConv(conv_dict, aggr="sum"))

            if use_layer_norm:
                self.norms.append(nn.ModuleDict({
                    "1d": LayerNorm(hidden_dim),
                    "2d": LayerNorm(hidden_dim),
                }))

            updates = nn.ModuleDict({
                "1d__pipe__1d": make_mlp(
                    hidden_dim * 3, hidden_dim, hidden_dim, 2, activation, dropout,
                ),
                "2d__surface__2d": make_mlp(
                    hidden_dim * 3, hidden_dim, hidden_dim, 2, activation, dropout,
                ),
            })
            self.edge_updates.append(updates)

        # === Node Decoders (水位変化量) ===
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })

        # === Edge Decoders (流量変化量) NEW ===
        self.edge_decoders = nn.ModuleDict({
            "1d__pipe__1d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
            "2d__surface__2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })

    def forward(self, data: HeteroData) -> Dict[str, Dict[str, Tensor]]:
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process ===
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])

            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

        # === Decode Nodes (水位変化量) ===
        node_out = {}
        for ntype in ["1d", "2d"]:
            node_out[ntype] = self.node_decoders[ntype](x_dict[ntype])

        # === Decode Edges (流量変化量) ===
        edge_out = {}
        for etype_key, decoder in self.edge_decoders.items():
            etype = tuple(etype_key.split("__"))
            if etype in edge_attr_dict:
                edge_out[etype] = decoder(edge_attr_dict[etype])

        # Dual Output
        return {
            "nodes": node_out,  # {"1d": [N_1d, 1], "2d": [N_2d, 1]}
            "edges": edge_out,  # {("1d","pipe","1d"): [E_1d, 1], ...}
        }

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HeteroFloodGNNv6(nn.Module):
    """v6: GNN-GRU Hybrid — temporal memory for autoregressive stability.

    v4c backbone (NodeEdgeConv + coupling edge features) + GRU per node type:
      Encode → Process (NodeEdgeConv×4) → GRU → Gated Residual → Decode

    - GRU preserves temporal context across autoregressive steps
    - Gated residual: gate初期bias=-2.0 → sigmoid(-2)≈0.12 → GNNパススルー優勢
      → 徐々にGRU活用を学習
    - step_ratio入力: NODE_DIMS +1 (v4cの13→14 for 1d, 18→19 for 2d)
    - coupling_edge_dim: 水頭差 edge features (v4c互換)
    """

    NODE_DIMS = {"1d": 14, "2d": 19}  # v4c + 1 step_ratio each
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 9,
        ("2d", "surface", "2d"): 7,
    }

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
        coupling_edge_dim: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.coupling_edge_dim = coupling_edge_dim

        # === Node Encoders (14-dim 1d, 19-dim 2d) ===
        self.node_encoders = nn.ModuleDict({
            ntype: make_mlp(dim, hidden_dim, hidden_dim, encoder_layers, activation)
            for ntype, dim in self.NODE_DIMS.items()
        })

        # === Edge Encoders ===
        self.edge_encoders = nn.ModuleDict()
        for etype, dim in self.EDGE_DIMS.items():
            key = "__".join(etype)
            self.edge_encoders[key] = make_mlp(
                dim, hidden_dim, hidden_dim, encoder_layers, activation,
            )

        # Coupling edge encoder (水頭差)
        if coupling_edge_dim > 0:
            self.edge_encoders["1d__coupling__2d"] = make_mlp(
                coupling_edge_dim, hidden_dim, hidden_dim, encoder_layers, activation,
            )
            self.edge_encoders["2d__coupling__1d"] = make_mlp(
                coupling_edge_dim, hidden_dim, hidden_dim, encoder_layers, activation,
            )

        coupling_conv_edim = hidden_dim if coupling_edge_dim > 0 else 0

        # === Processor: NodeEdgeConv with aggr="sum" (v4c identical) ===
        self.processors = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        self.edge_updates = nn.ModuleList()

        for _ in range(num_processor_layers):
            conv_dict = {}
            conv_dict[("1d", "pipe", "1d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("2d", "surface", "2d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("1d", "coupling", "2d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=coupling_conv_edim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            conv_dict[("2d", "coupling", "1d")] = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=coupling_conv_edim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            self.processors.append(HeteroConv(conv_dict, aggr="sum"))

            if use_layer_norm:
                self.norms.append(nn.ModuleDict({
                    "1d": LayerNorm(hidden_dim),
                    "2d": LayerNorm(hidden_dim),
                }))

            updates = nn.ModuleDict({
                "1d__pipe__1d": make_mlp(
                    hidden_dim * 3, hidden_dim, hidden_dim, 2, activation, dropout,
                ),
                "2d__surface__2d": make_mlp(
                    hidden_dim * 3, hidden_dim, hidden_dim, 2, activation, dropout,
                ),
            })
            self.edge_updates.append(updates)

        # === GRU per node type (temporal memory) ===
        self.gru_1d = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_2d = nn.GRUCell(hidden_dim, hidden_dim)

        # === Gated Residual: gate * gru_out + (1 - gate) * gnn_out ===
        # bias=-2.0 → sigmoid(-2)≈0.12 → 初期状態でGNNパススルー優勢
        self.gate_1d = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_2d = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.gate_1d.bias, -2.0)
        nn.init.constant_(self.gate_2d.bias, -2.0)

        # === Decoders ===
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })

    def forward(
        self,
        data: HeteroData,
        h_1d: Optional[Tensor] = None,
        h_2d: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        """Forward with GRU hidden state.

        Args:
            data: HeteroData graph at current timestep
            h_1d: GRU hidden state for 1D nodes [N_1d, hidden_dim]
            h_2d: GRU hidden state for 2D nodes [N_2d, hidden_dim]

        Returns:
            (out_dict, h_1d_new, h_2d_new)
        """
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        # Coupling edge features (水頭差)
        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process (NodeEdgeConv × L layers) ===
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])

            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

        # === GRU temporal memory ===
        gnn_1d = x_dict["1d"]  # [N_1d, hidden_dim]
        gnn_2d = x_dict["2d"]  # [N_2d, hidden_dim]

        if h_1d is None:
            h_1d = torch.zeros_like(gnn_1d)
        if h_2d is None:
            h_2d = torch.zeros_like(gnn_2d)

        h_1d_new = self.gru_1d(gnn_1d, h_1d)
        h_2d_new = self.gru_2d(gnn_2d, h_2d)

        # === Gated Residual ===
        gate_1d = torch.sigmoid(self.gate_1d(torch.cat([gnn_1d, h_1d_new], dim=-1)))
        gate_2d = torch.sigmoid(self.gate_2d(torch.cat([gnn_2d, h_2d_new], dim=-1)))
        fused_1d = gate_1d * h_1d_new + (1 - gate_1d) * gnn_1d
        fused_2d = gate_2d * h_2d_new + (1 - gate_2d) * gnn_2d

        # === Decode ===
        out = {}
        out["1d"] = self.node_decoders["1d"](fused_1d)
        out["2d"] = self.node_decoders["2d"](fused_2d)

        return out, h_1d_new, h_2d_new

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HeteroFloodGNNv2(nn.Module):
    """v2: GATv2Convでエッジ特徴量をprocessorで活用.

    v1との差分:
    - pipe/surface: GATv2Conv (edge_dim=hidden_dim, multi-head attention)
    - coupling: SAGEConv (変更なし, edge featなし)
    - edge_encodersの出力をprocessorに渡す
    - 6 processor layers (v1は4)
    """

    NODE_DIMS = {"1d": 10, "2d": 15}
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 9,
        ("2d", "surface", "2d"): 7,
    }

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 6,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        heads: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        # === Node Encoders ===
        self.node_encoders = nn.ModuleDict({
            ntype: make_mlp(dim, hidden_dim, hidden_dim, encoder_layers, activation)
            for ntype, dim in self.NODE_DIMS.items()
        })

        # === Edge Encoders (pipe/surface → hidden_dim) ===
        self.edge_encoders = nn.ModuleDict()
        for etype, dim in self.EDGE_DIMS.items():
            key = "__".join(etype)
            self.edge_encoders[key] = make_mlp(dim, hidden_dim, hidden_dim, encoder_layers, activation)

        # === Processor: GATv2Conv (edge features) + SAGEConv (coupling) ===
        assert hidden_dim % heads == 0, f"hidden_dim={hidden_dim} must be divisible by heads={heads}"
        head_dim = hidden_dim // heads

        self.processors = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None

        for _ in range(num_processor_layers):
            conv_dict = {}
            # GATv2Conv: edge_dim対応, multi-head attention
            conv_dict[("1d", "pipe", "1d")] = GATv2Conv(
                hidden_dim, head_dim, heads=heads,
                edge_dim=hidden_dim, concat=True, dropout=dropout,
            )
            conv_dict[("2d", "surface", "2d")] = GATv2Conv(
                hidden_dim, head_dim, heads=heads,
                edge_dim=hidden_dim, concat=True, dropout=dropout,
            )
            # SAGEConv: coupling (no edge features)
            conv_dict[("1d", "coupling", "2d")] = SAGEConv(
                hidden_dim, hidden_dim, aggr="mean",
            )
            conv_dict[("2d", "coupling", "1d")] = SAGEConv(
                hidden_dim, hidden_dim, aggr="mean",
            )
            self.processors.append(HeteroConv(conv_dict, aggr="sum"))

            if use_layer_norm:
                self.norms.append(nn.ModuleDict({
                    "1d": LayerNorm(hidden_dim),
                    "2d": LayerNorm(hidden_dim),
                }))

        # === Decoders ===
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges → edge_attr_dict (pipe/surfaceのみ) ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process (GATv2Conv uses edge_attr, SAGEConv ignores it) ===
        for i, conv in enumerate(self.processors):
            # HeteroConvはkwargが_dict suffix必須 → 各convにedge_attrとして渡される
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])

        # === Decode ===
        out = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])
        return out

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HeteroFloodGNNv11(HeteroFloodGNNv4):
    """v11: Full-Variable AR with correct flux/state paradigm.

    State variables (delta prediction, cumulative feedback):
      - 1D water_level_delta, 2D water_level_delta

    Flux variables (absolute prediction, per-step overwrite):
      - inlet_flow (1D node output dim 1)
      - 1D edge flow + velocity (edge decoder, 2dim)

    No water_volume (unpredictable, std=81.1).
    No 2D edge prediction (too many edges).

    Output: {"1d": [N_1d, 2], "2d": [N_2d, 1], "1d_edge": [E_1d, 2]}
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
        coupling_edge_dim: int = 0,
        global_pool_1d: bool = False,  # v52b: 1D global mean pooling per processor layer
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_processor_layers=num_processor_layers,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            noise_std=noise_std,
            coupling_edge_dim=coupling_edge_dim,
        )
        self.global_pool_1d = global_pool_1d
        # Override node decoders: 1D=2dim (wl_delta + inlet_flow_abs), 2D=1dim (wl_delta)
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })
        # NEW: 1D pipe edge decoder (from processed edge embeddings → flow + velocity absolute)
        self.edge_decoders = nn.ModuleDict({
            "1d__pipe__1d": make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation),
        })
        # v52b: global pooling MLPs (1 per processor layer)
        if global_pool_1d:
            self.global_pool_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_processor_layers)
            ])
            # Zero-init last linear so initial output is zero (preserves v39_p3 behavior)
            for mlp in self.global_pool_mlps:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        """v4c forward + edge decode. Returns node + edge predictions."""
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process (NodeEdgeConv + edge updates) ===
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])
            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

            # v52b: global pooling for 1D nodes — broadcast global info to all nodes
            if self.global_pool_1d:
                global_1d = x_dict["1d"].mean(dim=0, keepdim=True)  # [1, H]
                x_dict["1d"] = x_dict["1d"] + self.global_pool_mlps[i](global_1d)

        # === Decode Nodes ===
        out: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])

        # === Decode 1D Pipe Edges (flow + velocity absolute) ===
        pipe_key = ("1d", "pipe", "1d")
        if pipe_key in edge_attr_dict:
            out["1d_edge"] = self.edge_decoders["1d__pipe__1d"](edge_attr_dict[pipe_key])

        return out


class HeteroFloodGNNv11_TB(HeteroFloodGNNv11):
    """v11b + Temporal Bundling: predict K step-deltas from a single graph.

    Decoder出力を K 倍に拡張。encoder/processor は v11b と同一。
    Output layout (interleaved):
      1D:      [wl_delta_0, inlet_0, wl_delta_1, inlet_1, ..., wl_delta_{K-1}, inlet_{K-1}]  → [N, K*2]
      2D:      [wl_delta_0, wl_delta_1, ..., wl_delta_{K-1}]                                  → [N, K]
      1D_edge: [flow_0, vel_0, flow_1, vel_1, ..., flow_{K-1}, vel_{K-1}]                     → [E, K*2]

    Step k のアクセス:
      1D wl_delta: out["1d"][:, 2*k],   inlet: out["1d"][:, 2*k+1]
      2D wl_delta: out["2d"][:, k]
      edge flow:   out["1d_edge"][:, 2*k],  vel: out["1d_edge"][:, 2*k+1]
    """

    def __init__(self, K: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        hidden_dim = kwargs.get("hidden_dim", 128)
        decoder_layers = kwargs.get("decoder_layers", 2)
        activation = kwargs.get("activation", "gelu")
        # Override decoders: K * original output dims
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, K * 2, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, K * 1, hidden_dim, decoder_layers, activation),
        })
        self.edge_decoders = nn.ModuleDict({
            "1d__pipe__1d": make_mlp(hidden_dim, K * 2, hidden_dim, decoder_layers, activation),
        })


class HeteroFloodGNNv15(HeteroFloodGNNv11):
    """v15: v11 + 3-step temporal context + edge hydraulic gradients.

    GNN-SWS論文の知見を統合:
    - NODE_DIMS拡張: t-2動的特徴量を追加 (1D: 13→15, 2D: 18→21)
    - EDGE_DIMS拡張: 水力勾配Δdepthを追加 (pipe: 9→11, surface: 7→9)
    - depth≥0ペナルティは訓練スクリプト側で適用
    - v11bからweight transferでエンコーダ零パディング

    Feature order (v11bとの互換性):
      1D: [static(6), dyn_t(2), dyn_prev(2), future_rain(3), dyn_prev2(2)]
      2D: [static(9), dyn_t(3), dyn_prev(3), future_rain(3), dyn_prev2(3)]
      pipe edge: [static(7), dynamic(2), hgrad(2)]
      surface edge: [static(5), dynamic(2), hgrad(2)]
    """

    NODE_DIMS = {"1d": 15, "2d": 21}
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 11,
        ("2d", "surface", "2d"): 9,
    }


class HeteroFloodGNNv11_GRU(HeteroFloodGNNv11):
    """v11 + 1D専用GRU: GNNの空間処理 + GRUの時間記憶.

    GNN message passing後、1Dノードのみ GRUCell で時系列記憶を付加。
    2Dノードはそのまま (GNN出力をパススルー)。
    Gated residual で GNN ⇔ GRU のバランスを学習。

    forward signature:
      forward(data, h_1d=None) -> (out_dict, h_1d_new)
      h_1d: [N_1d, hidden_dim] GRU hidden state (Noneなら自動ゼロ初期化)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = kwargs.get("hidden_dim", 128)

        # 1D専用 GRUCell
        self.gru_1d = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_ln = nn.LayerNorm(hidden_dim)

        # Gated residual: gate = sigmoid(linear(gnn_out || gru_out))
        # bias=-2.0 → 初期gate≈0.12 → ほぼGNNパススルーから開始
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.gate_linear.bias, -2.0)

    def forward(
        self,
        data: HeteroData,
        h_1d: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """GNN forward + 1D GRU temporal update.

        Returns:
            out: {"1d": [N_1d, 2], "2d": [N_2d, 1], "1d_edge": [E_1d, 2]}
            h_1d_new: [N_1d, hidden_dim]
        """
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process (NodeEdgeConv + edge updates) ===
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])
            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

        # === 1D GRU temporal update (1Dノードのみ) ===
        gnn_1d = x_dict["1d"]  # [N_1d, H]
        if h_1d is None:
            h_1d = torch.zeros_like(gnn_1d)
        h_1d_new = self.gru_1d(gnn_1d, h_1d)
        h_1d_new = self.gru_ln(h_1d_new)

        # Gated residual: output = gate * gru + (1-gate) * gnn
        gate = torch.sigmoid(self.gate_linear(torch.cat([gnn_1d, h_1d_new], dim=-1)))
        x_dict["1d"] = gate * h_1d_new + (1.0 - gate) * gnn_1d

        # === Decode Nodes ===
        out: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])

        # === Decode 1D Pipe Edges ===
        pipe_key = ("1d", "pipe", "1d")
        if pipe_key in edge_attr_dict:
            out["1d_edge"] = self.edge_decoders["1d__pipe__1d"](edge_attr_dict[pipe_key])

        return out, h_1d_new


class HeteroFloodGNNv18(HeteroFloodGNNv4):
    """v18: v4 architecture + v15 input dims (3-step temporal + edge hgrad).
    v4のシンプル出力 (1d→[N,1], 2d→[N,1]) + v15の拡張入力.
    """
    NODE_DIMS = {"1d": 15, "2d": 21}
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 11,
        ("2d", "surface", "2d"): 9,
    }


class HeteroFloodGNNv10(HeteroFloodGNNv4):
    """v10: Multi-Variable Autoregressive.

    v4cベース。ノードデコーダを多出力化:
    - 1D: water_level_delta + inlet_flow (2dim)
    - 2D: water_level_delta + water_volume (2dim)

    推論時に全補助変数を予測→フィードバック。
    physics lossは使わず素直なMSEのみ。
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
        coupling_edge_dim: int = 0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_processor_layers=num_processor_layers,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            noise_std=noise_std,
            coupling_edge_dim=coupling_edge_dim,
        )
        # Override decoders: output 1 → 2 (wl_delta + auxiliary)
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation),
        })


class HeteroFloodGNNv20(HeteroFloodGNNv11):
    """v20: ba0対策 — upstream_inlet_flow_sum 特徴量追加.

    v11cと同一アーキテクチャだが、1D入力次元を13→14に拡張。
    14番目の特徴量 = 上流隣接ノードのinlet_flow合計。
    ba0ノード (base_area=0) に上流流入量の因果情報を直接提供。

    訓練時はba0ノードのloss weightを上げて使用 (訓練スクリプト側で制御)。
    """

    NODE_DIMS = {"1d": 14, "2d": 18}  # v11cの13→14 (upstream_inlet_flow_sum追加)


class HeteroFloodGNNv21(HeteroFloodGNNv11):
    """v21: Manning equilibrium fill ratio 特徴量追加.

    v11cと同一アーキテクチャだが、1D入力次元を13→14に拡張。
    14番目の特徴量 = Manning's equationで逆算したpipe fill ratio (0~1)。
    edge flow + pipe geometry → 物理的均衡水深を各ノードに提供。
    ba0ノードにも隣接edgeのflowから物理的アンカーを与える。
    """

    NODE_DIMS = {"1d": 14, "2d": 18}  # v11cの13→14 (Manning fill ratio追加)


class HeteroFloodGNNv22(HeteroFloodGNNv11):
    """v22: Full 1D self-attention + separate ba0/ba+ decoder heads.

    Hypotheses 5+9:
    - Self-attention over all 198 1D nodes after GNN → hop limit撤廃 (198²=39K, 軽量)
    - Separate decoders for ba0 (base_area=0) vs ba+ → loss干渉排除

    ba0_mask: [N_1d] bool tensor, True for ba0 nodes. Passed to forward().
    Transfer: v11cの全weight + 1D decoderをba0/ba+両方にコピー。
    """

    def __init__(self, n_attn_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = self.hidden_dim
        activation = kwargs.get("activation", "gelu")
        decoder_layers = kwargs.get("decoder_layers", 2)

        # 1D self-attention (198² = 39K, negligible)
        self.attn_1d = nn.MultiheadAttention(
            hidden_dim, n_attn_heads, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Separate 1D decoders for ba0 and ba+
        self.decoder_1d_ba0 = make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation)
        self.decoder_1d_ba_plus = make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation)

    def forward(self, data: HeteroData, ba0_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process (GNN + edge updates, identical to v11) ===
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])
            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

        # === 1D Self-Attention (NEW: global, hop制限なし) ===
        h_1d = x_dict["1d"].unsqueeze(0)  # [1, N_1d, H]
        attn_out, _ = self.attn_1d(h_1d, h_1d, h_1d)
        x_dict["1d"] = self.attn_norm(x_dict["1d"] + attn_out.squeeze(0))

        # === Decode ===
        out: Dict[str, Tensor] = {}

        # 1D: separate ba0/ba+ heads (NEW)
        # Both decoders process all nodes, select via mask (avoids dtype issues under autocast)
        if ba0_mask is not None:
            out_ba0 = self.decoder_1d_ba0(x_dict["1d"])       # [N, 2]
            out_ba_plus = self.decoder_1d_ba_plus(x_dict["1d"])  # [N, 2]
            out["1d"] = torch.where(ba0_mask.unsqueeze(1), out_ba0, out_ba_plus)
        else:
            out["1d"] = self.node_decoders["1d"](x_dict["1d"])

        out["2d"] = self.node_decoders["2d"](x_dict["2d"])

        # Edge decoder
        pipe_key = ("1d", "pipe", "1d")
        if pipe_key in edge_attr_dict:
            out["1d_edge"] = self.edge_decoders["1d__pipe__1d"](edge_attr_dict[pipe_key])

        return out


# ─── PDE-Refiner ─────────────────────────────────────────────────────
class FloodRefiner(HeteroFloodGNNv4):
    """PDE-Refiner: 軽量GNNで base model の予測を精緻化.

    入力: 通常のグラフ特徴量 + base model の予測 δ_0 を append
      - 1D: 13 (base) + 2 (wl_delta, inlet) = 15
      - 2D: 18 (base) + 1 (wl_delta) = 19
    出力: 補正量 (δ_0 に加算)
      - 1D: [N, 2] (wl_delta_correction, inlet_correction)
      - 2D: [N, 1] (wl_delta_correction)

    hidden_dim=64, num_processor_layers=2 で base の ~1/4 コスト.
    """

    NODE_DIMS = {"1d": 15, "2d": 19}  # base 13/18 + delta prediction 2/1
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 9,
        ("2d", "surface", "2d"): 7,
    }

    def __init__(
        self,
        hidden_dim: int = 64,
        num_processor_layers: int = 2,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.0,
        coupling_edge_dim: int = 0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_processor_layers=num_processor_layers,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            noise_std=noise_std,
            coupling_edge_dim=coupling_edge_dim,
        )
        # v11と同じ出力: 1D=2dim (wl_delta + inlet), 2D=1dim (wl_delta)
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })
        # Zero-init: 初期 correction ≈ 0 → refined ≈ δ_0 (identity start)
        for dec in self.node_decoders.values():
            nn.init.zeros_(dec[-1].weight)
            nn.init.zeros_(dec[-1].bias)
        # Edge decoderは不要 (base modelのedge予測をそのまま使う)


class HeteroFloodGNNv45(HeteroFloodGNNv11):
    """v45: v11 + 3-step temporal context (wl_t, wl_{t-1}, wl_{t-2}).

    NODE_DIMS拡張: prev_t2動的特徴量を末尾追加
      1D: 13→15 (+wl_{t-2}, inlet_{t-2})
      2D: 18→21 (+rain_{t-2}, wl_{t-2}, vol_{t-2})
    EDGE_DIMSはv11と同一 (9, 7)。

    build_graph_at_timestep(prev_t2=t-2) で3ステップ入力を生成。
    v11/v44からweight transfer: encoder first 13/18 dims コピー、残りzero-init。
    """

    NODE_DIMS = {"1d": 15, "2d": 21}  # v11の13/18 + prev_t2の2/3


class HeteroFloodGNNv46(HeteroFloodGNNv11):
    """v46: v11 with 8 layers + JumpingKnowledge.

    全processorレイヤーの中間表現を収集し、concat→projection→decode。
    受容野拡大 (4層→8層) で空間相関 r=0.4 のクラスタ化を解消。
    JK-Net (Xu et al. 2018) のconcat戦略により、ローカル/グローバル情報を両方活用。

    初期化時にprocessor layers 0-3 をコピーして layers 4-7 にも使用可能 (clone_layers)。
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
        coupling_edge_dim: int = 0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_processor_layers=num_processor_layers,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            noise_std=noise_std,
            coupling_edge_dim=coupling_edge_dim,
        )
        self._num_layers = num_processor_layers
        # JK projection: concat all layer outputs → hidden_dim
        jk_in = hidden_dim * num_processor_layers
        self.jk_project = nn.ModuleDict({
            "1d": nn.Linear(jk_in, hidden_dim),
            "2d": nn.Linear(jk_in, hidden_dim),
        })

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        """v11 forward with JumpingKnowledge aggregation."""
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)
        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process with JK collection ===
        jk_states: Dict[str, list] = {"1d": [], "2d": []}
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])
            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)
            # Collect intermediate representations
            for ntype in ["1d", "2d"]:
                jk_states[ntype].append(x_dict[ntype])

        # === JK Aggregation: concat all layers → project ===
        for ntype in ["1d", "2d"]:
            jk_cat = torch.cat(jk_states[ntype], dim=-1)  # [N, H*L]
            x_dict[ntype] = self.jk_project[ntype](jk_cat)  # [N, H]

        # === Decode Nodes ===
        out: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])

        # === Decode 1D Pipe Edges ===
        pipe_key = ("1d", "pipe", "1d")
        if pipe_key in edge_attr_dict:
            out["1d_edge"] = self.edge_decoders["1d__pipe__1d"](edge_attr_dict[pipe_key])

        return out


class HeteroFloodGNNv47(HeteroFloodGNNv11):
    """v47: v11 + テスト時計算可能な物理特徴量.

    特徴量監査に基づく入力情報拡張:
      - 1D storage: base_area * max(0, wl - invert_elev) → 1D node +1 dim
      - elapsed_ratio: step/total_steps → 全ノード +1 dim
      - hydraulic_gradient_norm: (wl_src - wl_dst)/edge_length → 1D edge +1 dim

    NODE_DIMS: 1D 13→15, 2D 18→19
    EDGE_DIMS: pipe 9→10, surface 7 (変更なし)
    """

    NODE_DIMS = {"1d": 15, "2d": 19}
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 10,  # +1: hydraulic_gradient_norm
        ("2d", "surface", "2d"): 7,
    }


class HeteroFloodGNNv47b(HeteroFloodGNNv11):
    """v47b: v11 + hydraulic_gradient_norm のみ.

    1Dエッジに (wl_src - wl_dst)/edge_length を追加。
    NODE_DIMSはv11と同一 (13/18)。
    EDGE_DIMSのみpipe: 9→10。
    """

    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 10,  # +1: hydraulic_gradient_norm
        ("2d", "surface", "2d"): 7,
    }


class HeteroFloodGNNv48(HeteroFloodGNNv4):
    """v48: テスト時ゼロ変数を完全除外.

    v4ベース (1D/2D decoder = 1dim, edge decoder なし):
    - inlet_flow 除外: 1D 13→11
    - water_volume 除外: 2D 18→16
    - edge dynamics (flow, velocity) 除外: pipe 9→7, surface 7→5
    - 1D decoder: 1dim (WL delta のみ, no inlet prediction)
    - edge decoder: なし (flow/velocity 予測なし)
    """

    NODE_DIMS = {"1d": 11, "2d": 16}
    EDGE_DIMS = {
        ("1d", "pipe", "1d"): 7,   # static only
        ("2d", "surface", "2d"): 5,  # static only
    }


class HeteroFloodGNNv11_GAT(HeteroFloodGNNv11):
    """v11 + GATv2Conv for 1D pipe processor — attention-based edge importance.

    1D pipe edges: GATv2Conv (attention learns which neighboring pipe nodes matter)
    All other edges (2D surface, coupling): NodeEdgeConv unchanged

    GATv2Convの利点:
    - 各ノードが「どの隣接パイプからのメッセージが重要か」を学習
    - hub (deg≥3) で効果大: 複数パイプの流れの優先順位を動的に決定
    - edge_dim=hidden_dim: エンコード済みパイプ属性(径,勾配,長さ等)でattention変調
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        num_heads: int = 4,
        **kwargs,
    ):
        kwargs["hidden_dim"] = hidden_dim
        kwargs["num_processor_layers"] = num_processor_layers
        super().__init__(**kwargs)
        self.num_heads = num_heads

        # Replace 1D pipe conv in each processor layer: NodeEdgeConv → GATv2Conv
        for hetero_conv in self.processors:
            pipe_key = ("1d", "pipe", "1d")
            hetero_conv.convs[pipe_key] = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                concat=True,  # output = heads * (hidden_dim // heads) = hidden_dim
                edge_dim=hidden_dim,  # encoded pipe edge features
                add_self_loops=False,  # physical pipes have no self-loops
                dropout=kwargs.get("dropout", 0.1),
                share_weights=False,
            )


class HeteroFloodGNNv11_GRU2(HeteroFloodGNNv11):
    """v11 + 1D専用GRU v2: 改良版 gated residual + LayerNorm.

    v11_GRU との差分:
    - forward() は GNN processor 後に 1D ノードのみ GRUCell 適用
    - gate bias=-2.0 (初期: GNN パススルー 88%, GRU 12%)
    - 2D ノードは GRU を通さない (GNN 出力そのまま)
    - GRU hidden state は 1D ノードのみ (198 nodes × hidden_dim)

    forward signature:
      forward(data, h_1d=None) -> (out_dict, h_1d_new)
      h_1d: [N_1d, hidden_dim] GRU hidden state (None なら自動ゼロ初期化)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = kwargs.get("hidden_dim", 128)

        # 1D専用 GRUCell: processor output を input, 前ステップ h_1d を hidden
        self.gru_1d = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_ln = nn.LayerNorm(hidden_dim)

        # Gated residual: gate = sigmoid(linear([gnn_out, gru_out]))
        # bias=-2.0 → sigmoid(-2)≈0.12 → 初期はほぼ GNN passthrough
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.gate_linear.bias, -2.0)

    def forward(
        self,
        data: HeteroData,
        h_1d: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """GNN forward + 1D GRU temporal update.

        Returns:
            out: {"1d": [N_1d, 2], "2d": [N_2d, 1], "1d_edge": [E_1d, 2]}
            h_1d_new: [N_1d, hidden_dim]
        """
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process (NodeEdgeConv + edge updates) ===
        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])
            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

            # v52b global pooling (inherited, skip if not enabled)
            if self.global_pool_1d:
                global_1d = x_dict["1d"].mean(dim=0, keepdim=True)
                x_dict["1d"] = x_dict["1d"] + self.global_pool_mlps[i](global_1d)

        # === 1D GRU temporal update (1D ノードのみ, 2D はスキップ) ===
        gnn_1d = x_dict["1d"]  # [N_1d, H]
        if h_1d is None:
            h_1d = torch.zeros_like(gnn_1d)
        h_1d_new = self.gru_1d(gnn_1d, h_1d)
        h_1d_new = self.gru_ln(h_1d_new)

        # Gated residual: output = gate * gru + (1-gate) * gnn
        gate = torch.sigmoid(self.gate_linear(torch.cat([gnn_1d, h_1d_new], dim=-1)))
        x_dict["1d"] = gate * h_1d_new + (1.0 - gate) * gnn_1d

        # === Decode Nodes ===
        out: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])

        # === Decode 1D Pipe Edges ===
        pipe_key = ("1d", "pipe", "1d")
        if pipe_key in edge_attr_dict:
            out["1d_edge"] = self.edge_decoders["1d__pipe__1d"](edge_attr_dict[pipe_key])

        return out, h_1d_new


class HeteroFloodGNNv11_DualDec(HeteroFloodGNNv11):
    """v93: Dual-Decoder for regime-conditional prediction.

    Encoder + Processor are shared (transferred from v76).
    Two decoder sets:
      - decoder_rain: used when rain > 0 (same as v76 decoder)
      - decoder_recess: used when rain = 0 (recession specialist)
    Hard switch based on rain input at each step.

    decoder_recess is initialized as copy of decoder_rain,
    then fine-tuned on recession-only steps.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = kwargs.get("hidden_dim", 128)
        decoder_layers = kwargs.get("decoder_layers", 2)
        activation = kwargs.get("activation", "gelu")

        # Recession decoder (separate MLP, same architecture as rain decoder)
        self.node_decoders_recess = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, 1, hidden_dim, decoder_layers, activation),
        })
        self.edge_decoders_recess = nn.ModuleDict({
            "1d__pipe__1d": make_mlp(hidden_dim, 2, hidden_dim, decoder_layers, activation),
        })

    def init_recess_from_rain(self):
        """Copy rain decoder weights to recession decoder as initialization."""
        for ntype in ["1d", "2d"]:
            self.node_decoders_recess[ntype].load_state_dict(
                self.node_decoders[ntype].state_dict())
        self.edge_decoders_recess["1d__pipe__1d"].load_state_dict(
            self.edge_decoders["1d__pipe__1d"].state_dict())

    def forward(self, data: HeteroData, rain_flag: bool = True) -> Dict[str, Tensor]:
        """Forward with regime-conditional decoder.

        Args:
            data: HeteroData graph
            rain_flag: True=rain decoder, False=recession decoder
        """
        # === Encode + Process (shared, identical to v11) ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        for i, conv in enumerate(self.processors):
            x_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for ntype in x_dict:
                if ntype in x_new:
                    x_dict[ntype] = x_dict[ntype] + x_new[ntype]
                    if self.norms is not None:
                        x_dict[ntype] = self.norms[i][ntype](x_dict[ntype])
            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

            if self.global_pool_1d:
                global_1d = x_dict["1d"].mean(dim=0, keepdim=True)
                x_dict["1d"] = x_dict["1d"] + self.global_pool_mlps[i](global_1d)

        # === Regime-conditional decode ===
        if rain_flag:
            node_dec = self.node_decoders
            edge_dec = self.edge_decoders
        else:
            node_dec = self.node_decoders_recess
            edge_dec = self.edge_decoders_recess

        out: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = node_dec[ntype](x_dict[ntype])

        pipe_key = ("1d", "pipe", "1d")
        if pipe_key in edge_attr_dict:
            out["1d_edge"] = edge_dec["1d__pipe__1d"](edge_attr_dict[pipe_key])

        return out


class HeteroFloodGPS(HeteroFloodGNNv11):
    """v94: Graph Transformer (GPS) processor replacing NodeEdgeConv.

    Architecture Break hypothesis: NodeEdgeConv processor (4-hop MP) is the
    ceiling. GPS adds global multi-head attention so degree-1 nodes can
    access the entire graph. Encoder + decoder reused from v76.

    Design:
      - 1D nodes: GPS layer (local MPNN + global attention on 198 nodes)
      - 2D nodes: GPS layer (local MPNN + global attention on ~4300 nodes)
      - Coupling: standard NodeEdgeConv (cross-type MP, kept from v76)
      - Edge updates: same as v76
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_processor_layers: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        noise_std: float = 0.02,
        coupling_edge_dim: int = 0,
        gps_heads: int = 4,
        gps_dropout: float = 0.1,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_processor_layers=num_processor_layers,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            noise_std=noise_std,
            coupling_edge_dim=coupling_edge_dim,
        )
        from torch_geometric.nn import GPSConv

        # Replace processors with GPS layers (per node type)
        # GPS = local_conv + global_attention + FFN
        self.gps_1d = nn.ModuleList()
        self.gps_2d = nn.ModuleList()

        for i in range(num_processor_layers):
            # 1D GPS: local conv = pipe MPNN
            local_1d = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            self.gps_1d.append(GPSConv(
                channels=hidden_dim,
                conv=local_1d,
                heads=gps_heads,
                dropout=gps_dropout,
                act=activation,
                norm="layer_norm",
            ))

            # 2D GPS: local conv = surface MPNN
            local_2d = NodeEdgeConv(
                hidden_dim, hidden_dim, edge_dim=hidden_dim,
                hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                aggr="sum",
            )
            self.gps_2d.append(GPSConv(
                channels=hidden_dim,
                conv=local_2d,
                heads=gps_heads,
                dropout=gps_dropout,
                act=activation,
                norm="layer_norm",
            ))

        # Coupling convs: keep standard NodeEdgeConv (cross-type, no self-attention)
        coupling_conv_edim = hidden_dim if coupling_edge_dim > 0 else 0
        self.coupling_convs = nn.ModuleList()
        self.coupling_norms = nn.ModuleList()
        for _ in range(num_processor_layers):
            coupling = nn.ModuleDict({
                "1d_to_2d": NodeEdgeConv(
                    hidden_dim, hidden_dim, edge_dim=coupling_conv_edim,
                    hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                    aggr="sum",
                ),
                "2d_to_1d": NodeEdgeConv(
                    hidden_dim, hidden_dim, edge_dim=coupling_conv_edim,
                    hidden_dim=hidden_dim, activation=activation, dropout=dropout,
                    aggr="sum",
                ),
            })
            self.coupling_convs.append(coupling)
            self.coupling_norms.append(nn.ModuleDict({
                "1d": LayerNorm(hidden_dim),
                "2d": LayerNorm(hidden_dim),
            }))

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        # === Encode Nodes ===
        x_dict: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            x = data[ntype].x
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x_dict[ntype] = self.node_encoders[ntype](x)

        # === Encode Edges ===
        edge_attr_dict: Dict[tuple, Tensor] = {}
        for etype in self.EDGE_DIMS:
            key = "__".join(etype)
            if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)
        if self.coupling_edge_dim > 0:
            for etype in [("1d", "coupling", "2d"), ("2d", "coupling", "1d")]:
                key = "__".join(etype)
                if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
                    edge_attr_dict[etype] = self.edge_encoders[key](data[etype].edge_attr)

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        # === Process: GPS layers + coupling ===
        for i in range(len(self.gps_1d)):
            # 1D GPS (local pipe MPNN + global attention)
            pipe_key = ("1d", "pipe", "1d")
            pipe_ei = edge_index_dict.get(pipe_key)
            pipe_ea = edge_attr_dict.get(pipe_key)
            x_1d_new = self.gps_1d[i](
                x_dict["1d"], pipe_ei,
                batch=None,  # single graph
                edge_attr=pipe_ea,
            )

            # 2D GPS (local surface MPNN + global attention)
            surf_key = ("2d", "surface", "2d")
            surf_ei = edge_index_dict.get(surf_key)
            surf_ea = edge_attr_dict.get(surf_key)
            x_2d_new = self.gps_2d[i](
                x_dict["2d"], surf_ei,
                batch=None,
                edge_attr=surf_ea,
            )

            # Coupling (1D <-> 2D cross-message passing)
            coup_1d_2d = ("1d", "coupling", "2d")
            coup_2d_1d = ("2d", "coupling", "1d")

            # 2D receives from 1D
            if coup_1d_2d in edge_index_dict:
                coup_ea_12 = edge_attr_dict.get(coup_1d_2d)
                msg_to_2d = self.coupling_convs[i]["1d_to_2d"](
                    (x_dict["1d"], x_dict["2d"]),
                    edge_index_dict[coup_1d_2d],
                    edge_attr=coup_ea_12,
                )
                x_2d_new = x_2d_new + msg_to_2d

            # 1D receives from 2D
            if coup_2d_1d in edge_index_dict:
                coup_ea_21 = edge_attr_dict.get(coup_2d_1d)
                msg_to_1d = self.coupling_convs[i]["2d_to_1d"](
                    (x_dict["2d"], x_dict["1d"]),
                    edge_index_dict[coup_2d_1d],
                    edge_attr=coup_ea_21,
                )
                x_1d_new = x_1d_new + msg_to_1d

            # GPS already has internal residual + norm, coupling adds residual
            x_dict["1d"] = self.coupling_norms[i]["1d"](x_1d_new)
            x_dict["2d"] = self.coupling_norms[i]["2d"](x_2d_new)

            # Edge updates (same as v76)
            for etype_key, update_mlp in self.edge_updates[i].items():
                etype = tuple(etype_key.split("__"))
                if etype in edge_attr_dict:
                    src_type, _, tgt_type = etype
                    ei = edge_index_dict[etype]
                    x_src = x_dict[src_type][ei[0]]
                    x_tgt = x_dict[tgt_type][ei[1]]
                    old_edge = edge_attr_dict[etype]
                    edge_input = torch.cat([x_src, old_edge, x_tgt], dim=-1)
                    edge_attr_dict[etype] = old_edge + update_mlp(edge_input)

        # === Decode Nodes ===
        out: Dict[str, Tensor] = {}
        for ntype in ["1d", "2d"]:
            out[ntype] = self.node_decoders[ntype](x_dict[ntype])

        # === Decode 1D Pipe Edges ===
        pipe_key = ("1d", "pipe", "1d")
        if pipe_key in edge_attr_dict:
            out["1d_edge"] = self.edge_decoders["1d__pipe__1d"](edge_attr_dict[pipe_key])

        return out


class HeteroFloodGNNv97(HeteroFloodGNNv11):
    """v97: Multi-step (K=5) + Larger (256/6) + Physics-aware.

    Combines:
      1. Multi-step prediction head (K sub-steps per forward pass)
      2. Larger model capacity (hidden_dim=256, num_processor_layers=6)
      3. Compatible with physics-informed loss (1D mass conservation)

    Output layout (same as v11_TB, interleaved):
      1D:      [wl_delta_0, inlet_0, ..., wl_delta_{K-1}, inlet_{K-1}]  -> [N, K*2]
      2D:      [wl_delta_0, ..., wl_delta_{K-1}]                        -> [N, K]
      1D_edge: [flow_0, vel_0, ..., flow_{K-1}, vel_{K-1}]              -> [E, K*2]
    """

    def __init__(self, K: int = 5, **kwargs):
        # Default to larger model if not overridden
        kwargs.setdefault("hidden_dim", 256)
        kwargs.setdefault("num_processor_layers", 6)
        super().__init__(**kwargs)
        self.K = K
        hidden_dim = kwargs.get("hidden_dim", 256)
        decoder_layers = kwargs.get("decoder_layers", 2)
        activation = kwargs.get("activation", "gelu")
        # Override decoders: K * original output dims
        self.node_decoders = nn.ModuleDict({
            "1d": make_mlp(hidden_dim, K * 2, hidden_dim, decoder_layers, activation),
            "2d": make_mlp(hidden_dim, K * 1, hidden_dim, decoder_layers, activation),
        })
        self.edge_decoders = nn.ModuleDict({
            "1d__pipe__1d": make_mlp(hidden_dim, K * 2, hidden_dim, decoder_layers, activation),
        })
