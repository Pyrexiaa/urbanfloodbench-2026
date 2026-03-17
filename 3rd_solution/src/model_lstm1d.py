"""FloodLSTM1D: 1D専用 LSTM モデル.

198ノードのパイプネットワーク向け。ノード共有LSTMで時系列パターンを学習し、
近傍集約で空間結合を取り込む。GNNの1D予測を置き換える。

Architecture:
  Input(~12) → Encoder(hidden) → LSTMCell×num_layers → Decoder → (wl_delta, inlet)
"""
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Optional


class FloodLSTM1D(nn.Module):
    """1D パイプネットワーク専用 LSTM.

    全ノードで重みを共有する LSTMCell を使用し、各タイムステップで:
    1. ノード特徴量 + 近傍集約 + 降雨 → encoder
    2. LSTMCell × num_layers (LayerNorm + residual)
    3. decoder → wl_delta (per-node normalized), inlet_pred
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: input → hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stacked LSTMCells with LayerNorm
        self.cells = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            self.cells.append(nn.LSTMCell(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

        # Decoder: hidden → (wl_delta, inlet)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def init_hidden(self, n_nodes: int, device: str):
        """Zero-initialize hidden/cell states for all layers."""
        h = [torch.zeros(n_nodes, self.hidden_dim, device=device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(n_nodes, self.hidden_dim, device=device)
             for _ in range(self.num_layers)]
        return h, c

    def step(
        self,
        x: torch.Tensor,
        h_list: list[torch.Tensor],
        c_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """1タイムステップ処理.

        Args:
            x: [N, input_dim]
            h_list: num_layers × [N, hidden_dim]
            c_list: num_layers × [N, hidden_dim]

        Returns:
            out: [N, 2] (wl_delta_norm, inlet_pred)
            h_new, c_new
        """
        inp = self.encoder(x)  # [N, H]

        h_new, c_new = [], []
        for i, (cell, ln) in enumerate(zip(self.cells, self.layer_norms)):
            h_i, c_i = cell(inp, (h_list[i], c_list[i]))
            h_i = ln(h_i)
            # Residual connection (skip first layer)
            if i > 0:
                h_i = h_i + inp
            if i < self.num_layers - 1:
                h_i = self.drop(h_i)
            inp = h_i
            h_new.append(h_i)
            c_new.append(c_i)

        out = self.decoder(inp)  # [N, 2]
        return out, h_new, c_new


def build_adjacency(edge_index: np.ndarray, num_nodes: int, max_neighbors: int = 8):
    """エッジインデックスからパディング済み隣接リストを構築.

    Returns:
        neighbor_idx: [N, max_neighbors] int64 (パディング=0)
        neighbor_mask: [N, max_neighbors] bool
        degree: [N] int
    """
    adj = defaultdict(list)
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        adj[src].append(dst)
        adj[dst].append(src)

    neighbor_idx = np.zeros((num_nodes, max_neighbors), dtype=np.int64)
    neighbor_mask = np.zeros((num_nodes, max_neighbors), dtype=bool)
    degree = np.zeros(num_nodes, dtype=np.int64)

    for node in range(num_nodes):
        nbrs = adj[node]
        degree[node] = len(nbrs)
        for j, nb in enumerate(nbrs[:max_neighbors]):
            neighbor_idx[node, j] = nb
            neighbor_mask[node, j] = True

    return neighbor_idx, neighbor_mask, degree


def build_rain_map(connections_1d2d: np.ndarray, num_1d_nodes: int) -> np.ndarray:
    """1D→2Dノード接続マップ. 未接続ノードは-1.

    Returns:
        rain_map: [N_1d] int64, index into 2D node array
    """
    rain_map = np.full(num_1d_nodes, -1, dtype=np.int64)
    for row in connections_1d2d:
        node_1d, node_2d = int(row[0]), int(row[1])
        rain_map[node_1d] = node_2d
    return rain_map
