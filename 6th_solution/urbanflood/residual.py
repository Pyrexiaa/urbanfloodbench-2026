# #####
# residual.py
# #####

# urbanflood/residual.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualGRUConfig:
    dyn_dim: int
    static_dim: int
    warm_ctx_dim: int = 0
    warm_seq_dim: int = 0

    node_emb_dim: int = 8
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    # If True, use a per-node linear readout head instead of a shared Linear.
    # This is practical for small node counts (e.g., Model 2 1D), but can be
    # memory-heavy for large graphs (e.g., 2D nodes).
    nodewise_head: bool = False
    expert_heads: int = 1
    expert_gate_hidden: int = 0
    expert_gate_dropout: float = 0.0

    # Optional learned graph mixing on the node dimension (uses edge_index).
    # 0 disables. When enabled, forward() must be called with edge_index.
    graph_mix_k: int = 0
    graph_mix_dropout: float = 0.0
    # Optional learned graph mixing applied *after* the GRU on the hidden sequence.
    # 0 disables. When enabled, forward() must be called with edge_index.
    graph_mix_post_k: int = 0

    # Clamp residual output in feet; 0 disables.
    clamp: float = 0.0
    # Clamp mode for residual output:
    # - "hard": torch.clamp (piecewise-constant gradients outside the range)
    # - "tanh": smooth clamp via clamp * tanh(x / clamp)
    clamp_mode: str = "hard"
    # If True, initialize the residual readout to exactly zero so training starts
    # from the incoming base trajectory instead of a random residual offset.
    zero_init_out: bool = False


def _neighbor_mean_time_major_torch(
    x: torch.Tensor,  # [T, N, H]
    *,
    edge_index: torch.LongTensor,  # [2, E]
    deg_inv: torch.Tensor | None = None,  # [N]
    edge_weight: torch.Tensor | None = None,  # [E]
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("x must be [T, N, H]")
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("edge_index must be [2, E]")

    T, N, H = x.shape
    src = edge_index[0]
    dst = edge_index[1]
    if src.ndim != 1 or dst.ndim != 1 or int(src.shape[0]) != int(dst.shape[0]):
        raise ValueError("edge_index must be [2, E] with 1D rows")

    out = torch.zeros((T, N, H), device=x.device, dtype=x.dtype)
    if edge_weight is None:
        out.index_add_(1, dst, x.index_select(1, src))
    else:
        if edge_weight.ndim != 1 or int(edge_weight.shape[0]) != int(src.shape[0]):
            raise ValueError("edge_weight must be [E] matching edge_index")
        w = edge_weight.to(device=x.device, dtype=x.dtype)
        out.index_add_(1, dst, x.index_select(1, src) * w[None, :, None])

    if deg_inv is None:
        if edge_weight is None:
            deg = torch.bincount(dst, minlength=int(N)).clamp(min=1).to(torch.float32)
        else:
            deg = torch.zeros((int(N),), device=x.device, dtype=torch.float32)
            deg.index_add_(0, dst.to(device=x.device), edge_weight.to(device=x.device, dtype=torch.float32))
            deg = deg.clamp(min=1.0)
        deg_inv = (1.0 / deg).to(device=x.device)
    else:
        if deg_inv.ndim != 1 or int(deg_inv.shape[0]) != int(N):
            raise ValueError("deg_inv must be [N]")
        deg_inv = deg_inv.to(device=x.device, dtype=torch.float32)

    out = out * deg_inv.to(dtype=x.dtype)[None, :, None]
    return out


class _GraphMixBlock(nn.Module):
    def __init__(self, *, dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(int(dim))
        self.self_lin = nn.Linear(int(dim), int(dim), bias=False)
        self.nbr_lin = nn.Linear(int(dim), int(dim), bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        edge_index: torch.LongTensor,
        deg_inv: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        x0 = self.ln(x)
        nbr = _neighbor_mean_time_major_torch(x0, edge_index=edge_index, deg_inv=deg_inv, edge_weight=edge_weight)
        y = self.self_lin(x0) + self.nbr_lin(nbr)
        y = self.act(y)
        y = self.drop(y)
        return x + y


class ResidualNodeGRU(nn.Module):
    """
    Sequence model that treats nodes as the batch dimension:
      input:  x_dyn[T, N, dyn_dim]
              node_static[N, static_dim]
      output: r_hat[T, N]  (residual correction)

    This is designed for stacking on top of a strong baseline without any
    autoregressive feedback of the corrected signal.
    """

    def __init__(self, *, n_nodes: int, cfg: ResidualGRUConfig):
        super().__init__()
        self.n_nodes = int(n_nodes)
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")
        self.cfg = cfg

        if cfg.dyn_dim < 1 or cfg.static_dim < 0:
            raise ValueError("invalid dyn/static dims")
        if int(cfg.warm_ctx_dim) < 0:
            raise ValueError("warm_ctx_dim must be >= 0")
        if int(cfg.warm_seq_dim) < 0:
            raise ValueError("warm_seq_dim must be >= 0")
        if cfg.node_emb_dim < 0:
            raise ValueError("node_emb_dim must be >= 0")
        if cfg.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if cfg.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if cfg.dropout < 0.0 or cfg.dropout > 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if int(cfg.expert_heads) < 1:
            raise ValueError("expert_heads must be >= 1")
        if int(cfg.expert_gate_hidden) < 0:
            raise ValueError("expert_gate_hidden must be >= 0")
        if float(cfg.expert_gate_dropout) < 0.0 or float(cfg.expert_gate_dropout) > 1.0:
            raise ValueError("expert_gate_dropout must be in [0, 1]")

        self.node_emb = nn.Embedding(self.n_nodes, int(cfg.node_emb_dim)) if int(cfg.node_emb_dim) > 0 else None

        in_dim = int(cfg.dyn_dim + cfg.static_dim + cfg.node_emb_dim)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(cfg.hidden_dim)),
            nn.GELU(),
        )
        self.h0_proj = None
        if int(cfg.warm_ctx_dim) > 0:
            out_h0 = int(cfg.hidden_dim) * int(cfg.num_layers) * (2 if bool(cfg.bidirectional) else 1)
            self.h0_proj = nn.Sequential(
                nn.LayerNorm(int(cfg.warm_ctx_dim)),
                nn.Linear(int(cfg.warm_ctx_dim), out_h0),
                nn.Tanh(),
            )
        self.warm_seq_proj = None
        self.warm_seq_encoder = None
        if int(cfg.warm_seq_dim) > 0:
            self.warm_seq_proj = nn.Sequential(
                nn.LayerNorm(int(cfg.warm_seq_dim)),
                nn.Linear(int(cfg.warm_seq_dim), int(cfg.hidden_dim)),
                nn.GELU(),
            )
            self.warm_seq_encoder = nn.GRU(
                input_size=int(cfg.hidden_dim),
                hidden_size=int(cfg.hidden_dim),
                num_layers=int(cfg.num_layers),
                dropout=float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0,
                batch_first=False,
                bidirectional=bool(cfg.bidirectional),
            )

        k_mix = int(cfg.graph_mix_k)
        if k_mix < 0:
            raise ValueError("graph_mix_k must be >= 0")
        mix_drop = float(cfg.graph_mix_dropout)
        if mix_drop < 0.0 or mix_drop > 1.0:
            raise ValueError("graph_mix_dropout must be in [0, 1]")
        self.graph_mix = nn.ModuleList([_GraphMixBlock(dim=int(cfg.hidden_dim), dropout=mix_drop) for _ in range(k_mix)])

        self.gru = nn.GRU(
            input_size=int(cfg.hidden_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0,
            batch_first=False,
            bidirectional=bool(cfg.bidirectional),
        )
        out_hidden = int(cfg.hidden_dim) * (2 if bool(cfg.bidirectional) else 1)
        self.expert_heads = int(cfg.expert_heads)

        post_k = int(getattr(cfg, "graph_mix_post_k", 0) or 0)
        if post_k < 0:
            raise ValueError("graph_mix_post_k must be >= 0")
        self.graph_mix_post = nn.ModuleList([_GraphMixBlock(dim=out_hidden, dropout=mix_drop) for _ in range(post_k)])

        if self.expert_heads > 1:
            gate_hidden = int(cfg.expert_gate_hidden)
            gate_drop = float(cfg.expert_gate_dropout)
            if gate_hidden > 0:
                self.out_gate = nn.Sequential(
                    nn.LayerNorm(out_hidden),
                    nn.Linear(out_hidden, gate_hidden),
                    nn.GELU(),
                    nn.Dropout(gate_drop) if gate_drop > 0.0 else nn.Identity(),
                    nn.Linear(gate_hidden, self.expert_heads),
                )
            else:
                self.out_gate = nn.Linear(out_hidden, self.expert_heads)
        else:
            self.out_gate = None

        if bool(cfg.nodewise_head):
            self.out_w = nn.Embedding(self.n_nodes, out_hidden * self.expert_heads)
            self.out_b = nn.Embedding(self.n_nodes, self.expert_heads)
            nn.init.normal_(self.out_w.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.out_b.weight)
            self.out = None
        else:
            self.out = nn.Linear(out_hidden, self.expert_heads)
            self.out_w = None
            self.out_b = None
        if bool(cfg.zero_init_out):
            if self.out is not None:
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)
            if self.out_w is not None:
                nn.init.zeros_(self.out_w.weight)
            if self.out_b is not None:
                nn.init.zeros_(self.out_b.weight)

    def forward(
        self,
        x_dyn: torch.Tensor,
        node_static: torch.Tensor,
        h0: torch.Tensor | None = None,
        warm_ctx: torch.Tensor | None = None,
        warm_seq: torch.Tensor | None = None,
        *,
        node_ids: torch.LongTensor | None = None,
        expert_group_idx: torch.Tensor | None = None,
        edge_index: torch.LongTensor | None = None,
        edge_deg_inv: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        return_h: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x_dyn: [T, N, dyn_dim]
          node_static: [N, static_dim]
          h0: optional initial hidden state for truncated BPTT, shape [L*(dir), N, hidden]
          node_ids: optional global node indices for the N nodes in this call. Enables node-chunked forward passes.
          return_h: if True, also return the final hidden state h_T
        Returns:
          residual_hat: [T, N]
        """
        if x_dyn.ndim != 3:
            raise ValueError("x_dyn must be [T, N, dyn_dim]")
        T, N, D = x_dyn.shape
        if int(D) != int(self.cfg.dyn_dim):
            raise ValueError(f"x_dyn dyn_dim mismatch: got {D}, expected {self.cfg.dyn_dim}")

        if node_ids is None:
            if N != self.n_nodes:
                raise ValueError(f"x_dyn N mismatch: got {N}, expected {self.n_nodes}")
            node_ids2 = torch.arange(self.n_nodes, device=x_dyn.device)
        else:
            if node_ids.ndim != 1 or int(node_ids.shape[0]) != int(N):
                raise ValueError("node_ids must be [N]")
            node_ids2 = node_ids.to(device=x_dyn.device, dtype=torch.long)
            if int(node_ids2.min().item()) < 0 or int(node_ids2.max().item()) >= int(self.n_nodes):
                raise ValueError("node_ids out of range")
            if len(self.graph_mix):
                raise ValueError("graph_mix is not supported with node_ids (chunked forward)")
            if len(self.graph_mix_post):
                raise ValueError("graph_mix_post is not supported with node_ids (chunked forward)")

        if node_static.ndim != 2 or int(node_static.shape[0]) != int(N) or int(node_static.shape[1]) != int(self.cfg.static_dim):
            raise ValueError("node_static must be [N, static_dim]")
        if warm_ctx is not None:
            if int(self.cfg.warm_ctx_dim) <= 0 or self.h0_proj is None:
                raise ValueError("warm_ctx provided but model has no warm_ctx_dim")
            if warm_ctx.ndim != 2 or int(warm_ctx.shape[0]) != int(N) or int(warm_ctx.shape[1]) != int(self.cfg.warm_ctx_dim):
                raise ValueError("warm_ctx must be [N, warm_ctx_dim]")
            if h0 is not None or warm_seq is not None:
                raise ValueError("pass only one of h0, warm_ctx, or warm_seq")
        if warm_seq is not None:
            if int(self.cfg.warm_seq_dim) <= 0 or self.warm_seq_proj is None or self.warm_seq_encoder is None:
                raise ValueError("warm_seq provided but model has no warm_seq_dim")
            if warm_seq.ndim != 3 or int(warm_seq.shape[1]) != int(N) or int(warm_seq.shape[2]) != int(self.cfg.warm_seq_dim):
                raise ValueError("warm_seq must be [W, N, warm_seq_dim]")
            if h0 is not None or warm_ctx is not None:
                raise ValueError("pass only one of h0, warm_ctx, or warm_seq")

        if self.node_emb is None:
            node_feat = node_static
        else:
            emb = self.node_emb(node_ids2)  # [N, node_emb_dim]
            node_feat = torch.cat([node_static, emb], dim=-1)

        # Expand static node features across time.
        node_feat_t = node_feat.unsqueeze(0).expand(T, -1, -1)  # [T, N, static+emb]
        x = torch.cat([x_dyn, node_feat_t], dim=-1)  # [T, N, in_dim]
        x = self.in_proj(x)  # [T, N, hidden]

        need_edges = bool(len(self.graph_mix) or len(self.graph_mix_post))
        if need_edges and edge_index is None:
            raise ValueError("edge_index is required when graph mixing is enabled")
        if len(self.graph_mix):
            for blk in self.graph_mix:
                x = blk(x, edge_index=edge_index, deg_inv=edge_deg_inv, edge_weight=edge_weight)

        if warm_ctx is not None:
            dirs = 2 if bool(self.cfg.bidirectional) else 1
            Ld = int(self.cfg.num_layers) * dirs
            h0 = self.h0_proj(warm_ctx).view(N, Ld, int(self.cfg.hidden_dim)).permute(1, 0, 2).contiguous()
        elif warm_seq is not None:
            warm_x = self.warm_seq_proj(warm_seq)
            _, h0 = self.warm_seq_encoder(warm_x)

        h_seq, h_T = self.gru(x, h0)  # [T, N, hidden], [L*(dir), N, hidden]
        if len(self.graph_mix_post):
            for blk in self.graph_mix_post:
                h_seq = blk(h_seq, edge_index=edge_index, deg_inv=edge_deg_inv, edge_weight=edge_weight)
        if self.out is not None:
            r_heads = self.out(h_seq)  # [T, N, E]
        else:
            # Per-node linear head: r[t,n,e] = <h[t,n], w[n,e]> + b[n,e]
            if self.out_w is None or self.out_b is None:
                raise RuntimeError("internal error: nodewise_head enabled but out_w/out_b not set")
            w = self.out_w(node_ids2).view(N, self.expert_heads, h_seq.shape[-1])  # [N, E, H]
            b = self.out_b(node_ids2)  # [N, E]
            r_heads = torch.einsum("tnh,neh->tne", h_seq, w) + b.unsqueeze(0)  # [T, N, E]

        if self.expert_heads > 1:
            if expert_group_idx is not None:
                if expert_group_idx.ndim != 1 or int(expert_group_idx.shape[0]) != int(N):
                    raise ValueError("expert_group_idx must be [N]")
                expert_group_idx = expert_group_idx.to(device=x_dyn.device, dtype=torch.long)
                if int(expert_group_idx.min().item()) < 0 or int(expert_group_idx.max().item()) >= int(self.expert_heads):
                    raise ValueError("expert_group_idx out of range for expert_heads")
                gather_idx = expert_group_idx.view(1, N, 1).expand(T, -1, 1)
                r = torch.gather(r_heads, dim=-1, index=gather_idx).squeeze(-1)
            else:
                if self.out_gate is None:
                    raise RuntimeError("internal error: expert_heads enabled but out_gate not set")
                gate_logits = self.out_gate(h_seq)  # [T, N, E]
                gate = torch.softmax(gate_logits, dim=-1)
                r = (r_heads * gate).sum(dim=-1)
        else:
            r = r_heads.squeeze(-1)

        clamp = float(self.cfg.clamp)
        if clamp > 0.0:
            mode = str(self.cfg.clamp_mode)
            if mode == "hard":
                r = torch.clamp(r, -clamp, clamp)
            elif mode == "tanh":
                # Smooth clamp: avoids dead gradients for tiny clamp values (useful for booster stages).
                r = clamp * torch.tanh(r / clamp)
            else:
                raise ValueError(f"unknown clamp_mode: {mode}")
        if return_h:
            return r, h_T
        return r
