"""
Attention-BiLSTM Neural Network (PyTorch)
─────────────────────────────────────────
Architecture:
  • BiLSTM encoder  : 2 layers, hidden_size=64, bidirectional → output dim 128
  • Additive attention : learned soft spotlight over all timestep hidden states
  • Dropout(0.3)    : regularisation before the output projection
  • Linear(128 → 1) : scalar kWh-per-timestep prediction

Input shape : (batch, seq_len, n_features)
Output shape: (batch, 1)

This is a drop-in replacement for NARXNet — the interface (forward returns
(batch, 1)) is identical so the existing IF+CUSUM pipeline is unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class AttentionBiLSTM(nn.Module):
    """
    Bidirectional LSTM with additive (Bahdanau-style) attention.

    Parameters
    ----------
    n_features  : number of exogenous input features per timestep
    seq_len     : number of past timesteps in the input window
    hidden_size : LSTM hidden units per direction  (total = hidden_size × 2)
    num_layers  : number of stacked BiLSTM layers
    dropout     : dropout rate before the output linear layer
    """

    def __init__(
        self,
        n_features:  int,
        seq_len:     int   = 4,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.n_features  = n_features
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Residual Projection ────────────────────────────────────
        self.input_proj = nn.Linear(n_features, hidden_size * 2)

        # ── Encoder ───────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # ── Additive attention ────────────────────────────────────
        self.attn = nn.Linear(hidden_size * 2, 1, bias=False)

        # ── Output head ───────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x               : (batch, seq_len, n_features)
        return_attention: if True, also return the attention weight tensor

        Returns
        -------
        out     : (batch, 1)
        weights : (batch, seq_len)  — only if return_attention=True
        """
        # BiLSTM: hidden states for every timestep
        enc, _ = self.bilstm(x)                   # (batch, seq_len, hidden*2)

        # Residual connection
        res = self.input_proj(x)
        enc = self.layer_norm(enc + res)

        # Attention scores → weights
        scores  = self.attn(enc)                   # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)     # (batch, seq_len, 1)

        # Weighted context vector
        context = (weights * enc).sum(dim=1)       # (batch, hidden*2)

        out = self.fc(self.dropout(context))       # (batch, 1)

        if return_attention:
            return out, weights.squeeze(-1)        # (batch, seq_len)
        return out

    # ──────────────────────────────────────────────────────────────
    # Closed-loop (autoregressive) multi-step prediction
    # ──────────────────────────────────────────────────────────────
    def closed_loop_predict(
        self,
        X_seq:  np.ndarray,
        y_seed: np.ndarray,
        seq_len: int = 4,
        device:  Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Autoregressive multi-step prediction.

        Parameters
        ----------
        X_seq   : (N, n_features) — scaled exogenous inputs
        y_seed  : (seq_len,)       — initial y values for priming
        seq_len : window length
        device  : torch device

        Returns
        -------
        y_pred : (N - seq_len,)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        n = len(X_seq)
        predictions = []
        past_X = list(X_seq[:seq_len])   # prime with first seq_len exog rows

        with torch.no_grad():
            for t in range(seq_len, n):
                window = np.stack(past_X[-seq_len:], axis=0)       # (seq_len, n_feat)
                inp = torch.tensor(
                    window[np.newaxis, ...], dtype=torch.float32, device=device
                )  # (1, seq_len, n_features)
                y_hat = self.forward(inp).item()
                predictions.append(y_hat)
                past_X.append(X_seq[t])

        return np.array(predictions, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SEQ_LEN    = 4
    N_FEATURES = 15
    BATCH      = 8

    model = AttentionBiLSTM(n_features=N_FEATURES, seq_len=SEQ_LEN)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
    out, attn = model(dummy, return_attention=True)
    print(f"Output shape   : {out.shape}")    # (8, 1)
    print(f"Attention shape: {attn.shape}")   # (8, 4)
    print(f"Attn sum ≈ 1   : {attn.sum(dim=1)}")  # should be all ~1.0
