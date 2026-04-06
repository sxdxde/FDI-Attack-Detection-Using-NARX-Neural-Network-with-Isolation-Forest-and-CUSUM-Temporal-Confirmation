"""
NARX Neural Network (PyTorch)
─────────────────────────────
Architecture (matching the paper):
  • 1 hidden layer, 10 neurons, sigmoid activation
  • Linear output neuron
  • Input size = n_exog_features × mx + my

Open-loop (series-parallel) forward pass:
  Takes pre-assembled delay vector → standard MLP forward.

Closed-loop forward pass:
  Receives seed y values and raw exogenous sequence; auto-regressively
  feeds back its own predictions as the "output delay" inputs.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class NARXNet(nn.Module):
    """
    Single-hidden-layer NARX network with sigmoid activation.

    Parameters
    ----------
    input_size : int
        Total number of inputs = n_features * mx + my
    hidden_size : int
        Number of hidden neurons (10 per paper)
    """

    def __init__(self, input_size: int, hidden_size: int = 10):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),   # linear output unit
        )

        # Weight initialisation: small random weights (Nguyen-Widrow-like)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Open-loop forward pass.

        Parameters
        ----------
        x : (batch, input_size)  — pre-assembled NARX input vector

        Returns
        -------
        out : (batch, 1)
        """
        return self.net(x)

    # ──────────────────────────────────────────────────────────────
    # Closed-loop (autonomous) inference
    # ──────────────────────────────────────────────────────────────
    def closed_loop_predict(
        self,
        X_seq: np.ndarray,
        y_seed: np.ndarray,
        mx: int = 2,
        my: int = 2,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Closed-loop (recurrent) multi-step prediction.

        Parameters
        ----------
        X_seq  : (N, n_features)  — scaled exogenous inputs [u(0)…u(N-1)]
        y_seed : (my,)            — initial true y values for priming the loop
        mx     : exogenous delay order
        my     : output delay order
        device : torch device

        Returns
        -------
        y_pred : (N - max(mx, my),)  — predictions in scaled space
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        n_delay = max(mx, my)
        n = len(X_seq)
        predictions = []

        # Circular buffer for past y (primed with seed values)
        past_y = list(y_seed[-my:])   # most-recent first after reversal below

        with torch.no_grad():
            for t in range(n_delay, n):
                exog_window = X_seq[t - mx + 1 : t + 1][::-1].flatten()  # (n_features*mx,)
                past_y_arr  = np.array(past_y[-my:][::-1], dtype=np.float32)  # (my,)
                inp = torch.tensor(
                    np.concatenate([exog_window, past_y_arr]),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)   # (1, input_size)

                y_hat = self.net(inp).item()
                predictions.append(y_hat)
                past_y.append(y_hat)

        return np.array(predictions, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MX, MY        = 2, 2
    N_FEATURES    = 15        # 15 exogenous features
    INPUT_SIZE    = N_FEATURES * MX + MY   # 32

    model = NARXNet(input_size=INPUT_SIZE, hidden_size=10)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    dummy = torch.randn(8, INPUT_SIZE)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")  # (8, 1)
