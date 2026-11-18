"""LoRALinear: a frozen layer plus a trainable low-rank detour, written from scratch (no peft)."""
import torch
import torch.nn as nn


def _dims(base):
    "nn.Linear has in/out_features; GPT-2's Conv1D stores its weight transposed as [d_in, d_out]."
    if hasattr(base, "in_features"):
        return base.in_features, base.out_features, False
    d_in, d_out = base.weight.shape
    return d_in, d_out, True


class LoRALinear(nn.Module):
    """Frozen base layer + trainable detour B@A, scaled by alpha/r.

    A is small random, B is zero, so the detour starts as an exact no-op and the
    wrapped layer behaves identically to the base until training moves B.
    """

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # freeze W and bias
        d_in, d_out, self.transposed = _dims(base)
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        detour = (self.dropout(x) @ self.A.T) @ self.B.T  # [*,d_in] -> [*,r] -> [*,d_out]
        return self.base(x) + detour * self.scale

    def merged(self):
        "Fold the detour into the frozen weight and return the plain layer (Lesson 7)."
        with torch.no_grad():
            delta = (self.B @ self.A) * self.scale  # [d_out, d_in]
            self.base.weight += delta.T if self.transposed else delta
        return self.base
