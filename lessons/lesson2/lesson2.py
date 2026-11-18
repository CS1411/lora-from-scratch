"""Lesson 2: the math of low-rank. A weight update dW = B @ A with small rank r captures most of a realistic update."""
import numpy as np


def make_update(d=768, true_rank=6, noise=0.02, seed=0):
    "A realistic fine-tuning update: a genuinely low-rank signal plus a little full-rank noise."
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((d, true_rank)) @ rng.standard_normal((true_rank, d))
    return signal + noise * rng.standard_normal((d, d))


def energy_by_rank(dW):
    "Fraction of the update captured by its top-r singular directions, for every r."
    sv = np.linalg.svd(dW, compute_uv=False)
    return np.cumsum(sv**2) / np.sum(sv**2)


if __name__ == "__main__":
    d = 768
    energy = energy_by_rank(make_update(d))
    print(f"a {d}x{d} update = {d*d:,} numbers if stored in full\n")
    print(f"{'rank r':>7s}{'% of update captured':>24s}{'LoRA params (2*d*r)':>22s}")
    for r in [1, 2, 4, 8, 16, 64, d]:
        params = 2 * d * r if r < d else d * d
        tag = " <- full dW" if r == d else ""
        print(f"{r:7d}{energy[r-1]*100:22.1f}%{params:20,d}{tag}")
    r = 8
    print(f"\nAt rank {r}: ~{energy[r-1]*100:.0f}% of the update, for {2*d*r:,} params, "
          f"{d*d/(2*d*r):.0f}x smaller than the full {d*d:,}.")
    assert energy[7] > 0.99 and energy[0] < 0.25
