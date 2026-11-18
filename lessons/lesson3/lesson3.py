"""Lesson 3: build LoRALinear and check its two invariants."""
import torch
import torch.nn as nn

from lora import LoRALinear

if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(768, 2304)  # the shape of GPT-2's fused Q/K/V projection
    lora = LoRALinear(base, r=8, alpha=16)

    x = torch.randn(4, 768)
    same = torch.allclose(lora(x), base(x))
    print("at init, LoRA(x) == base(x):", same, " (because B=0, so B@A=0)")

    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in lora.parameters() if not p.requires_grad)
    print(f"trainable (A,B): {trainable:,}  frozen (W,bias): {frozen:,}"
          f"  -> {100*trainable/(trainable+frozen):.1f}% trainable")

    with torch.no_grad():
        lora.B += 0.1 * torch.randn_like(lora.B)
    live = not torch.allclose(lora(x), base(x))
    print("after nudging B, LoRA(x) != base(x):", live, " (detour is now live)")

    assert same and live and trainable == 24_576 and frozen == 1_771_776
