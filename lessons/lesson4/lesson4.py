"""Lesson 4: inject LoRA into distilgpt2's attention and count what trains."""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson3"))
from lora import LoRALinear  # noqa: E402


def add_lora(model, r=8, alpha=16, dropout=0.05):
    """Freeze the whole pretrained model, then wrap the fused Q/K/V projection (c_attn) in every block.

    Freeze first, wrap second: the adapters are created afterwards, so they are the only tensors
    that still want a gradient.
    """
    for p in model.parameters():
        p.requires_grad = False
    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, r=r, alpha=alpha, dropout=dropout)
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    attn = model.transformer.h[0].attn.c_attn
    print("what c_attn actually is:", type(attn).__name__, " weight", tuple(attn.weight.shape), "\n")

    ids = torch.tensor([[15496, 11, 995]])  # "Hello, world"
    with torch.no_grad():
        before = model(ids).logits
    add_lora(model, r=8, alpha=16)
    with torch.no_grad():
        after = model(ids).logits
    identical = torch.allclose(before, after, atol=1e-6)
    print("identical to the base at injection:", identical, " (every B=0, so every detour is a no-op)\n")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable : {trainable:>12,}")
    print(f"frozen    : {total - trainable:>12,}")
    print(f"-> {100 * trainable / total:.2f}% of the model trains ({total / trainable:,.0f}x fewer weights)\n")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n:40s} {tuple(p.shape)}")
    assert identical and trainable == 147_456
