"""Lesson 8: QLoRA idea. Quantize the frozen base to 4 bits, re-score the tagger, price a 70B fine-tune.
Run from lessons/lesson5 (needs lora-adapter.pt from Lesson 5)."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson6"))
from lesson6 import classify, load_tagger  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson5"))
from task import heldout, names  # noqa: E402


def quantize_4bit(w, block=64):
    "Block-wise absmax: 16 levels per block of 64 weights, one fp32 scale per block."
    flat = w.detach().flatten().float()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.view(-1, block)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.round(blocks / scale * 7).clamp(-8, 7).to(torch.int8)  # 4 bits of signal
    return q, scale, pad, w.shape


def dequantize(q, scale, pad, shape):
    out = (q.float() / 7 * scale).flatten()
    if pad:
        out = out[:out.numel() - pad]
    return out.view(shape)


def memory_gb(params):
    "Memory to fine-tune a model with `params` parameters, by method."
    return {
        "full fine-tune, fp32": params * 16 / 1e9,
        "LoRA, fp16 frozen base": (params * 2 + params * 0.0018 * 16) / 1e9,
        "QLoRA, 4-bit frozen base": (params * 0.5 + params * 0.0018 * 16) / 1e9,
    }


if __name__ == "__main__":
    tok, model, *_ = load_tagger()
    val = heldout()
    trues = [names[e["label"]] for e in val]

    def accuracy(m):
        return sum(classify(m, tok, e["text"]) == t for e, t in zip(val, trues)) / len(val)

    print(f"fp32 frozen base + adapter : {accuracy(model):.1%}")
    n_quant, err = 0, []
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, (Conv1D, nn.Linear)):  # every frozen base weight; A and B are plain Parameters
                w = mod.weight.data
                dq = dequantize(*quantize_4bit(w))
                err.append((w - dq).abs().mean().item() / w.abs().mean().item())
                mod.weight.data = dq
                n_quant += w.numel()
    print(f"quantized {n_quant:,} frozen weights in {len(err)} layers "
          f"(mean relative error {100*sum(err)/len(err):.1f}%)")
    print(f"4-bit frozen base + adapter: {accuracy(model):.1%}")

    print("\nmemory to FINE-TUNE, by method (70B model):")
    for name, gb in memory_gb(70e9).items():
        print(f"  {name:26s}: {gb:>7,.0f} GB")
    print(f"  {'a single 80 GB GPU':26s}: {80:>7,.0f} GB")
