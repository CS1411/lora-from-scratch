# Lesson 3 - Build LoRALinear from scratch

- `LoRALinear` wraps a frozen layer and adds a trainable detour: output = base(x) + (x @ A.T @ B.T) * (alpha / r).
- Three decisions: freeze the base; initialise A small random and B zero so the detour starts as an exact no-op; scale by alpha / r.
- Checks that prove it: at init LoRA(x) equals base(x); after nudging B the output changes.
- For a 768 -> 2304 layer at rank 8: 24,576 trainable parameters against 1,771,776 frozen (1.4%).
- Code: `lessons/lesson3/lora.py` (the module) and `lesson3.py` (the checks).
