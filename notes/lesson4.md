# Lesson 4 - Inject it into GPT-2

- Adapters go on the attention projections only. GPT-2 fuses Q, K and V into one layer, `c_attn` (768 -> 2304), so wrapping it once adapts all three.
- Freeze first, then wrap: the new A and B are then the only tensors that still want a gradient.
- `c_attn` is Hugging Face's `Conv1D`, not `nn.Linear`: same y = xW + b but the weight is stored transposed as [d_in, d_out]. The wrapper only needs the two widths.
- Result for distilgpt2 at rank 8: 147,456 trainable parameters against 81,912,576 frozen (0.18%, 557x fewer), and the output is unchanged at injection.
- Caveat: the frozen base still runs on every forward pass, so LoRA saves memory and storage, not forward-pass speed.
- Code: `lessons/lesson4/lesson4.py`.
