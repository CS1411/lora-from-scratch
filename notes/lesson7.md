# Lesson 7 - Merge and swap adapters

- One frozen base can serve many tasks. A second adapter was trained on the same LEDGAR text with plain next-token prediction and no label tokens, so it learns a "voice" instead of a classifier. Its file has just the twelve A and B tensors (594 KB).
- Swapping: one base model stays in memory and `load_adapter` overwrites the twelve tensors. With the voice adapter the model continues "The parties hereby agree that" in contract language; with the clause adapter it tags a New York governing-law clause as `Governing Laws`.
- Merging: `W' = W + (B @ A) * (alpha / r)`. GPT-2's `Conv1D` stores its weight as `[d_in, d_out]`, so the update goes in transposed. The wrapper disappears and the model is a plain GPT-2 again.
- Merged and un-merged logits agree to float32 round-off (max difference 1.8e-04 on logits spanning about -183 to 9) and the prediction is unchanged.
- Cost: the detour adds 24,576 multiply-adds per token per layer, 1.39% on top of 1,769,472. Merged removes it.
- Merge when a model does one job forever (plain checkpoint, no custom classes, but back to a full-size copy per task). Keep it un-merged when one base serves many tasks.
- Code: `lessons/lesson7/lesson7.py` (run from `lessons/lesson5`; it writes `lora-voice.pt` there).
