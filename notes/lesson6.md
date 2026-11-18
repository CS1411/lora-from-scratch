# Lesson 6 - Does it match full fine-tuning?

- Load the 613 KB adapter onto a fresh, untouched distilgpt2 and score it on the same 120 held-out clauses (`shuffle(seed=2)`) that gave full fine-tuning 95.8%, using constrained prediction: one forward pass, argmax over only the six label-token ids.
- Result: 95.8% for LoRA r=8, equal to full fine-tuning, with 147,456 trainable parameters instead of 81.9M and a 613 KB file instead of 327 MB.
- The label rows must be written back into the embedding table before scoring. `lm_head` is tied to `wte`, so one write fixes both reading and emitting a label token.
- It slips on the same boilerplate look-alikes as full fine-tuning (Severability, Entire Agreements, Survival, Notices): it learned the same skill and inherited the same hard cases.
- Honest ledger: LoRA needed 6 epochs against 3, so on this small CPU model it was not faster end to end. The wins are memory (no gradient or optimizer state for 99.8% of the model), shipping (kilobytes per task) and reversibility (the base is bit-identical).
- Code: `lessons/lesson6/lesson6.py` (run from `lessons/lesson5` after Lesson 5).
