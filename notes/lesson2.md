# Lesson 2 - The math of low-rank

- Fine-tuning changes a weight W by some update dW (768 x 768 = 589,824 numbers for GPT-2). LoRA writes it as dW = B @ A with A of shape r x d and B of shape d x r, costing 2*d*r parameters instead of d^2.
- The bet is that useful updates live in a low-rank subspace. On a test update (rank-6 signal plus a little noise), rank 8 captures about 100% of it with 12,288 parameters, 48x fewer than the full matrix.
- Rank r is the dial between capacity and cost.
- The detour output is scaled by alpha / r so changing the rank does not change how strongly the adapter speaks.
- Code: `lessons/lesson2/lesson2.py`.
