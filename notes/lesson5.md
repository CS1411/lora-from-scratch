# Lesson 5 - Train just the adapter

- Same LEDGAR clause-tagging task as the Legal-Clause GPT drill: 600 clauses (100 for each of the six commonest types), each rewritten as `clause <Label>`, with the loss masked to the single label-token position.
- Only difference from full fine-tuning: which parameters have `requires_grad=True`. The freeze from Lesson 4 is the whole configuration; `Trainer` just optimises A and B.
- Learning rate 1e-3 (about 10x full fine-tuning) because B starts at zero and the update is confined to a rank-8 subspace. 6 epochs instead of 3.
- Result: loss fell from 2.59 to 0.057 (mean 0.72), the same figures as the course. The saved file is 613 KB.
- The artifact is the adapter plus the six label-token embedding rows. Those rows were added after pretraining, so they are random vectors that the frozen base never trained; the adapter learned to aim at them. Ship A and B without them and a fresh model scores below random guessing.
- Environment note: with transformers 5 the new rows are initialised to the same mean vector by default, so the six labels are indistinguishable and the loss stays at about 10.9. Passing `mean_resizing=False` restores distinct random rows and reproduces the course numbers.
- Code: `lessons/lesson5/lesson5.py`, with the shared task helpers in `task.py`.
