# Lesson 1 - Why not just fine-tune everything?

- Training in fp32 costs about 16 bytes per parameter: weight + gradient + two Adam states (4 bytes each).
- That is 1.3 GB for distilgpt2 (82M), but 112 GB for a 7B model and 1,120 GB for 70B. A single high-end GPU has about 80 GB.
- Full fine-tuning also means a whole new model copy for every task.
- LoRA's bet: freeze the pretrained weights and learn only a small, low-rank adjustment per task. Only the adapter carries gradient and optimizer state, so a 7B model needs about 14 GB.
- Code: `lessons/lesson1/lesson1.py`.
