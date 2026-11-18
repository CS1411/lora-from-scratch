# LoRA from Scratch

Working through the askThili drill **Build LoRA from Scratch**: freeze a pretrained weight, learn a low-rank update in its place, inject it into GPT-2, train only the adapter, then merge, swap and quantize.

Papers: [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) and [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314).

## Progress

- [x] Lesson 1 - Why not just fine-tune everything?
- [x] Lesson 2 - The math of low-rank
- [x] Lesson 3 - Build LoRALinear from scratch
- [x] Lesson 4 - Inject it into GPT-2
- [x] Lesson 5 - Train just the adapter
- [ ] Lesson 6 - Does it match full fine-tuning?
- [ ] Lesson 7 - Merge and swap adapters
- [ ] Lesson 8 - QLoRA
- [ ] Completion test
- [ ] Project

## Layout

- `lessons/lessonN/` code for each lesson
- `notes/` short notes per lesson
