# LoRA from Scratch

A from-scratch implementation of **LoRA** (Low-Rank Adaptation) and **QLoRA**-style quantization, built while working through the askThili drill *Build LoRA from Scratch*. No `peft`: the adapter is a small PyTorch module written lesson by lesson, injected into GPT-2, trained, merged, swapped and quantized.

Papers: [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) and [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314).

**Project built with it:** [sports-commentary-generator](https://github.com/CS1411/sports-commentary-generator), a Streamlit tool whose commentary comes from a LoRA-adapted GPT-2 with its own trained and merged adapter.

## Progress

- [x] Lesson 1 - Why not just fine-tune everything?
- [x] Lesson 2 - The math of low-rank
- [x] Lesson 3 - Build LoRALinear from scratch
- [x] Lesson 4 - Inject it into GPT-2
- [x] Lesson 5 - Train just the adapter
- [x] Lesson 6 - Does it match full fine-tuning?
- [x] Lesson 7 - Merge and swap adapters
- [x] Lesson 8 - QLoRA
- [x] Completion test
- [x] Project ([sports-commentary-generator](https://github.com/CS1411/sports-commentary-generator))

## Results

All figures are reproduced by the code here (each lesson's `output.txt` holds the run output).

| Lesson | Result |
|---|---|
| 1 | Full fine-tuning needs about 16 bytes per parameter: 112 GB for a 7B model, against about 14 GB with LoRA |
| 2 | A rank-8 update captures about 100% of a realistic 768 x 768 update with 12,288 parameters instead of 589,824 |
| 3 | `LoRALinear` is an exact no-op at initialisation (B = 0) and comes alive once B trains |
| 4 | LoRA on GPT-2's fused Q/K/V in all 6 blocks of distilgpt2: **147,456 trainable parameters, 0.18% of the model** |
| 5 | Clause-tagger training loss 2.59 to 0.057 in 6 epochs; the adapter file is 613 KB |
| 6 | **95.8% accuracy, equal to full fine-tuning**, with 557x fewer trainable weights and a file 534x smaller |
| 7 | One frozen base serves two adapters by swapping twelve tensors; merging `B @ A` into `W` changes the logits by only 1.8e-04 |
| 8 | 4-bit block-wise quantization of the frozen base: 95.8% falls to 91.7%; QLoRA-style memory for a 70B fine-tune is about 37 GB, one 80 GB GPU |

## Run it

Requires Python 3.12 and git. Lessons 1 and 2 need only NumPy; lessons 3 to 8 need PyTorch and the Hugging Face libraries. Lessons 4 to 8 download `distilgpt2` and the LEDGAR legal-clause dataset from the Hugging Face Hub on first use.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd lessons/lesson3 && python lesson3.py      # LoRALinear and its two invariants
cd ../lesson4 && python lesson4.py           # inject into GPT-2, count trainable parameters
cd ../lesson5 && python lesson5.py           # train the adapter, about 10 to 15 minutes on CPU
python ../lesson6/lesson6.py                 # score it against full fine-tuning
python ../lesson7/lesson7.py                 # voice adapter, swap, merge
python ../lesson8/lesson8.py                 # 4-bit quantization
```

Lessons 6 to 8 read `lora-adapter.pt` (and Lesson 7 `lora-voice.pt`) from the current directory, so run them from `lessons/lesson5` after Lesson 5 as above. Weight files are git-ignored.

## How the code fits together

```text
lessons/
  lesson1  training-memory arithmetic
  lesson2  low-rank approximation of a weight update (SVD)
  lesson3  lora.py: LoRALinear (frozen base + B @ A detour, scaled by alpha / r, merge support)
  lesson4  add_lora: freeze the model, wrap c_attn in every block
  lesson5  task.py (LEDGAR clause-tagging task) and the adapter training run
  lesson6  load the adapter on a fresh base and evaluate it
  lesson7  second adapter, hot-swap, merge into W
  lesson8  4-bit block-wise quantization and memory arithmetic
notes/     short notes for every lesson
```

## Environment notes

With transformers 5 a resized embedding table initialises every new row to the same mean vector, which makes the six clause-label tokens identical and stops Lesson 5 from learning (the loss stays near 10.9). The lessons here pass `mean_resizing=False` to restore distinct random rows, which reproduces the course numbers exactly. Library versions are pinned in `requirements.txt`.
