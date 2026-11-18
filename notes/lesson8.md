# Lesson 8 - QLoRA

- After LoRA the frozen base still has to sit in memory: 140 GB in fp16 for a 70B model. QLoRA stores the read-only base at about 4 bits per weight and keeps the small adapter in full precision.
- The scheme built here is block-wise absmax: blocks of 64 weights, one fp32 scale per block, each weight rounded to one of 16 levels. Small blocks mean one outlier only spoils its own block. Real QLoRA adds NF4 levels, double quantization of the scales and paged optimizers.
- Result on the Lesson 5 tagger: 95.8% with the fp32 base, 91.7% with the 4-bit base (mean weight error 13.1% across 25 layers, 81,069,312 weights).
- The four-point drop is explained by this being naive absmax rather than NF4, and by quantizing after training. Real QLoRA quantizes first, so the adapter learns while seeing the quantization error and compensates for it.
- Memory to fine-tune a 70B model: full fine-tune in fp32 about 1,120 GB; LoRA with an fp16 base about 142 GB; QLoRA about 37 GB, which fits one 80 GB GPU.
- Course note: the evaluation looks for your own choices, such as a different rank, adapting the MLP layers too, or NF4. The project for this drill (Sports Commentary Generator) is the place for that: a separate repository that trains and merges its own adapter.
- Code: `lessons/lesson8/lesson8.py` (run from `lessons/lesson5`).
