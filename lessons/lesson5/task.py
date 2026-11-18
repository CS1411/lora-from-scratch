"""The LEDGAR clause-tagging task shared by Lessons 5-8 (same recipe as the Legal-Clause GPT drill)."""
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson4"))
from lesson4 import add_lora  # noqa: E402,F401  (re-exported for later lessons)

ds = load_dataset("coastalcph/lex_glue", "ledgar")
names = ds["train"].features["label"].names
TOP = [lab for lab, _ in Counter(ds["train"]["label"]).most_common(6)]  # the six commonest clause types
topset = set(TOP)


def balanced_subset(split, per_class, seed):
    "per_class clauses of each of the six commonest types, shuffled."
    picked = defaultdict(list)
    for i, lab in enumerate(split["label"]):
        if lab in topset and len(picked[lab]) < per_class:
            picked[lab].append(i)
        if all(len(picked[t]) >= per_class for t in TOP):
            break
    flat = [i for t in TOP for i in picked[t]]
    random.Random(seed).shuffle(flat)
    return split.select(flat)


def make_tokenizer(with_labels=True):
    "distilgpt2 tokenizer; optionally one atomic <clause type> token per label."
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.pad_token = tok.eos_token
    if with_labels:
        tok.add_tokens([f"<{names[i]}>" for i in TOP])
    return tok


def label_ids(tok):
    return {tok.convert_tokens_to_ids(f"<{names[i]}>"): names[i] for i in TOP}


def heldout():
    "The exact 120 validation clauses used to score full fine-tuning (95.8%)."
    return ds["validation"].filter(lambda e: e["label"] in topset).shuffle(seed=2).select(range(120))
