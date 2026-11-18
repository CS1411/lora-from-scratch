"""Lesson 6: load the adapter on a fresh distilgpt2 and score it against full fine-tuning (95.8%).
Run from lessons/lesson5 after lesson5.py, or copy lora-adapter.pt next to this file."""
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson5"))
from task import TOP, add_lora, heldout, label_ids, make_tokenizer, names  # noqa: E402


def load_tagger(path="lora-adapter.pt"):
    "A fresh pretrained distilgpt2 + the adapter file. The 6 label rows travel with the adapter."
    tok = make_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    add_lora(model)
    state = torch.load(path)
    rows = state.pop("label_embeddings")
    with torch.no_grad():
        model.transformer.wte.weight[-len(TOP):] = rows  # tied to lm_head, so both get them
    missing, unexpected = model.load_state_dict(state, strict=False)
    return tok, model.eval(), len(state), tuple(rows.shape), len(missing), len(unexpected)


@torch.no_grad()
def classify(model, tok, clause):
    "One forward pass; argmax restricted to the six label tokens."
    ids = label_ids(tok)
    logits = model(**tok(clause.strip() + " ", return_tensors="pt")).logits[0, -1]
    return ids[max(ids, key=lambda t: logits[t].item())]


if __name__ == "__main__":
    path = "lora-adapter.pt"
    tok, model, n_tensors, rows_shape, n_missing, n_unexpected = load_tagger(path)
    print(f"loaded {n_tensors} adapter tensors + {rows_shape} label rows from a {os.path.getsize(path):,}-byte file")
    print(f"unexpected keys: {n_unexpected} (the {n_missing:,} 'missing' keys are the frozen base)\n")

    val = heldout()
    preds = [classify(model, tok, e["text"]) for e in val]
    trues = [names[e["label"]] for e in val]
    acc = sum(p == t for p, t in zip(preds, trues)) / len(val)

    print(f"{'':18s}{'accuracy':>10s}{'trainable':>14s}{'shipped per task':>19s}")
    print(f"{'full fine-tune':18s}{'95.8%':>10s}{'81,912,576':>14s}{'327 MB':>19s}")
    print(f"{'LoRA r=8':18s}{acc:>9.1%}{'147,456':>14s}{os.path.getsize(path)/1e3:>16,.0f} KB")
    print(f"\n(6 clause types, so random guessing = {1/len(TOP):.1%})")
    print("\nwhere it still slips:")
    for p, t, e in [x for x in zip(preds, trues, val) if x[0] != x[1]][:5]:
        print(f"  true={t:20s} pred={p:20s} :: {e['text'][:52]}...")
