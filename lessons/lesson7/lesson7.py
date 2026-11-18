"""Lesson 7: train a second ("voice") adapter, hot-swap adapters on one base, then merge B@A into W.
Run from lessons/lesson5 (needs lora-adapter.pt from Lesson 5); it writes lora-voice.pt there."""
import os
import sys
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          default_data_collator)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson5"))
from task import TOP, add_lora, balanced_subset, ds, label_ids, make_tokenizer  # noqa: E402

PROMPT = "The parties hereby agree that"
CLAUSE = ("This Agreement shall be governed by and construed in accordance with the laws of the "
          "State of New York.")


def train_voice(tok, path="lora-voice.pt"):
    "Plain next-token prediction on contract text, no label tokens: the adapter learns a voice."
    voice_train = balanced_subset(ds["train"], per_class=100, seed=7)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    add_lora(model, r=8, alpha=16, dropout=0.05)

    def encode_lm(e):
        enc = tok(e["text"].strip(), truncation=True, padding="max_length", max_length=96)
        enc["labels"] = [t if m == 1 else -100 for t, m in zip(enc["input_ids"], enc["attention_mask"])]
        return enc

    data = voice_train.map(encode_lm, remove_columns=voice_train.column_names)
    data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    args = TrainingArguments(output_dir="voice-out", num_train_epochs=3, per_device_train_batch_size=8,
                             learning_rate=1e-3, logging_steps=75, report_to="none", save_strategy="no",
                             disable_tqdm=True)
    Trainer(model=model, args=args, train_dataset=data, data_collator=default_data_collator).train()
    voice = {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
    torch.save(voice, path)
    print(f"\nsaved {path}: {os.path.getsize(path):,} bytes ({len(voice)} tensors)")


def load_adapter(model, path, n_rows=6):
    "Swap in an adapter: 12 small tensors (+ label rows, if that adapter added tokens)."
    state = torch.load(path)
    rows = state.pop("label_embeddings", None)
    if rows is not None:
        with torch.no_grad():
            model.transformer.wte.weight[-n_rows:] = rows
    model.load_state_dict(state, strict=False)
    return model.eval()


@torch.no_grad()
def generate(m, t, prompt=PROMPT, n=22):
    out = m.generate(**t(prompt, return_tensors="pt"), max_new_tokens=n, do_sample=False,
                     pad_token_id=t.eos_token_id)
    return t.decode(out[0], skip_special_tokens=True).replace("\n", " ")


@torch.no_grad()
def classify(m, tok, clause=CLAUSE):
    ids = label_ids(tok)
    logits = m(**tok(clause.strip() + " ", return_tensors="pt")).logits[0, -1]
    return ids[max(ids, key=lambda t: logits[t].item())]


if __name__ == "__main__":
    tok = make_tokenizer()
    if not os.path.exists("lora-voice.pt"):
        train_voice(tok)

    plain_tok = AutoTokenizer.from_pretrained("distilgpt2")
    plain = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    print("base, no adapter :", generate(plain, plain_tok)[:120])

    base = AutoModelForCausalLM.from_pretrained("distilgpt2")
    base.resize_token_embeddings(len(tok), mean_resizing=False)
    add_lora(base)
    load_adapter(base, "lora-voice.pt")
    print("+ voice adapter  :", generate(base, tok)[:120])
    load_adapter(base, "lora-adapter.pt")  # swap: same object in memory
    print("+ clause adapter :", f"tags the New York clause as {classify(base, tok)!r}")
    print("  (one base in memory throughout, only 12 small tensors changed)\n")

    ids = tok(CLAUSE + " ", return_tensors="pt")
    with torch.no_grad():
        before = base(**ids).logits
    tag_before = classify(base, tok)
    for blk in base.transformer.h:
        blk.attn.c_attn = blk.attn.c_attn.merged()  # wrapper gone
    with torch.no_grad():
        after = base(**ids).logits
    diff = (before - after).abs().max().item()
    print("after merging into W:")
    print(f"  c_attn is a plain {type(base.transformer.h[0].attn.c_attn).__name__} again, no wrapper to ship")
    print(f"  same prediction: {classify(base, tok)!r}")
    print(f"  logits agree to float32 round-off: max |diff| = {diff:.1e}\n")

    d, d_out, r = base.config.n_embd, 3 * base.config.n_embd, 8
    mac_base, mac_detour = d * d_out, d * r + r * d_out
    print("inference cost, per token, per layer:")
    print(f"  base c_attn : {mac_base:>10,} multiply-adds")
    print(f"  LoRA detour : {mac_detour:>10,} (+{100*mac_detour/mac_base:.2f}%)")
    print(f"  merged      : {mac_base:>10,} (+0.00%), the detour is folded into W")
    assert classify(base, tok) == tag_before and diff < 1e-2
