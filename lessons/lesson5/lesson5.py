"""Lesson 5: train only the adapter (A and B) on LEDGAR clause tagging. About 15 minutes on CPU."""
import os

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

from task import TOP, add_lora, balanced_subset, ds, make_tokenizer, names

if __name__ == "__main__":
    train = balanced_subset(ds["train"], per_class=100, seed=0)
    tok = make_tokenizer()

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    # mean_resizing=False: transformers >= 5 would otherwise initialise all new rows to the same mean vector,
    # making the six label tokens indistinguishable. The recipe needs distinct random rows (as in older versions).
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    add_lora(model, r=8, alpha=16, dropout=0.05)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable:,} / {total:,} = {100*trainable/total:.2f}%\n", flush=True)

    def encode(e):
        "clause <Label>, with the loss masked to the single label-token position."
        label_str = "<" + names[e["label"]] + ">"
        enc = tok(e["text"].strip() + " " + label_str, truncation=True, padding="max_length", max_length=96)
        label_tid = tok.convert_tokens_to_ids(label_str)
        labels = [-100] * len(enc["input_ids"])
        if label_tid in enc["input_ids"]:
            labels[enc["input_ids"].index(label_tid)] = label_tid
        enc["labels"] = labels
        return enc

    tokenized = train.map(encode, remove_columns=train.column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(output_dir="lora-out", num_train_epochs=6, per_device_train_batch_size=8,
                             learning_rate=1e-3, logging_steps=50, report_to="none", save_strategy="no",
                             disable_tqdm=True)
    Trainer(model=model, args=args, train_dataset=tokenized, data_collator=default_data_collator).train()

    # Save the whole artifact: the adapter AND the label rows it learned to aim at.
    state = {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
    state["label_embeddings"] = model.transformer.wte.weight.data[-len(TOP):].clone()
    torch.save(state, "lora-adapter.pt")
    print(f"\nsaved lora-adapter.pt: {os.path.getsize('lora-adapter.pt'):,} bytes "
          f"({len(state)-1} adapter tensors + the {len(TOP)} label rows)")
