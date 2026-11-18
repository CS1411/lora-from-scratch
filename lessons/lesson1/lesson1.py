"""Lesson 1: why not just fine-tune everything? Training memory, full fine-tune vs LoRA."""
GB = 1e9


def full_ft_gb(n):
    "fp32 training: weight + gradient + Adam mean + Adam variance = 16 bytes/param."
    return n * 16 / GB


def lora_train_gb(n, trainable_frac=0.0018):
    "LoRA: the frozen base only needs to be resident (fp16, 2 B/param); only the adapter carries grad + optimizer."
    base = n * 2 / GB
    adapter = n * trainable_frac * 16 / GB
    return base + adapter


if __name__ == "__main__":
    print(f"{'model':14s}{'full FT (fp32)':>18s}{'LoRA':>14s}")
    for name, n in [("distilgpt2", 82e6), ("Llama-2-7B", 7e9), ("Llama-2-70B", 70e9)]:
        print(f"{name:14s}{full_ft_gb(n):15,.1f} GB{lora_train_gb(n):11,.1f} GB")
    print("\nA single high-end GPU tops out around 80 GB.")
    print("Full fine-tuning a 7B model in fp32 (~112 GB) does not fit; LoRA (~14 GB) does.")
    assert round(full_ft_gb(7e9)) == 112 and round(lora_train_gb(7e9), 1) == 14.2
