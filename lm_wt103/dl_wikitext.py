"""One-time data bootstrap: download WikiText-2/103 + prefetch the bert-base-cased tokenizer (Hub address set by HF_ENDPOINT).
Usage:  DATA=./data/wikitext python dl_wikitext.py
Afterwards the training scripts run offline with the same DATA path (they force HF offline mode by default)."""
import os

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
DATA = os.environ.get("DATA", "./data/wikitext")

from datasets import load_dataset  # noqa: E402

for cfg in ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]:
    d = load_dataset("Salesforce/wikitext", cfg, cache_dir=DATA)
    print(cfg, "OK:", {k: len(v) for k, v in d.items()}, flush=True)

from transformers import AutoTokenizer  # noqa: E402

tk = AutoTokenizer.from_pretrained("bert-base-cased")
print("tokenizer bert-base-cased OK, vocab =", tk.vocab_size, flush=True)
print("ALL WIKITEXT READY (DATA =", DATA + ")")
