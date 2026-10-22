"""prep_slimpajama.py - stream SlimPajama, tokenize on the fly, and write a uint16 memmap (+.meta.json).
Files are fetched directly by URL in a fixed order, so the token stream is reproducible.

Usage:
  TOKENS=2000000000 OUTBIN=data/slim_2b.bin python prep_slimpajama.py
env:
  SOURCE  reupload (default; Reverent/SlimPajama-627B_Reupload, 250 parquet shards, resumable)
          | dkyoon6b (DKYoon/SlimPajama-6B subsample) | cerebras (official repo, train/chunk{c}/example_train_{n}.jsonl.zst)
  TOK     tokenizer (default mistralai/Mistral-7B-v0.1; if gated, TOK=TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  TOKENS  target token count (default 2e9)
  OUTBIN  output bin path (+.meta.json)
  HF_ENDPOINT Hub address (default https://huggingface.co)
Dependencies: pip install requests zstandard transformers sentencepiece"""
import io
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.pop("HF_HUB_OFFLINE", None)       # clear an offline switch possibly set by the outer environment
os.environ.pop("HF_DATASETS_OFFLINE", None)

import numpy as np  # noqa: E402
import requests  # noqa: E402
import zstandard as zstd  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

BASE = os.environ["HF_ENDPOINT"].rstrip("/")
REPO = "datasets/cerebras/SlimPajama-627B"
SOURCE = os.environ.get("SOURCE", "reupload")
TOK = os.environ.get("TOK", "mistralai/Mistral-7B-v0.1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TOKENS = int(float(os.environ.get("TOKENS", "2e9")))
OUTBIN = os.environ.get("OUTBIN", "data/slim_2b.bin")

os.makedirs(os.path.dirname(OUTBIN) or ".", exist_ok=True)
tok = AutoTokenizer.from_pretrained(TOK, token=HF_TOKEN or None)
assert tok.vocab_size < 65536, "uint16 requires vocab < 65536"
eos = tok.eos_token_id
print(f"source={SOURCE} tokenizer={TOK} vocab={tok.vocab_size} eos={eos} "
      f"target={TOKENS:,} mirror={BASE} auth={'yes' if HF_TOKEN else 'no'}", flush=True)

_rp = (os.path.exists(OUTBIN) and os.path.getsize(OUTBIN) == TOKENS * 2
       and os.path.exists(OUTBIN + ".progress.json"))   # resume: do not clear data already written
mm = np.memmap(OUTBIN, dtype=np.uint16, mode=("r+" if _rp else "w+"), shape=(TOKENS,))
pos = 0
sess = requests.Session()
if HF_TOKEN:
    sess.headers["Authorization"] = f"Bearer {HF_TOKEN}"


def flush(id_buf):
    global pos
    if not id_buf:
        return
    n = min(len(id_buf), TOKENS - pos)
    mm[pos:pos + n] = np.asarray(id_buf[:n], dtype=np.uint16)
    pos += n


def encode_texts(texts):
    out = []
    for ids in tok(texts, add_special_tokens=False)["input_ids"]:
        out.extend(ids)
        out.append(eos)
    return out


import time as _time


def fetch_shard(url, tmp):
    """Resumable download (Range + exponential backoff); returns False = 404, skip."""
    got, total = 0, None
    for attempt in range(15):
        try:
            hdr = {"Range": f"bytes={got}-"} if got else {}
            r = sess.get(url, stream=True, timeout=300, headers=hdr)
            if r.status_code == 404:
                return False
            if r.status_code == 200 and got:      # server does not support Range: safely re-download
                got = 0
            if r.status_code not in (200, 206):
                raise requests.exceptions.RequestException(f"HTTP {r.status_code}")
            if r.status_code == 206 and "Content-Range" in r.headers:
                total = int(r.headers["Content-Range"].split("/")[-1])
            elif total is None and "Content-Length" in r.headers:
                total = got + int(r.headers["Content-Length"])
            with open(tmp, "ab" if got else "wb") as f:
                for cb in r.iter_content(1 << 22):
                    f.write(cb)
                    got += len(cb)
            if total is None or got >= total:
                return True
            print(f"    ended early @ {got/1e9:.2f}/{total/1e9:.2f}GB, resuming download", flush=True)
        except (requests.exceptions.RequestException, OSError) as e:
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            print(f"    connection lost @ {got/1e9:.2f}GB ({type(e).__name__}), "
                  f"retry {attempt + 1}/15", flush=True)
            _time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"download failed 15 times: {url}")


done = False
if SOURCE == "reupload":
    import pyarrow.parquet as pq
    RREPO = "datasets/Reverent/SlimPajama-627B_Reupload"
    tmp = OUTBIN + ".shard.tmp"
    prog_f = OUTBIN + ".progress.json"
    start_shard = 0
    if os.path.exists(prog_f) and os.path.exists(OUTBIN):
        _pg = json.load(open(prog_f))
        if _pg.get("target") == TOKENS and _pg.get("tok") == TOK:
            start_shard, pos = _pg["next_shard"], _pg["pos"]
            mm = np.memmap(OUTBIN, dtype=np.uint16, mode="r+", shape=(TOKENS,))
            print(f"  resume: from shard {start_shard}, pos={pos:,}", flush=True)
    for i in range(start_shard, 250):
        if done:
            break
        if os.path.exists(tmp):
            os.remove(tmp)
        url = f"{BASE}/{RREPO}/resolve/main/data/train-{i:05d}-of-00250.parquet"
        if not fetch_shard(url, tmp):
            print(f"  shard {i}: 404, skip", flush=True)
            continue
        pf = pq.ParquetFile(tmp)
        col = "text" if "text" in pf.schema_arrow.names else pf.schema_arrow.names[0]
        for batch in pf.iter_batches(batch_size=512, columns=[col]):
            texts = [t for t in batch.column(0).to_pylist() if t]
            flush(encode_texts(texts))
            if pos >= TOKENS:
                done = True
                break
        os.remove(tmp)
        with open(prog_f, "w") as pf:
            json.dump({"next_shard": i + 1, "pos": pos, "target": TOKENS, "tok": TOK}, pf)
        print(f"  shard {i}: {pos/1e6:.0f}M / {TOKENS/1e6:.0f}M tokens", flush=True)
    done = True

if SOURCE == "dkyoon6b":                     # ~6B-token subsample
    from datasets import load_dataset
    ds = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
    texts = []
    for ex in ds:
        texts.append(ex["text"])
        if len(texts) >= 512:
            flush(encode_texts(texts))
            texts = []
            if pos >= TOKENS:
                done = True
                break
            if pos % 50_000_000 < 600_000:
                print(f"  {pos/1e6:.0f}M / {TOKENS/1e6:.0f}M tokens", flush=True)
    if texts and not done:
        flush(encode_texts(texts))
    done = True

for chunk in range(1, 11):
    if done:
        break
    misses = 0
    n = 0
    while misses < 2 and not done:
        url = f"{BASE}/{REPO}/resolve/main/train/chunk{chunk}/example_train_{n}.jsonl.zst"
        r = sess.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            misses += 1
            n += 1
            continue
        misses = 0
        reader = zstd.ZstdDecompressor().stream_reader(r.raw)
        texts = []
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
            if len(texts) >= 512:
                flush(encode_texts(texts))
                texts = []
                if pos >= TOKENS:
                    done = True
                    break
        if texts and not done:
            flush(encode_texts(texts))
            if pos >= TOKENS:
                done = True
        if n % 20 == 0 or done:
            print(f"  chunk{chunk}/file{n}: {pos/1e6:.0f}M / {TOKENS/1e6:.0f}M tokens",
                  flush=True)
        n += 1

mm.flush()
meta = dict(tokenizer=TOK, vocab_size=tok.vocab_size, tokens=pos, dtype="uint16",
            source={"reupload": "Reverent/SlimPajama-627B_Reupload data/train-*.parquet sequential", "dkyoon6b": "DKYoon/SlimPajama-6B", "cerebras": "cerebras/SlimPajama-627B train chunk-sequential"}[SOURCE])
with open(OUTBIN + ".meta.json", "w") as f:
    json.dump(meta, f)
print(f"DONE: {pos:,} tokens -> {OUTBIN} (+.meta.json)", flush=True)
assert pos >= TOKENS * 0.99, f"only got {pos:,}/{TOKENS:,} tokens - check network/mirror and rerun"
