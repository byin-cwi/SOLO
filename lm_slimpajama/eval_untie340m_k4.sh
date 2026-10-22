#!/bin/bash
# eval_untie340m_k4.sh - export a 340M K=4 pipeline checkpoint (export_hf.py) and evaluate it with lm-eval-harness.
#   Terminal exit: wikitext + six zero-shot tasks; intermediate exits 0/1/2: wikitext only.
# Usage: bash eval_untie340m_k4.sh [TAG] [GPU] [EXITS]     # default untie340m_k4_untie_s42  0  "-1 0 1 2"
#   bash eval_untie340m_k4.sh untie340m_k4_tie_s42 1
set -e
cd "$(dirname "$0")"
TAG=${1:-untie340m_k4_untie_s42}; GPU=${2:-0}; EXITS=${3:--1 0 1 2}
for r in 0 1 2 3; do [ -f ckpt/$TAG.pt.r$r ] || { echo "missing ckpt/$TAG.pt.r$r"; exit 1; }; done
mkdir -p hf_out evals
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} MKL_THREADING_LAYER=GNU
ZS="lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge"
for e in $EXITS; do
  x=exit$e; [ "$e" -lt 0 ] && x=term
  echo "== export $TAG/$x"
  SRC=pipe CKPT=ckpt/$TAG.pt K=4 EXIT=$e HEAD_DEPTH=2 NLAYER=24 DMODEL=1024 NHEAD=16 FFN=0 SEQ=2048 V=32000 \
    OUTDIR=hf_out/${TAG}_$x python -u export_hf.py
  echo "== wikitext $TAG/$x"
  CUDA_VISIBLE_DEVICES=$GPU lm_eval --model hf --model_args pretrained=hf_out/${TAG}_$x,dtype=bfloat16 \
    --tasks wikitext --batch_size 8 --output_path evals/${TAG}_$x
done
if echo " $EXITS " | grep -q " -1 "; then
  echo "== zero-shot six tasks $TAG/term"
  CUDA_VISIBLE_DEVICES=$GPU lm_eval --model hf --model_args pretrained=hf_out/${TAG}_term,dtype=bfloat16 \
    --tasks $ZS --batch_size 16 --output_path evals/${TAG}_zs
fi
echo "== $TAG EVAL DONE"
grep -h "word_perplexity" evals/${TAG}_*/*.json 2>/dev/null | head; true
