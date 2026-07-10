# Day 10 Perplexity Baseline

## Setup

- Model: `models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- Corpus: `data/wikitext-2-valid.txt`
- Source: `https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt`
- Command:

```bash
llama-perplexity \
  -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -f ./data/wikitext-2-valid.txt \
  -ngl 99 \
  -c 512 \
  --chunks 16 \
  --no-warmup
```

## Result

- `Final estimate: PPL = 6.8790 +/- 0.30335`

## Interpretation

PPL is `exp(cross_entropy)`. A PPL of about `6.88` means the model's average next-token uncertainty on this eval slice is roughly like choosing among `6.88` equally plausible tokens. Lower is better, but numbers are only comparable under the same corpus, tokenizer, context length, and eval settings.
