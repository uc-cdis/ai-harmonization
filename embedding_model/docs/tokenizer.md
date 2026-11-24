# Tokenizer

A tokenizer cuts a text into tokens: words, pieces of words and punctuation
marks. `tokenizer train` learns a tokenizer's vocabulary from a
[corpus](corpus.md)'s training pairs, by
[byte-pair encoding (BPE)](https://huggingface.co/learn/llm-course/chapter6/5),
with Hugging Face's [`tokenizers`](https://github.com/huggingface/tokenizers)
library.

## Running `tokenizer train`

```
python -m embedding_model tokenizer train --corpus all-sources                # 16,000 tokens: all-sources-16k
python -m embedding_model tokenizer train --corpus example --vocab-size 8000  # a small corpus: example-8k
```

- **`--vocab-size`** sets the vocabulary size: 16,000 tokens by default. A
  small corpus can give fewer tokens than asked for. `tokenizer train` then
  prints a note, and the manifest holds the vocabulary size reached. To
  choose between sizes, train a model with each, and compare their
  validation MRR@10 in each run's [`manifest.json`](training.md#manifestjson).
- **`--min-frequency`** sets how often two tokens must occur together before
  BPE merges them: 2 by default.
- **`--name`** sets the tokenizer's name. The default is the corpus's name and
  the vocabulary size, in thousands when it can be: `all-sources-16k`,
  `example-8k`, `example-12500`.

A run's config names the tokenizer in `data.tokenizer`, for example
`tokenizer: example-8k`. Use the name from `--name`, or the default name. After
each new `corpus build`, train the corpus's tokenizer again: `train` refuses a
tokenizer that learned from another build of its corpus.
[The workflow](../README.md#the-workflow) gives the order of the steps.


## What `tokenizer train` writes

`tokenizer train` writes to `embedding_model/inputs/tokenizers/<tokenizer>/`.

```
tokenizers/<tokenizer>/
    tokenizer.json    the tokenizer's rules and its vocabulary
    manifest.json     the tokenizer's manifest
```

### `tokenizer.json`

[`tokenizers`](https://huggingface.co/docs/tokenizers/v0.23.2/en/quicktour)
reads the file with `Tokenizer.from_file("tokenizer.json")`. The file holds
these fields, among others:

| field | what it holds |
|---|---|
| `normalizer` | four steps. The text cleaning of [BERT's tokenizer](https://huggingface.co/docs/tokenizers/v0.23.2/en/api/normalizers#tokenizers.normalizers.BertNormalizer) removes control, format and private-use characters. NFD splits most accented letters into the letter and its accents. `Lowercase` lowercases the text. `StripAccents` drops the accents. |
| `pre_tokenizer` | the split, `BertPreTokenizer`: at whitespace, at punctuation and at ASCII symbols |
| `model` | BPE's tokens and merges, learned from the training texts |
| `added_tokens` | the five special tokens, with the first five ids: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]` and `[MASK]` |

### `manifest.json`

The tokenizer's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: `tokenizer.json`, with its SHA-256
- `sources`: the corpus, with its kept pairs' checksums
- `settings`:
  - `vocab_size`: the vocabulary size asked for
  - `min_frequency`: the minimum frequency
  - `tokenizers`: the version of `tokenizers`
- `counts`:
  - `training_texts`: the number of distinct training texts
  - `vocabulary`: the vocabulary size reached
  - `diagnostics`: the [diagnostics](#diagnostics), as `fertility`,
    `words_split`, `unknown_rate` and `vocabulary_used`

## Details

### Diagnostics

After training, `tokenizer train` prints how the tokenizer cuts its training
texts. A word here is a piece that the `pre_tokenizer` splits off. So
`SUBJECT_ID` holds three words: `subject`, `_` and `id`.

| measure | what it says | a bad value |
|---|---|---|
| fertility | tokens per word | high: many words are cut into pieces, so the vocabulary does not fit the texts |
| words split | the share of words cut into two or more tokens | a large share, for the same reason |
| unknown rate | the share of tokens that are `[UNK]` | above 0: the training texts have characters that the vocabulary lacks, which is a sign of a broken tokenizer |
| vocabulary used | the share of the vocabulary's tokens that the training texts use, not counting the special tokens | well below 1: many tokens never occur, so their vectors keep the random values that they start with |
