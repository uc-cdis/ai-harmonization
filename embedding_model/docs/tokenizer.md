# Tokenizer

A tokenizer cuts a text into tokens: words, pieces of words and punctuation
marks. Each token in its vocabulary has a number, the token's id. `tokenizer
train` learns the vocabulary from a [corpus](corpus.md)'s training pairs, by
[byte-pair encoding (BPE)](https://huggingface.co/learn/llm-course/chapter6/5).
It uses Hugging Face's [`tokenizers`](https://github.com/huggingface/tokenizers) library.

`tokenizer train` writes the tokenizer and its [manifest](manifests.md) to
the tokenizer's folder.
[What `tokenizer train` writes](#what-tokenizer-train-writes) lists the
files. [Details](#details) describes how a tokenizer cuts a text and how it
learns its vocabulary. The section also describes the tokenizer's settings,
diagnostics and name.

[design_decisions.md](design_decisions.md#tokenizer) gives the reasons for the
tokenizer's settings.

## Running `tokenizer train`

```
python -m embedding_model tokenizer train --corpus all-sources                # 16,000 tokens: all-sources-16k
python -m embedding_model tokenizer train --corpus example --vocab-size 8000  # a small corpus: example-8k
```

The corpus's pairs must have the checksum that the corpus's manifest holds.
If they do not, `tokenizer train` stops, and the error gives the command that
builds the corpus again.

## What `tokenizer train` writes

`tokenizer train` writes to `embedding_model/inputs/tokenizers/<name>/`, or
under the folder that `--inputs` names. It writes the folder whole. So if the
training fails, the previous tokenizer stays as it was.
[Inputs and outputs](../README.md#inputs-and-outputs) describes how a step
writes a folder whole.

```
tokenizers/<name>/
    tokenizer.json    the tokenizer's rules and its vocabulary
    manifest.json     the tokenizer's manifest
```

### `tokenizer.json`

`tokenizer.json` holds the tokenizer's rules and its vocabulary, in one file.
Hugging Face
[`tokenizers`](https://huggingface.co/docs/tokenizers/v0.23.2/en/quicktour)
reads the file with `Tokenizer.from_file("tokenizer.json")`, and runs its
own code for each rule that the file names. The file holds these fields,
among others:

| field | what it holds | where it comes from |
|---|---|---|
| `normalizer` | [step 1](#step-1-normalize), as in BERT's tokenizer: cleaning, NFD, lowercase, no accents | the code, fixed |
| `pre_tokenizer` | [step 2](#step-2-split): `BertPreTokenizer` | the code, fixed |
| `model` | [step 3](#step-3-cut-into-tokens): BPE's tokens and merges | learned from the training texts |
| `added_tokens` | the [special tokens](#special-tokens) | the code, fixed |

### `manifest.json`

`manifest.json` is the tokenizer's manifest. It has the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: `tokenizer.json`, with its
  [SHA-256](https://doi.org/10.6028/NIST.FIPS.180-4)
- `sources`: the corpus, with its kept pairs' checksums
- `settings`: the vocabulary size asked for, the minimum frequency, and the
  version of `tokenizers`. The manifest records the version because the
  bytes of `tokenizer.json` depend on it. `tokenizer.json` itself holds the
  rules.
- `counts`: the number of training texts, the vocabulary size reached, and
  the [diagnostics](#diagnostics)

## Details

### What a tokenizer does

The tokenizer gives the [model](../README.md#how-the-model-works) one id per
token, and the model sees only these ids. Each token of the vocabulary has its
own id. So a vocabulary of 8,000 tokens has the ids 0 to 7,999. For example,
with made-up ids:

| text | tokens | ids |
|---|---|---|
| `Body Weight` | `body`, `weight` | 812, 1407 |
| `body weight` | `body`, `weight` | 812, 1407 |
| `body_weight` | `body`, `_`, `weight` | 812, 63, 1407 |
| `bodyweight` | `body`, `weight` | 812, 1407 |

The first two texts get the same ids, so the model cannot tell them apart.
Here BPE cuts the last one into the same two tokens, so it gets the same ids
too. Byte-pair encoding does not promise that: it can cut another compound
elsewhere, as `heartrate` into `hear` and `trate`.

A tokenizer cuts a text into tokens in three steps:

| step | what it does | example |
|---|---|---|
| 1. normalize | removes control, format and private-use characters, as [BERT's tokenizer](https://huggingface.co/docs/tokenizers/v0.23.2/en/api/normalizers#tokenizers.normalizers.BertNormalizer) does. Then it splits most accented letters into the letter and its accents (Unicode's NFD form), lowercases the text and drops the accents. | `Café` -> `cafe` |
| 2. split | splits at whitespace, at every punctuation character and at every ASCII symbol, for example `+`, as BERT's tokenizer does (`BertPreTokenizer`) | `SUBJECT_ID (float)` -> `subject`, `_`, `id`, `(`, `float`, `)` |
| 3. cut into tokens | cuts each piece into vocabulary tokens, by [byte-pair encoding (BPE)](https://huggingface.co/learn/llm-course/chapter6/5) | `bodyweight` -> `body`, `weight` |

The same rules run at two times:

- **When the tokenizer learns its vocabulary.** Steps 1 and 2 prepare each
  training text, and BPE learns its tokens from the pieces.
- **Every time the model embeds a text.** `tokenizer.json` saves the rules
  with the vocabulary, and the trained model keeps that file. All three steps
  then cut each text that the model reads.

So the tokenizer normalises and splits a text that the model reads later
exactly as it normalised and split the training texts. The section
[What `tokenizer train` writes](#what-tokenizer-train-writes) has more details.

#### Step 1: normalize

The tokenizer first removes control, format and private-use characters.
They hold no text, and nobody can see them:

- a control character, for example `\x92`, a byte left from a broken quote
- a format character, for example `\u200b`, a zero-width space
- a private-use character, for example `\ue000`, which a font can use for
  its own icon

For example, a description copied from a PDF can read `body\u200bweight`.
On the screen it looks like `bodyweight`.

| text | tokens without this step | tokens with it |
|---|---|---|
| `body\u200bweight` | `body`, `[UNK]`, `weight` | `body`, `weight` |

The vocabulary has no token for the invisible character. Without this step,
the character would become the unknown token, `[UNK]`.

A tab, a line feed or a carriage return is a control character too. The
tokenizer turns each of them into a space, so the character still separates
two words.

NFD stands for Normalization Form D, one of the four forms that Unicode
defines in [Unicode Normalization Forms](https://www.unicode.org/reports/tr15/#Norm_Forms).
D is for canonical decomposition.

NFD writes most accented letters as the plain letter, then one or more
accent marks. So `é` becomes `e` followed by the accent. Some letters, for
example `ø` and `ł`, have no decomposition and stay whole. NFD makes the
accents removable. Without it, `é` stays one character, and there is no
separate accent to drop.

Lowercasing makes `BMI`, `Bmi` and `bmi` one text. Variable names come in
any case, and their case carries no meaning. So the three get the same
tokens, and the model gives them the same embedding.

#### Step 2: split

The split cuts a text into pieces, at whitespace, at punctuation and at
ASCII symbols. BPE then cuts each piece on its own, so a token never runs
across a space, a punctuation mark or an ASCII symbol. For example:

| text | pieces | tokens |
|---|---|---|
| `subject_id` | `subject`, `_`, `id` | `subject`, `_`, `id` |
| `subject id` | `subject`, `id` | `subject`, `id` |

Both texts get the tokens `subject` and `id`. The `_` is a token of its own.

#### Step 3: cut into tokens

BPE cuts each piece into tokens of the vocabulary.
[How BPE learns its vocabulary](#how-bpe-learns-its-vocabulary) explains how
BPE learns those tokens.

### How BPE learns its vocabulary

[BPE](https://huggingface.co/learn/llm-course/chapter6/5) starts from the single characters of the training texts. It then finds
the two adjacent tokens that occur together most often, and merges them into
a new token. It repeats this until the vocabulary has its size. A common word
ends as one token. BPE cuts a rare word into pieces that are tokens, so the
vocabulary still represents the rare word.

BPE never merges two tokens that occur together fewer times than the minimum
frequency. The minimum frequency is 2 by default, and `--min-frequency` sets
another.

### The vocabulary size

The vocabulary size is the main setting of a tokenizer:

- It sets the number of rows in the model's
  [table](../README.md#how-the-model-works): one row per token.
- A larger vocabulary keeps more words whole. But each rare token then
  appears in fewer training pairs, so its vector learns less.
- A smaller one cuts more words into pieces, and the model must learn their
  meaning from the pieces.

The default is 16,000 tokens. The README's small corpus, `example`, uses
8,000. To choose between sizes, train a model on each and
compare their MRR@10 on the validation pairs. Each run writes its validation
MRR@10 in its [`manifest.json`](training.md#manifestjson).

A small corpus can give fewer tokens than asked for. `tokenizer train` then
prints a note, and the manifest holds the vocabulary size reached.

### Special tokens

Every tokenizer has five special tokens, with the first five ids: `[PAD]`,
`[UNK]`, `[CLS]`, `[SEP]` and `[MASK]`. A static model uses one of them,
`[UNK]`, which stands for a character that the training texts never had.
The [`StaticEmbedding`](https://sbert.net/docs/package_reference/sentence_transformer/modules.html#sentence_transformers.sentence_transformer.modules.StaticEmbedding) module of sentence-transformers,
which Hugging Face maintains,
[pads nothing](https://github.com/huggingface/sentence-transformers/blob/v6.0.1/sentence_transformers/sentence_transformer/modules/static_embedding.py#L106).

The [BERT](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/bert)
model here uses `[UNK]` and `[PAD]`. `[PAD]` fills a short text in a batch.
`[CLS]`, `[SEP]` and `[MASK]` are for a later BERT setting, for example
masked-language pretraining.
[Five special tokens](design_decisions.md#five-special-tokens) says why.

### What it learns from

The tokenizer learns from the corpus's **training pairs**: their distinct
texts, the [anchors](pairs.md) first, then the positives, each once.
Spellings that the tokenizer normalises to one text count once: `BMI` and
`bmi` are one text.

### Diagnostics

After training, `tokenizer train` checks how the tokenizer cuts its training
texts. Here a word is a piece from [step 2](#step-2-split). So `SUBJECT_ID`
holds three words: `subject`, `_` and `id`.

| measure | what it says | a bad value |
|---|---|---|
| fertility | tokens per word | high: many words are cut into pieces, so the vocabulary does not fit the texts |
| words split | the share of words cut into two or more tokens | a large share, for the same reason |
| unknown rate | the share of tokens that are `[UNK]` | above 0: the training texts have characters that the vocabulary lacks, which is a sign of a broken tokenizer |
| vocabulary used | the share of the vocabulary's tokens that the training texts use, not counting the special tokens | well below 1: many tokens never occur, so their vectors keep their random start |

The diagnostics show whether the tokenizer works. A broken tokenizer shows
at once, for example as unknown tokens or a high fertility. Fertility and the
share of words split are the measures of [Rust et al. (2021)](https://doi.org/10.18653/v1/2021.acl-long.243).

**Reference:**

> Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How
> Good is Your Tokenizer? On the Monolingual Performance of Multilingual
> Language Models. *Proceedings of the 59th Annual Meeting of the Association
> for Computational Linguistics and the 11th International Joint Conference on
> Natural Language Processing (Volume 1: Long Papers)*, 3118–3135.
> https://doi.org/10.18653/v1/2021.acl-long.243

### Names

A tokenizer's default name is its corpus's name and its vocabulary size, in
thousands when it can be: `all-sources-16k`, `example-8k`, `example-12500`.
`--name` gives another name.
