# Tokenizer

A tokenizer cuts a text into tokens: words, pieces of words and punctuation
marks. Each token in its vocabulary has a number, the token's id. `tokenizer
train` learns the vocabulary from a [corpus](corpus.md)'s training pairs, by
[byte-pair encoding (BPE)](https://huggingface.co/learn/llm-course/chapter6/5):

```
python -m embedding_model tokenizer train --corpus all-sources                # 16,000 tokens: all-sources-16k
python -m embedding_model tokenizer train --corpus example --vocab-size 8000  # a small corpus: example-8k
```

The tokenizer gives the [model](../README.md#how-the-model-works) one id per
token, and the model sees only these ids. Each token of the vocabulary has its
own id. So a vocabulary of 8,000 tokens has the ids 0 to 7,999. For example,
with made-up ids:

| text | tokens | ids |
|---|---|---|
| `Body Weight` | `body`, `weight` | 812, 1407 |
| `body weight` | `body`, `weight` | 812, 1407 |
| `body_weight` | `body`, `_`, `weight` | 812, 63, 1407 |
| `heartrate` | `heart`, `rate` | 955, 2031 |

The first two texts get the same ids, so the model cannot tell them apart.
The last one is cut into two tokens when `heartrate` is not a token itself.

## What a tokenizer does

It cuts a text into tokens in three steps:

| step | what it does | example |
|---|---|---|
| 1. normalize | removes control and format characters, as BERT does; splits each accented letter into the letter and its accent (Unicode's NFD form); lowercases; and drops the accents | `Café` -> `cafe` |
| 2. split | splits at whitespace and at every punctuation character, as BERT does (`BertPreTokenizer`) | `SUBJECT_ID (float)` -> `subject`, `_`, `id`, `(`, `float`, `)` |
| 3. cut into tokens | cuts each piece into vocabulary tokens, by byte-pair encoding (BPE) | `heartrate` -> `heart`, `rate`, if `heartrate` is not a token itself |

The same rules run at two times:

- **When the tokenizer learns its vocabulary.** Steps 1 and 2 prepare each
  training text, and BPE learns its tokens from the pieces.
- **Every time the model embeds a text.** `tokenizer.json` saves the rules
  with the vocabulary, and the trained model keeps that file. All three steps
  then cut each text that the model reads.

So a text that the model reads later is normalised and split exactly as the
training texts were. The section
[What a tokenizer holds](#what-a-tokenizer-holds) has more details.

### Step 1: normalize

The tokenizer first removes control and format characters. They hold no
text, and nobody can see them. For example, a description copied from a PDF
can read `body\u200bweight`, where `\u200b` is a zero-width space. On the
screen it looks like `bodyweight`.

| text | tokens without this step | tokens with it |
|---|---|---|
| `body\u200bweight` | `body`, `[UNK]`, `weight` | `body`, `weight` |

The vocabulary has no token for the invisible character. Without this step,
the character would become the unknown token, `[UNK]`.

NFD stands for Normalization Form D, one of the four forms that Unicode
defines in [Unicode Normalization Forms](https://www.unicode.org/reports/tr15/#Norm_Forms).
D is for canonical decomposition. It writes each accented letter as two
characters: the plain letter, then a separate accent mark. So `é` becomes
`e` followed by the accent. NFD makes the accents removable. Without it, `é`
stays one character, and there is no separate accent to drop.

Lowercasing makes `BMI`, `Bmi` and `bmi` one text. Variable names come in
any case, and their case carries no meaning. So the three get the same
tokens, and the model gives them the same embedding.

### Step 2: split

The split cuts a text into pieces, at whitespace and at punctuation. BPE then
cuts each piece on its own, so a token never runs across a space or a
punctuation mark. For example:

| text | pieces | tokens |
|---|---|---|
| `subject_id` | `subject`, `_`, `id` | `subject`, `_`, `id` |
| `subject id` | `subject`, `id` | `subject`, `id` |

Both texts get the tokens `subject` and `id`. The `_` is a token of its own.

### Step 3: cut into tokens

BPE cuts each piece into tokens of the vocabulary. The next section explains
how BPE learns those tokens.

## How BPE learns its vocabulary

BPE starts from the single characters of the training texts. It then finds
the two adjacent tokens that occur together most often, and merges them into
a new token. It repeats this until the vocabulary has its size. A common word
ends as one token. A rare word is cut into pieces that are tokens, so it is
still represented. Two tokens that occur together fewer times than the
minimum frequency are never merged. The minimum frequency is 2 by default, and
`--min-frequency` sets another.

## The vocabulary size

The vocabulary size is the main setting of a tokenizer:

- It sets the number of rows in the model's
  [table](../README.md#how-the-model-works): one row per token.
- A larger vocabulary keeps more words whole. But each rare token then
  appears in fewer training pairs, so its vector learns less.
- A smaller one cuts more words into pieces, and the model must learn their
  meaning from the pieces.

The default is 16,000 tokens. A small corpus, such as the README's example,
uses 8,000. To choose between sizes, train a model on each and compare their
scores on a benchmark.

A small corpus can give fewer tokens than asked for, so the manifest records
the size that was reached.

## Special tokens

Every tokenizer has five special tokens, with the first five ids: `[PAD]`,
`[UNK]`, `[CLS]`, `[SEP]` and `[MASK]`. A static model uses two of them:

- `[PAD]` fills a short text in a batch;
- `[UNK]` stands for a character that the training texts never had.

The other three come from BERT's convention. A text that holds a special
token's name, such as `[PAD]`, is read as that token, as in every Hugging
Face tokenizer.

## What it learns from

The tokenizer learns from the corpus's **training pairs**: their distinct
texts, the anchors first, then the positives, each once. A text counts once
in all its spellings: `BMI` and `bmi` are one text.

## Diagnostics

After training, `tokenizer train` checks how the tokenizer cuts its training
texts. Here a word is a piece of [step 2](#step-2-split), so `SUBJECT_ID`
holds three words.

| measure | what it says | a bad value |
|---|---|---|
| fertility | tokens per word | well above 2: many words are cut into pieces, so the vocabulary does not fit the texts |
| words split | the share of words cut into two or more tokens | a large share, for the same reason |
| unknown rate | the share of tokens that are `[UNK]` | above 0: the training texts have characters that the vocabulary lacks, which is a sign of a broken tokenizer |
| vocabulary used | the share of the vocabulary's tokens that the training texts use | well below 1: many tokens never occur, so their vectors keep their random start |

The diagnostics show whether the tokenizer works. A broken tokenizer shows
at once, for example as unknown tokens or a high fertility.

## What a tokenizer holds

`tokenizer train` writes to `embedding_model/inputs/tokenizers/<name>/`, or
under the folder that `--inputs` names. It writes to a staging folder first,
so a failed training leaves the previous tokenizer as it was.

### `tokenizer.json`

The tokenizer's rules and its vocabulary, in one file. Hugging Face
`tokenizers` reads it, with `Tokenizer.from_file("tokenizer.json")`, and runs
its own code for each rule that the file names. The file holds:

| field | what it holds | where it comes from |
|---|---|---|
| `normalizer` | [step 1](#step-1-normalize): BERT's cleaning, NFD, lowercase, no accents | the code, fixed |
| `pre_tokenizer` | [step 2](#step-2-split): `BertPreTokenizer` | the code, fixed |
| `model` | [step 3](#step-3-cut-into-tokens): BPE's tokens and merges | learned from the training texts |
| `added_tokens` | the [special tokens](#special-tokens) | the code, fixed |

### `manifest.json`

The tokenizer's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: `tokenizer.json`, with its SHA-256;
- `sources`: the corpus, with its kept pairs' checksums;
- `settings`: the vocabulary size asked for, the minimum frequency, and the
  version of `tokenizers`, since the bytes of `tokenizer.json` depend on it.
  `tokenizer.json` itself holds the rules;
- `counts`: the training texts, the tokens learned, and the diagnostics.

The corpus's pairs must have the checksum that the corpus's manifest records.
Otherwise the training stops, and asks for the corpus to be built again.

## Names

A tokenizer's default name is its corpus's name and its vocabulary size, in
thousands when it can be: `all-sources-16k`, `example-8k`, `example-12500`.
`--name` gives another name.
