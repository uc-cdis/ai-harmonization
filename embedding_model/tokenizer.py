"""Tokenizers: the vocabulary that a static model's table is built on.

``tokenizer train`` learns a vocabulary from a corpus's training pairs and
writes one tokenizer under ``<inputs>/tokenizers/<name>/``: ``tokenizer.json``,
which Hugging Face ``tokenizers`` reads, and ``manifest.json``.

The tokenizer works in three steps:

1. Text is cleaned as BERT cleans it, which removes control and format
   characters, such as a zero-width space. It is decomposed into letters and
   accents (NFD), lowercased, and stripped of its accents: ``Café`` becomes
   ``cafe``.
2. It is split at whitespace and at every punctuation character, as
   ``BertPreTokenizer`` splits it: ``SUBJECT_ID`` becomes ``subject``, ``_``
   and ``id``.
3. Each piece is cut into vocabulary tokens by byte-pair encoding (BPE).

The tokenizer uses BPE, since its training is reproducible: the same texts
give the same tokens and the same ids. A token's id picks its row in the
model's table, so other ids would give a model another starting point.

Only the training pairs' texts shape the vocabulary. The diagnostics are
measured on the training texts too: they are a sanity check that the
tokenizer is not broken.
"""

import itertools
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from embedding_model.corpus import TRAIN, read_corpus
from embedding_model.download import file_sha256
from embedding_model.folders import FOLDER_NAME, staged
from embedding_model.normalization import TOKENIZER_NORMALIZER, normalised

# The special tokens of every tokenizer here, in id order. A static model uses
# only [PAD] and [UNK]. The other three follow BERT's convention, and stay so
# that the special tokens' ids never change.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
UNKNOWN = "[UNK]"
VOCAB_SIZE = 16_000
MIN_FREQUENCY = 2
# How many texts are encoded at once for the diagnostics, to bound memory.
BATCH_SIZE = 100_000


def default_name(corpus, vocab_size):
    """A tokenizer's name: its corpus and its vocabulary size, such as ``example-8k``.

    A size that is not a multiple of 1,000 keeps its full number, such as
    ``example-12500``.
    """
    size = f"{vocab_size // 1000}k" if vocab_size % 1000 == 0 else str(vocab_size)
    return f"{corpus}-{size}"


def training_texts(pairs):
    """The training pairs' distinct texts: the anchors, then the positives.

    Each text comes once, as it is first written. Two texts are one when the
    tokenizer normalises them to one text, as the other steps compare texts:
    ``BMI`` and ``bmi`` count once. The order does not change the vocabulary,
    since BPE counts each word's frequency. It is fixed so that the input is
    the same on every run.

    Args:
        pairs (pd.DataFrame): A corpus's kept pairs, with ``split``.

    Returns:
        list[str]: The texts.
    """
    is_training = pairs.split == TRAIN
    texts = pd.concat(
        [pairs.anchor[is_training], pairs.positive[is_training]], ignore_index=True
    )
    return texts[~normalised(texts).duplicated()].tolist()


def new_tokenizer():
    """An untrained tokenizer, with the three steps that the module describes.

    It has no decoder yet: training and using the model never turn tokens
    back into text, so none is needed. One may be added later.

    Returns:
        Tokenizer: The tokenizer, with no vocabulary yet.
    """
    tokenizer = Tokenizer(models.BPE(unk_token=UNKNOWN))
    tokenizer.normalizer = TOKENIZER_NORMALIZER
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return tokenizer


def learn(texts, vocab_size, min_frequency):
    """A tokenizer trained on texts.

    The vocabulary always holds the special tokens and every character of the
    texts. So a small corpus can give fewer tokens than asked for, and a size
    below the special tokens plus the characters gives more.

    Args:
        texts (list[str]): The texts to learn the vocabulary from.
        vocab_size (int): The vocabulary's size, special tokens included.
        min_frequency (int): How often a merge must occur to become a token.

    Returns:
        Tokenizer: The trained tokenizer.
    """
    tokenizer = new_tokenizer()
    tokenizer.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        ),
    )
    return tokenizer


def diagnostics(tokenizer, texts):
    """How a tokenizer cuts texts: a sanity check, not a measure of quality.

    A word is a piece after the split at whitespace and punctuation, so
    ``SUBJECT_ID`` holds three words. Fertility near 1 means that words stay
    whole. A high share of words split, or a high fertility, means that the
    vocabulary does not fit the texts. Unknown tokens mean characters that the
    training texts never had. The texts are encoded without special tokens, as
    a static model encodes them.

    Args:
        tokenizer (Tokenizer): The tokenizer.
        texts (list[str]): The texts to cut.

    Returns:
        dict: Four numbers:

            - ``fertility``: the tokens per word;
            - ``words_split``: the share of words cut into two or more tokens;
            - ``unknown_rate``: the share of tokens that are UNKNOWN;
            - ``vocabulary_used``: the share of the vocabulary's tokens,
              specials left out, that the texts use.
    """
    unknown_id = tokenizer.token_to_id(UNKNOWN)
    special_ids = {tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    token_count = word_count = split_word_count = unknown_count = 0
    used_ids = set()
    for batch in itertools.batched(texts, BATCH_SIZE):
        for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
            ids = encoding.ids
            per_word = Counter(encoding.word_ids)
            token_count += len(ids)
            word_count += len(per_word)
            split_word_count += sum(count > 1 for count in per_word.values())
            unknown_count += ids.count(unknown_id)
            used_ids.update(ids)
    non_special_count = tokenizer.get_vocab_size() - len(special_ids)
    return {
        "fertility": round(token_count / max(word_count, 1), 3),
        "words_split": round(split_word_count / max(word_count, 1), 4),
        "unknown_rate": round(unknown_count / max(token_count, 1), 5),
        "vocabulary_used": round(
            len(used_ids - special_ids) / max(non_special_count, 1), 4
        ),
    }


def manifest(
    *,
    name,
    corpus_manifest,
    vocab_size,
    min_frequency,
    tokenizer_sha256,
    training_text_count,
    learned_vocab_size,
    measured,
):
    """The tokenizer's manifest, for ``manifest.json``.

    Args:
        name (str): The tokenizer's name.
        corpus_manifest (dict): The manifest of the corpus that it learned
            from, from ``read_corpus``.
        vocab_size (int): The vocabulary size asked for.
        min_frequency (int): The minimum frequency of a merge.
        tokenizer_sha256 (str): The SHA-256 of the written ``tokenizer.json``.
        training_text_count (int): How many training texts it learned from.
        learned_vocab_size (int): The vocabulary size reached, special tokens
            included.
        measured (dict): The diagnostics, from ``diagnostics``.

    Returns:
        dict: The manifest.
    """
    return {
        "tokenizer": name,
        "built": date.today().isoformat(),
        "files": {"tokenizer.json": {"sha256": tokenizer_sha256}},
        "sources": [
            {
                "role": "corpus",
                "name": corpus_manifest["corpus"],
                "files": {"pairs.parquet": corpus_manifest["files"]["pairs.parquet"]},
            }
        ],
        "settings": {
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "tokenizers": tokenizers.__version__,
        },
        "counts": {
            "training_texts": training_text_count,
            "vocabulary": learned_vocab_size,
            "diagnostics": measured,
        },
    }


def train(corpus, name, vocab_size, min_frequency, inputs):
    """Learn a corpus's vocabulary, and write the tokenizer with its manifest.

    The tokenizer is written to a staging folder first, and replaces the last
    one of that name only once it is complete.

    Args:
        corpus (str): The corpus's name, under ``<inputs>/corpora/``.
        name (str | None): The tokenizer's name; ``default_name`` when None.
        vocab_size (int): The vocabulary's size.
        min_frequency (int): How often a merge must occur to become a token.
        inputs (Path): The inputs folder. The tokenizer goes to its
            ``tokenizers/<name>/``.

    Returns:
        int: 0 if the tokenizer was written, 1 if a name, a setting or the
            corpus stops it.

    Raises:
        Exception: If the tokenizer cannot be written or moved into place. The
            last tokenizer of that name is then kept. ``tokenizers`` raises a
            plain Exception when it cannot write.
    """
    if not FOLDER_NAME.fullmatch(corpus):
        print(f"{corpus!r} is not a corpus name", file=sys.stderr)
        return 1
    if vocab_size <= len(SPECIAL_TOKENS) or min_frequency < 1:
        print(
            f"the vocabulary size must be above {len(SPECIAL_TOKENS)}, and the"
            " minimum frequency at least 1",
            file=sys.stderr,
        )
        return 1
    if name is None:
        name = default_name(corpus, vocab_size)
    if not FOLDER_NAME.fullmatch(name):
        print(
            f"{name!r} is not a tokenizer name: use letters, digits, '.', '_', '-'",
            file=sys.stderr,
        )
        return 1
    try:
        pairs, corpus_manifest = read_corpus(inputs, corpus)
    except Exception as error:
        print(error, file=sys.stderr)
        return 1
    texts = training_texts(pairs)
    # The pairs are not needed past this point. Freeing them lowers the peak
    # memory while the tokenizer trains.
    del pairs
    if not texts:
        print(f"corpus {corpus!r} has no training pairs", file=sys.stderr)
        return 1
    print(f"  corpus {corpus}: {len(texts):,} distinct training texts")
    tokenizer = learn(texts, vocab_size, min_frequency)
    measured = diagnostics(tokenizer, texts)
    destination = Path(inputs) / "tokenizers" / name
    with staged(destination) as built:
        tokenizer.save(str(built / "tokenizer.json"))
        written = manifest(
            name=name,
            corpus_manifest=corpus_manifest,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            tokenizer_sha256=file_sha256(built / "tokenizer.json"),
            training_text_count=len(texts),
            learned_vocab_size=tokenizer.get_vocab_size(),
            measured=measured,
        )
        (built / "manifest.json").write_text(
            json.dumps(written, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"  {tokenizer.get_vocab_size():,} tokens -> {destination}\n"
        f"  on the training texts: fertility {measured['fertility']}, "
        f"{measured['words_split']:.1%} of words split, "
        f"unknown rate {measured['unknown_rate']}, "
        f"{measured['vocabulary_used']:.1%} of the vocabulary used"
    )
    return 0
