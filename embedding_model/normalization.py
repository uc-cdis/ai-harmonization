"""How a text is normalised, the same way in every step that compares texts.

The normalisation is BERT's text cleaning, Unicode's NFD, lowercase, and no
accents. It is used in three places:

1. Preparing the training pairs, in ``pairs prepare`` and ``corpus build``.
   ``pairs.common.clean`` removes invisible characters with
   ``BERT_CLEANING``. The rule that keeps each pair once,
   ``pairs.common.distinct_pairs``, and leakage prevention, through
   ``pairs.common.key``, compare texts after the whole normalisation.
2. Training the tokenizer. ``tokenizer train`` counts each training text
   once after this normalisation, and BPE learns the vocabulary from the
   normalised texts.
3. Using the trained model. The model keeps the tokenizer with these rules.
   Before it embeds any text, it normalises the text by the same rules.

So two texts that the tokenizer normalises to one text are one text in every
step.
"""

from tokenizers import normalizers

# BERT's text cleaning: it removes every control, format and private-use
# character, such as a byte left from a broken quote, a soft hyphen, a
# zero-width space or a direction mark. It turns every whitespace character,
# such as a tab, a line break or a no-break space, into a plain space.
BERT_CLEANING = normalizers.BertNormalizer(
    clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False
)
# The tokenizer's normaliser: BERT's text cleaning, Unicode's NFD, lower case,
# and no accents.
TOKENIZER_NORMALIZER = normalizers.Sequence(
    [
        BERT_CLEANING,
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
    ]
)


def normalised(texts):
    """Each text as the tokenizer normalises it, each distinct text once.

    Args:
        texts (pd.Series): Texts.

    Returns:
        pd.Series: The normalised texts, with the index of ``texts``.
    """
    distinct = texts.unique()
    return texts.map(
        dict(zip(distinct, map(TOKENIZER_NORMALIZER.normalize_str, distinct)))
    )
