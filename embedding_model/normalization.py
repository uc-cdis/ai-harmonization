"""How a text is normalised, the same way in every step that compares texts.

TOKENIZER_NORMALIZER normalises a text as BERT's tokenizer normalises it:
text cleaning, Unicode's NFD, lowercase, and no accents. Four places use the
normalisation:

1. Preparing the training pairs, in ``pairs prepare`` and ``corpus build``.
   ``pairs.common.clean`` removes invisible characters with
   ``BERT_CLEANING``. The rule that keeps each pair once,
   ``pairs.common.distinct_pairs``, compares texts after the whole
   normalisation. ``pairs.common.text_key`` starts from it too.
2. Training the tokenizer. ``tokenizer train`` counts each training text
   once after this normalisation, and BPE learns the vocabulary from the
   normalised texts.
3. Validation in ``train``. ``train.validation_texts`` keys each validation
   anchor and positive by its normalised text, so two spellings that the
   tokenizer reads as one are one text there.
4. Using the trained model. The model keeps the tokenizer with these rules.
   Before it embeds any text, it normalises the text by the same rules.

So two texts that the tokenizer normalises to one text are one text in every
step.
"""

from tokenizers import normalizers

# The text cleaning of BERT's tokenizer. It removes control, format and
# private-use characters. Examples are a byte left from a broken quote, a soft
# hyphen, a zero-width space and a direction mark. The cleaning turns a tab, a
# line feed, a carriage return or a no-break space into a plain space.
BERT_CLEANING = normalizers.BertNormalizer(
    clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False
)
# The tokenizer's normaliser, as BERT's tokenizer normalises a text for its
# uncased models: text cleaning, Unicode's NFD, lowercase, and no accents.
TOKENIZER_NORMALIZER = normalizers.Sequence(
    [
        BERT_CLEANING,
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
    ]
)


def normalised(texts):
    """Each text as the tokenizer normalises it.

    The normaliser runs only once for each distinct text.

    Args:
        texts (pd.Series): Texts.

    Returns:
        pd.Series: The normalised texts, with the index of ``texts``.
    """
    distinct = texts.unique()
    return texts.map(
        dict(zip(distinct, map(TOKENIZER_NORMALIZER.normalize_str, distinct)))
    )
