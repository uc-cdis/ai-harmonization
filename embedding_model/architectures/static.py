"""A static embedding table: one vector per token, averaged over a text's tokens.

sentence-transformers' ``StaticEmbedding`` holds the table, in an
``nn.EmbeddingBag`` that averages the vectors of a text's tokens. It starts
from the EmbeddingBag's random values, so the seed set before it is built
decides where training starts.
"""

from dataclasses import dataclass, field

# Hugging Face's recipe for a static model trains at this rate.
LEARNING_RATE = 0.2


@dataclass
class Settings:
    """A static model's settings, in the config's model section: its vectors' length."""

    embedding_dim: int = field(
        default=1024, metadata={"help": "the length of each embedding"}
    )


def build(settings, tokenizer):
    """An untrained static model over a tokenizer's vocabulary.

    Args:
        settings (Settings): The static model's settings; ``embedding_dim`` is the
            length of each token's vector.
        tokenizer (Tokenizer): The tokenizer, from ``tokenizer.read_tokenizer``.

    Returns:
        SentenceTransformer: The model, with one module, its StaticEmbedding.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding

    return SentenceTransformer(
        modules=[StaticEmbedding(tokenizer, embedding_dim=settings.embedding_dim)]
    )


def features(token_ids, **kwargs):
    """The batch that StaticEmbedding reads, built from token ids.

    The trainer's data collator calls this function on each batch, in place
    of ``StaticEmbedding.preprocess``. ``preprocess`` takes texts, and
    encodes them first. A run encodes its texts once, before training, so
    each batch already holds token ids. This function builds from them what
    ``preprocess`` builds after encoding. That is one flat tensor of all the
    batch's ids, ``input_ids``, and where each text starts in it,
    ``offsets``.

    Args:
        token_ids (list[list[int]]): Each text's token ids, without special
            tokens.
        **kwargs: The collator's other arguments, ``prompt`` and ``task``. A
            run refuses the ``prompts`` setting, so the prompt is None, and a
            static model has no task.

    Returns:
        dict[str, torch.Tensor]: ``input_ids`` and ``offsets``.
    """
    import numpy as np
    import torch

    lengths = [len(ids) for ids in token_ids]
    return {
        "input_ids": torch.tensor(
            [token for ids in token_ids for token in ids], dtype=torch.long
        ),
        "offsets": torch.from_numpy(np.cumsum([0] + lengths[:-1])),
    }
