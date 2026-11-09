"""A transformer embedding model with BERT's architecture, trained from scratch.

The model is a BERT body that starts from random weights, followed by mean
pooling over its token vectors. Nothing of a pretrained BERT is used: only the
architecture, ``BertModel`` of transformers. The model trains on the texts
themselves. sentence-transformers' ``Transformer`` module tokenizes each
batch, cuts each text to ``max_position_embeddings`` tokens, and pads the
batch.
"""

import functools
from dataclasses import asdict, dataclass, field

# The BERT paper pretrains at this rate (Devlin et al., 2019, Appendix A.2).
LEARNING_RATE = 1e-4


@functools.cache
def bert_config():
    """BertConfig at its defaults, built once."""
    from transformers import BertConfig

    return BertConfig()


def bert_default(name, help_text):
    """A setting whose default is BertConfig's own, read when it is needed.

    So the defaults are BertConfig's, without a copy of its numbers, and
    transformers is imported only when a config is read.

    Args:
        name (str): The field of BertConfig.
        help_text (str): What the setting sets, for ``--help``.

    Returns:
        dataclasses.Field: The field.
    """

    def default():
        """BertConfig's default for the setting."""
        return getattr(bert_config(), name)

    return field(default_factory=default, metadata={"help": help_text})


@dataclass
class Settings:
    """A BERT's settings, in the config's model section, under BertConfig's names.

    The shape's defaults are BERT-Small's, from Google's release of smaller
    BERT models (Turc et al., 2019): 4 layers of 512 dimensions, with 8
    attention heads. The feed-forward part is 4 times the width, as the BERT
    paper sets it. A text is cut at 128 tokens, the length at which the BERT
    paper pretrains for most of its steps. A batch is padded to its longest
    text, and the BERT paper notes that "attention is quadratic to the
    sequence length". The other defaults are BertConfig's.
    """

    hidden_size: int = field(
        default=512,
        metadata={"help": "the length of each token's vector, and of the embedding"},
    )
    num_hidden_layers: int = field(
        default=4, metadata={"help": "the number of transformer layers"}
    )
    num_attention_heads: int = field(
        default=8, metadata={"help": "the attention heads of each layer"}
    )
    intermediate_size: int = field(
        default=2048, metadata={"help": "the size of each layer's feed-forward part"}
    )
    hidden_dropout_prob: float = bert_default(
        "hidden_dropout_prob", "the dropout on the embeddings and each layer's output"
    )
    attention_probs_dropout_prob: float = bert_default(
        "attention_probs_dropout_prob", "the dropout on the attention weights"
    )
    max_position_embeddings: int = field(
        default=128,
        metadata={"help": "the longest text, in tokens; a longer one is cut"},
    )
    initializer_range: float = bert_default(
        "initializer_range", "the standard deviation of the random starting weights"
    )

    def __post_init__(self):
        """Refuse a width that the attention heads do not divide.

        BertModel refuses it too, but only when it is built, after the run has
        read its corpus.

        Raises:
            ValueError: If ``hidden_size`` is not a multiple of
                ``num_attention_heads``.
        """
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size, {self.hidden_size}, is not a multiple of"
                f" num_attention_heads, {self.num_attention_heads}"
            )


def build(settings, tokenizer):
    """An untrained BERT body over a tokenizer's vocabulary, with mean pooling.

    sentence-transformers' ``Transformer`` loads a model only from a folder.
    So the random model and its tokenizer are written to a temporary folder
    first, and loaded from it.

    Args:
        settings (Settings): The BERT's settings.
        tokenizer (Tokenizer): The tokenizer, from ``tokenizer.read_tokenizer``.

    Returns:
        SentenceTransformer: The model: its Transformer, then its Pooling.
    """
    import tempfile

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    config = BertConfig(
        **asdict(settings),
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("[PAD]"),
    )
    with tempfile.TemporaryDirectory() as folder:
        BertModel(config).save_pretrained(folder)
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        ).save_pretrained(folder)
        body = Transformer(folder, max_seq_length=settings.max_position_embeddings)
    model = SentenceTransformer(
        modules=[body, Pooling(settings.hidden_size, pooling_mode="mean")]
    )
    # The model card would look up the temporary folder's path on the Hugging
    # Face Hub, as the id of a base model. The path names no model.
    model.model_card_data.local_files_only = True
    return model
