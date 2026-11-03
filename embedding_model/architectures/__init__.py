"""The architectures that ``train`` builds, by the name that a config gives.

Each architecture is a module with:

- ``Settings``: the dataclass of its own config section, named after it;
- ``LEARNING_RATE``: its default learning rate;
- ``build(settings, tokenizer)``: an untrained SentenceTransformer;
- ``features``: a function that builds the batch that the model reads, from
  texts already turned into token ids, or None. With a function, a run turns
  its texts into token ids once, and the trainer's data collator calls it.
  With None, the model's own ``preprocess`` tokenizes each batch of texts.

The trainer trains every architecture the same way.
"""

from embedding_model.architectures import bert, static

ARCHITECTURES = {"static": static, "bert": bert}
