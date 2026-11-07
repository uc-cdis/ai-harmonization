"""The architectures that ``train`` builds, by the name that a config gives.

Each architecture is a module with:

- ``Settings``: the dataclass of its own config section, named after it;
- ``LEARNING_RATE``: its default learning rate;
- ``build(settings, tokenizer)``: an untrained SentenceTransformer;
- ``features(token_ids, **kwargs)``, if the architecture reads token ids:
  the batch that the model reads, from texts already turned into token ids.
  A run then turns its texts into token ids once, and the trainer's data
  collator calls ``features``. An architecture without ``features`` trains
  on the texts: the model's own ``preprocess`` tokenizes each batch.

The trainer trains every architecture the same way.
"""

from embedding_model.architectures import bert, static

ARCHITECTURES = {"static": static, "bert": bert}
