"""The architectures that ``train`` builds, by the name that a config gives.

Each architecture is a module with two functions:

- ``build(settings, tokenizer)``: an untrained SentenceTransformer;
- ``features(token_ids, **kwargs)``: the batch that the model reads, from
  texts already turned into token ids. The trainer's data collator calls it.

The trainer trains every architecture the same way.
"""

from embedding_model.architectures import static

ARCHITECTURES = {"static": static}
