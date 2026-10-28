"""Tests for embedding_model.tokenizer: a tokenizer trained on a small fictional corpus."""

import json

import pandas as pd
import pytest
import tokenizers
from tokenizers import Tokenizer

from embedding_model import corpus, tokenizer
from embedding_model.__main__ import main
from embedding_model.manifests import file_sha256, sha256
from embedding_model.pairs import pairs_sha256
from embedding_model.pairs.common import COLUMNS

TRAINING = [
    ("abcd score", "Abcd Score (float)"),
    ("abcd index", "Abcd Index (float)"),
    ("efgh level", "Efgh Level (string)"),
    ("efgh score", "Efgh Score (float)"),
    ("Café SUBJECT_ID", "Abcd Score (float)"),
]
# Validation texts use letters that no training text has.
VALIDATION = [("qwkz", "Qwkz (string)")]


def pair(anchor, positive, split):
    """One kept pair, with every column of a corpus's pairs."""
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        anchor=anchor,
        positive=positive,
        source="abcd",
        kind="name",
        anchor_values=[],
        positive_values=[],
        split=split,
    )
    return row


def write_corpus(inputs, name, rows):
    """Write a fictional corpus's pairs and its manifest, as ``corpus build`` would."""
    folder = inputs / "corpora" / name
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(folder / "pairs.parquet", index=False)
    written = pd.read_parquet(folder / "pairs.parquet")
    record = {
        "corpus": name,
        "files": {
            "pairs.parquet": {
                "rows": len(written),
                "sha256": file_sha256(folder / "pairs.parquet"),
                "csv_sha256": pairs_sha256(written),
            }
        },
    }
    (folder / "manifest.json").write_text(json.dumps(record))


def corpus_rows():
    """The fictional corpus's pairs: the training pairs, then the validation pair."""
    rows = [pair(*texts, corpus.TRAIN) for texts in TRAINING]
    return rows + [pair(*texts, corpus.VALIDATION) for texts in VALIDATION]


@pytest.fixture
def inputs(tmp_path):
    """An inputs folder with the fictional corpus ``abcd``."""
    write_corpus(tmp_path, "abcd", corpus_rows())
    return tmp_path


class TestTokenizer:
    """The tokenizer's steps, its training texts and its diagnostics."""

    def test_text_is_cleaned_lowercased_stripped_of_accents_and_split(self):
        """BERT's cleaning, NFD, lowercase and no accents, then a split at punctuation."""
        untrained = tokenizer.new_tokenizer()
        assert untrained.normalizer.normalize_str("Café Größe") == "cafe große"
        assert untrained.normalizer.normalize_str("Ab\u200bcd\u202c") == "abcd"
        pieces = untrained.pre_tokenizer.pre_tokenize_str("subject_id (float)")
        assert [piece for piece, _ in pieces] == [
            "subject",
            "_",
            "id",
            "(",
            "float",
            ")",
        ]

    def test_training_texts_are_the_training_pairs_distinct_texts(self):
        """Anchors first, then positives, each once, without validation pairs."""
        texts = tokenizer.training_texts(pd.DataFrame(corpus_rows()))
        assert texts == [anchor for anchor, _ in TRAINING] + [
            "Abcd Score (float)",
            "Abcd Index (float)",
            "Efgh Level (string)",
            "Efgh Score (float)",
        ]

    def test_spellings_of_one_text_count_once(self):
        """Training texts that normalise to one text count once, as first written."""
        found = pd.DataFrame(
            [pair("BMI", "Abcd", corpus.TRAIN), pair("bmi", "Ábcd", corpus.TRAIN)]
        )
        assert tokenizer.training_texts(found) == ["BMI", "Abcd"]

    def test_decoding_shows_the_tokens(self):
        """Without a decoder, decoding joins the tokens with spaces, words kept apart."""
        trained = tokenizer.learn(["body weight", "body_weight"], 100, 2)
        assert trained.decode(trained.encode("Body Weight").ids) == "body weight"
        assert trained.decode(trained.encode("body_weight").ids) == "body _ weight"

    def test_the_same_texts_give_the_same_tokenizer(self):
        """BPE training is reproducible, ids included."""
        texts = [text for row in TRAINING for text in row]
        first = tokenizer.learn(texts, 100, 2)
        second = tokenizer.learn(texts, 100, 2)
        assert first.to_str() == second.to_str()

    def test_the_diagnostics(self):
        """Whole words give fertility 1; a word cut into 4 tokens gives 4; unknown characters show."""
        trained = tokenizer.learn(["abcd abcd", "abcd abcd"], 100, 2)
        whole = tokenizer.diagnostics(trained, ["abcd abcd"])
        assert (whole["fertility"], whole["words_split"]) == (1.0, 0.0)
        assert whole["unknown_rate"] == 0.0
        tokens = trained.get_vocab_size() - len(tokenizer.SPECIAL_TOKENS)
        assert whole["vocabulary_used"] == round(1 / tokens, 4)
        split = tokenizer.diagnostics(trained, ["dcba"])
        assert (split["fertility"], split["words_split"]) == (4.0, 1.0)
        assert tokenizer.diagnostics(trained, ["wxyz"])["unknown_rate"] == 1.0

    def test_the_diagnostics_are_the_same_in_batches(self, monkeypatch):
        """Texts encoded in several batches give the same diagnostics as one batch."""
        texts = [text for row in TRAINING for text in row]
        trained = tokenizer.learn(texts, 100, 2)
        whole = tokenizer.diagnostics(trained, texts)
        monkeypatch.setattr(tokenizer, "BATCH_SIZE", 2)
        assert tokenizer.diagnostics(trained, texts) == whole

    def test_default_names(self):
        """A default name is the corpus and the size, in thousands when it can be."""
        assert tokenizer.default_name("abcd", 16_000) == "abcd-16k"
        assert tokenizer.default_name("abcd", 8_000) == "abcd-8k"
        assert tokenizer.default_name("abcd", 12_500) == "abcd-12500"


class TestTrain:
    """``tokenizer train`` on a fictional inputs folder."""

    def test_the_tokenizer_and_its_manifest(self, inputs, capsys):
        """The tokenizer is written with a manifest of its corpus, settings and counts."""
        assert tokenizer.train("abcd", None, 100, 2, inputs) == 0
        folder = inputs / "tokenizers" / "abcd-100"
        trained = Tokenizer.from_file(str(folder / "tokenizer.json"))
        written = json.loads((folder / "manifest.json").read_text())
        corpus_manifest = json.loads(
            (inputs / "corpora" / "abcd" / "manifest.json").read_text()
        )
        assert written["tokenizer"] == "abcd-100"
        assert written["files"] == {
            "tokenizer.json": {
                "sha256": sha256((folder / "tokenizer.json").read_bytes())
            }
        }
        assert written["sources"] == [
            {
                "role": "corpus",
                "name": "abcd",
                "files": {"pairs.parquet": corpus_manifest["files"]["pairs.parquet"]},
            }
        ]
        assert written["settings"] == {
            "vocab_size": 100,
            "min_frequency": 2,
            "tokenizers": tokenizers.__version__,
        }
        assert [
            token.content for token in trained.get_added_tokens_decoder().values()
        ] == tokenizer.SPECIAL_TOKENS
        assert written["counts"]["training_texts"] == 9
        assert written["counts"]["vocabulary"] == trained.get_vocab_size()
        assert written["counts"]["diagnostics"]["unknown_rate"] == 0.0
        assert "9 distinct training texts" in capsys.readouterr().out

    def test_only_training_texts_shape_the_vocabulary(self, inputs):
        """A validation pair's letters are not in the vocabulary."""
        assert tokenizer.train("abcd", None, 100, 2, inputs) == 0
        trained = Tokenizer.from_file(
            str(inputs / "tokenizers" / "abcd-100" / "tokenizer.json")
        )
        assert trained.token_to_id("q") is None
        assert set(trained.encode("qwkz").tokens) == {tokenizer.UNKNOWN}

    def test_a_retrain_replaces_the_previous_tokenizer(self, inputs):
        """A second training of one name replaces the first, and leaves no work folder."""
        assert tokenizer.train("abcd", "abcd-tok", 100, 2, inputs) == 0
        assert tokenizer.train("abcd", "abcd-tok", 100, 3, inputs) == 0
        written = json.loads(
            (inputs / "tokenizers" / "abcd-tok" / "manifest.json").read_text()
        )
        assert written["settings"]["min_frequency"] == 3
        assert [path.name for path in (inputs / "tokenizers").iterdir()] == ["abcd-tok"]

    def test_a_missing_corpus_names_the_command(self, tmp_path, capsys):
        """A corpus that was not built stops the training, with the command to build it."""
        assert tokenizer.train("abcd", None, 100, 2, tmp_path) == 1
        assert "corpus build --name abcd" in capsys.readouterr().err

    def test_a_corpus_without_training_pairs_is_refused(self, tmp_path, capsys):
        """A corpus of validation pairs only gives no tokenizer."""
        write_corpus(
            tmp_path, "abcd", [pair(*texts, corpus.VALIDATION) for texts in VALIDATION]
        )
        assert tokenizer.train("abcd", None, 100, 2, tmp_path) == 1
        assert "has no training pairs" in capsys.readouterr().err
        assert not (tmp_path / "tokenizers" / "abcd-100").exists()

    def test_a_setting_out_of_range_is_refused(self, inputs, capsys):
        """A vocabulary of no more than the special tokens, or a frequency of 0."""
        assert tokenizer.train("abcd", None, 5, 2, inputs) == 1
        assert tokenizer.train("abcd", None, 100, 0, inputs) == 1
        error = capsys.readouterr().err
        assert "--vocab-size must be above 5" in error
        assert "--min-frequency must be at least 1" in error

    def test_fewer_tokens_than_asked_for_are_noted(self, inputs, capsys):
        """A small corpus that cannot fill the vocabulary says so."""
        assert tokenizer.train("abcd", None, 10_000, 2, inputs) == 0
        assert "fewer than the 10,000 asked for" in capsys.readouterr().out

    def test_a_name_that_is_not_a_folder_name_is_refused(self, inputs, capsys):
        """A tokenizer's and a corpus's names must be plain folder names."""
        assert tokenizer.train("abcd", "../abcd", 100, 2, inputs) == 1
        assert "not a tokenizer name" in capsys.readouterr().err
        assert tokenizer.train("../abcd", None, 100, 2, inputs) == 1
        assert "'../abcd' is not a corpus name: use letters, digits" in (
            capsys.readouterr().err
        )
        assert tokenizer.train("abcd", "", 100, 2, inputs) == 1
        assert "'' is not a tokenizer name" in capsys.readouterr().err

    def test_the_command(self, inputs):
        """``tokenizer train`` trains a tokenizer from the corpus named."""
        argv = ["tokenizer", "train", "--inputs", str(inputs), "--corpus", "abcd"]
        assert main([*argv, "--vocab-size", "100", "--min-frequency", "3"]) == 0
        written = json.loads(
            (inputs / "tokenizers" / "abcd-100" / "manifest.json").read_text()
        )
        assert written["settings"]["vocab_size"] == 100
        assert written["settings"]["min_frequency"] == 3
