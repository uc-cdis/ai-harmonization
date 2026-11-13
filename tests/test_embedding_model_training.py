"""Tests for embedding_model.train: a static model trained on a small fictional corpus, on a CPU."""

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from embedding_model import benchmarks, corpus, scoring, texts, tokenizer, train
from embedding_model.__main__ import main
from embedding_model.manifests import file_sha256
from embedding_model.architectures import bert, static
from embedding_model.pairs import pairs_sha256
from embedding_model.pairs.common import COLUMNS

TRAINING = [
    ("Systolic blood pressure", "abcd_sbp (float): Systolic blood pressure"),
    ("Diastolic blood pressure", "abcd_dbp (float): Diastolic blood pressure"),
    ("Body mass index", "abcd_bmi (float): Body mass index"),
    ("Heart rate", "abcd_hr (integer): Heart rate"),
    ("Body weight", "abcd_weight (float): Body weight"),
    ("Body height", "abcd_height (float): Body height"),
    ("Fasting glucose", "abcd_glucose (float): Fasting glucose"),
    ("Total cholesterol", "abcd_chol (float): Total cholesterol"),
    ("Smoking status", "abcd_smoke (string): Smoking status"),
    ("Age at enrollment", "abcd_age (integer): Age at enrollment"),
    ("Sex at birth", "abcd_sex (string): Sex at birth"),
    ("Waist circumference", "abcd_waist (float): Waist circumference"),
]
VALIDATION = [
    ("Pulse rate", "abcd_pulse (integer): Pulse rate"),
    ("Blood glucose", "abcd_bg (float): Blood glucose"),
    ("Hip circumference", "abcd_hip (float): Hip circumference"),
    ("Body temperature", "abcd_temp (float): Body temperature"),
]
CORPUS = "abcd-corpus"
TOKENIZER = "abcd-tokenizer"
# Two epochs of three batches of four training pairs.
CONFIG = {
    "data": {"corpus": CORPUS, "tokenizer": TOKENIZER},
    "model": {"embedding_dim": 16},
    "training": {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "eval_strategy": "steps",
        "eval_steps": 2,
        "logging_steps": 1,
        "use_cpu": True,
        "dataloader_num_workers": 0,
    },
}


def pair(anchor, positive, split):
    """One kept pair, with every column of a corpus's pairs."""
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        anchor=anchor,
        positive=positive,
        source="abcd-source",
        kind="name_description",
        anchor_values=[],
        positive_values=[],
        split=split,
    )
    return row


def write_corpus(
    inputs, rows, checked=(("topmed", "1"),), text_format=texts.DEFAULT_FORMAT
):
    """Write the fictional corpus's pairs and its manifest, as ``corpus build`` would.

    Args:
        inputs (Path): The inputs folder.
        rows (list[dict]): The pairs.
        checked (tuple[tuple[str, str]]): The benchmarks that the corpus was
            checked against, each with its version.
        text_format (str | None): The pairs' text format, or None for a
            manifest that records none.
    """
    folder = inputs / "corpora" / CORPUS
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(folder / "pairs.parquet", index=False)
    written = pd.read_parquet(folder / "pairs.parquet")
    record = {
        "corpus": CORPUS,
        "files": {
            "pairs.parquet": {
                "rows": len(written),
                "sha256": file_sha256(folder / "pairs.parquet"),
                "content_sha256": pairs_sha256(written),
            }
        },
        "sources": [
            {"role": "benchmark", "name": name, "version": version, "files": {}}
            for name, version in checked
        ],
    }
    if text_format is not None:
        record["settings"] = {"format": text_format}
    (folder / "manifest.json").write_text(json.dumps(record))


def corpus_rows():
    """The fictional corpus's pairs: the training pairs, then the validation pairs."""
    rows = [pair(*pair_texts, corpus.TRAIN) for pair_texts in TRAINING]
    return rows + [pair(*pair_texts, corpus.VALIDATION) for pair_texts in VALIDATION]


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    """Every test sees no GPU, so that the training runs and the checks stay on the CPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture
def inputs(tmp_path):
    """An inputs folder with the fictional corpus and a tokenizer trained on it."""
    folder = tmp_path / "inputs"
    write_corpus(folder, corpus_rows())
    assert tokenizer.train(CORPUS, TOKENIZER, 300, 1, folder) == 0
    return folder


@pytest.fixture
def config(tmp_path):
    """The run's config, as a YAML file named abcd-run.yaml."""
    path = tmp_path / "abcd-run.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return path


def trained_tokenizer(inputs):
    """The tokenizer that the inputs fixture trained."""
    return tokenizer.read_tokenizer(inputs, TOKENIZER)[0]


def write_benchmark(inputs):
    """A fictional benchmark named topmed at version 1, of vital signs that the corpus does not hold.

    The corpus's manifest then records the benchmark's files, as ``corpus
    build`` does for each benchmark that it checks.
    """
    folder = inputs / "benchmarks" / "topmed"
    folder.mkdir(parents=True)
    targets = [
        {
            "target_property": key,
            "target_type": target_type,
            "target_description": description,
            "target_values": "",
        }
        for key, target_type, description in (
            ("vital_signs.mean_arterial_pressure", "float", "Mean arterial pressure"),
            ("vital_signs.respiratory_rate", "integer", "Respiratory rate"),
            ("vital_signs.oxygen_saturation", "float", "Oxygen saturation"),
        )
    ]
    mappings = [
        {
            "source_study_id": "phs999999",
            "source_table_id": "pht999999.v1",
            "source_variable_id": "phv99999901.v1",
            "source_variable_name": "MAP01",
            "source_type": "float",
            "source_description": "Mean arterial pressure",
            "source_values": "",
            "expected_target": ["vital_signs.mean_arterial_pressure"],
        }
    ]
    manifest = benchmarks.write_benchmark(
        folder, "topmed", "1", targets, mappings, [], {}
    )
    corpus_path = inputs / "corpora" / CORPUS / "manifest.json"
    record = json.loads(corpus_path.read_text())
    for source in record["sources"]:
        if source["role"] == "benchmark" and source["name"] == "topmed":
            source["files"] = manifest["files"]
    corpus_path.write_text(json.dumps(record))


class TestFeatures:
    """Token ids made before training give the batches that live tokenization gives."""

    def test_ids_give_what_staticembedding_preprocess_gives(self, inputs):
        """features() on the ids equals StaticEmbedding.preprocess() on the texts."""
        corpus_tokenizer = trained_tokenizer(inputs)
        model = static.build(static.Settings(embedding_dim=8), corpus_tokenizer)
        anchors = [anchor for anchor, _ in TRAINING]
        ids = [
            encoding.ids
            for encoding in corpus_tokenizer.encode_batch(
                anchors, add_special_tokens=False
            )
        ]
        expected = model[0].preprocess(anchors)
        found = static.features(ids)
        assert found.keys() == expected.keys()
        for key in expected:
            assert torch.equal(found[key], expected[key])

    def test_the_collator_gives_the_same_batches(self, inputs):
        """The trainer's collator gives the same batch from cached ids as from the texts."""
        from sentence_transformers.sentence_transformer.data_collator import (
            SentenceTransformerDataCollator,
        )

        corpus_tokenizer = trained_tokenizer(inputs)
        model = static.build(static.Settings(embedding_dim=8), corpus_tokenizer)
        pairs = pd.DataFrame(TRAINING, columns=["anchor", "positive"])
        ids = train.token_ids(pairs, corpus_tokenizer)
        live = SentenceTransformerDataCollator(preprocess_fn=model.preprocess)
        cached = SentenceTransformerDataCollator(preprocess_fn=static.features)
        expected = live(pairs.to_dict("records"))
        found = cached([ids[number] for number in range(len(ids))])
        assert list(found) == list(expected)
        for key in expected:
            assert torch.equal(found[key], expected[key])

    def test_the_validation_pairs_become_the_evaluators_texts(self):
        """A text the tokenizer reads as another is one entry; an anchor's positives are all relevant."""
        full = "Body mass index: weight divided by height squared"
        pairs = pd.DataFrame(
            [("BMI", "Body mass index"), ("Quetelet index", "Body mass index"),
             ("BMI", full), ("bmi", "Body Mass Index")],
            columns=["anchor", "positive"],
        )  # fmt: skip
        queries, corpus_texts, relevant = train.validation_texts(pairs)
        full_key = "body mass index: weight divided by height squared"
        assert queries == {"bmi": "BMI", "quetelet index": "Quetelet index"}
        assert corpus_texts == {"body mass index": "Body mass index", full_key: full}
        assert relevant == {
            "bmi": {"body mass index", full_key},
            "quetelet index": {"body mass index"},
        }


class TestConfig:
    """The config's sections, the defaults under them and the flags over them."""

    def test_the_readme_example_reads(self):
        """The README's run_example.yaml is a config that train reads, on the example corpus and tokenizer."""
        path = Path(train.__file__).parent / "examples" / "run_example.yaml"
        values, _ = train.read_config(path, 0)
        assert (values["corpus"], values["tokenizer"]) == ("example", "example-8k")

    def test_flags_override_the_config_and_the_config_the_defaults(self, config):
        """A flag wins over the config, and the config over TRAINING_DEFAULTS."""
        values, given = train.read_config(
            config, 3, ["--learning_rate", "2e-2", "--embedding_dim", "32"]
        )
        assert values["learning_rate"] == 0.02
        assert values["num_train_epochs"] == 2
        assert values["warmup_steps"] == train.TRAINING_DEFAULTS["warmup_steps"]
        assert values["seed"] == 3
        assert values["embedding_dim"] == 32
        assert values["corpus"] == CORPUS
        assert {"learning_rate", "num_train_epochs"} <= given

    def test_the_defaults_are_the_recipes(self, tmp_path):
        """A config that names only its data trains at the recipe's settings, with this pipeline's own.

        The recipe gives the batch size, the warmup, the learning rate and the
        width. This pipeline gives 5 epochs and random batches. The loss's
        scale is the library's.
        """
        path = tmp_path / "abcd-run.yaml"
        path.write_text(yaml.safe_dump({"data": CONFIG["data"]}))
        values, _ = train.read_config(path, 0, ["--use_cpu"])
        settings = train.apply_config(values)
        training = settings["training"]
        assert training.per_device_train_batch_size == 2048
        assert training.learning_rate == 0.2
        assert training.warmup_steps == 0.1
        assert training.num_train_epochs == 5
        sampler = training.batch_sampler
        assert getattr(sampler, "value", sampler) == "batch_sampler"
        assert settings["static"].embedding_dim == 1024
        assert settings["loss"].scale == 20.0

    def test_a_number_that_yaml_reads_as_text_becomes_a_number(self, tmp_path):
        """YAML reads 1e-3 as text; the parser reads it by the setting's type."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(
            yaml.safe_dump({"data": CONFIG["data"]})
            + "training:\n  learning_rate: 1e-3\n  use_cpu: true\n"
        )
        values, _ = train.read_config(path, 0)
        assert values["learning_rate"] == 0.001

    @pytest.mark.parametrize(
        "text, message",
        [
            ("abcd_section:\n  efgh_setting: 1\n", "no section abcd_section"),
            ("model:\n  dimension: 8\n", "no setting dimension"),
            ("model: 8\n", "holds no settings"),
            (
                "model:\n  learning_rate: 0.1\n",
                "the model section has no setting learning_rate;"
                " learning_rate belongs in the training section",
            ),
            (
                "training:\n  embedding_dim: 8\n",
                "embedding_dim belongs in the model section, with architecture static",
            ),
            (
                "training:\n  hidden_size: 256\n",
                "hidden_size belongs in the model section, with architecture bert",
            ),
        ],
    )
    def test_an_unknown_section_or_setting_is_refused(self, tmp_path, text, message):
        """A section or a setting that does not exist names the config and itself."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            train.read_config(path, 0)

    @pytest.mark.parametrize(
        "training, overrides, message",
        [
            ({"prompts": {"anchor": "query: "}}, [], "prompts setting cannot be used"),
            ({}, ["--prompts", "query: "], "prompts setting cannot be used"),
            (
                {"router_mapping": {"anchor": "query"}},
                [],
                "router_mapping setting cannot be used",
            ),
            (
                {"load_best_model_at_end": True},
                [],
                "load_best_model_at_end setting cannot be used",
            ),
            (
                {},
                ["--resume_from_checkpoint", "abcd-run/trainer"],
                "resume_from_checkpoint setting cannot be used",
            ),
        ],
    )
    def test_a_setting_that_a_run_cannot_honour_is_refused(
        self, tmp_path, training, overrides, message
    ):
        """Each setting of REFUSED_TRAINING is refused, in the config or as a flag, before training."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(yaml.safe_dump({"data": CONFIG["data"], "training": training}))
        with pytest.raises(ValueError, match=message):
            train.read_config(path, 0, overrides)

    @pytest.mark.parametrize(
        "training, overrides",
        [({"save_strategy": "epoch"}, []), ({}, ["--save_strategy", "steps"])],
    )
    def test_a_save_strategy_other_than_no_is_refused(
        self, tmp_path, training, overrides
    ):
        """A save_strategy other than "no" is refused, in the config or as a flag."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(yaml.safe_dump({"data": CONFIG["data"], "training": training}))
        with pytest.raises(ValueError, match="save_strategy must be 'no'"):
            train.read_config(path, 0, overrides)

    def test_a_save_strategy_of_no_is_read(self, tmp_path):
        """save_strategy "no", which every run's config.yaml records, is read."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(
            yaml.safe_dump(
                {"data": CONFIG["data"], "training": {"save_strategy": "no"}}
            )
        )
        values, _ = train.read_config(path, 0)
        assert values["save_strategy"] == "no"

    def test_an_unknown_architecture_is_refused(self, tmp_path):
        """argparse checks choices only for a flag, so read_config checks a config's architecture."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(
            yaml.safe_dump(
                {"data": CONFIG["data"], "model": {"architecture": "abcd-architecture"}}
            )
        )
        with pytest.raises(ValueError, match="no architecture 'abcd-architecture'"):
            train.read_config(path, 0)

    def test_a_value_of_the_wrong_type_raises_an_error(self, tmp_path):
        """A value that does not fit its setting's type raises, rather than exiting the process."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(
            yaml.safe_dump({"data": CONFIG["data"], "model": {"embedding_dim": "abcd"}})
        )
        with pytest.raises(argparse.ArgumentError, match="embedding_dim"):
            train.read_config(path, 0)

    def test_a_bert_config_reads_its_settings_from_the_model_section(self, tmp_path):
        """A BERT run reads its settings from the model section, at the BERT paper's learning rate, 1e-4."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "data": CONFIG["data"],
                    "model": {"architecture": "bert", "num_hidden_layers": 1},
                }
            )
        )
        values, _ = train.read_config(path, 0, ["--use_cpu"])
        assert values["learning_rate"] == 1e-4
        settings = train.apply_config(values)
        assert "static" not in settings
        assert settings["bert"].num_hidden_layers == 1

    def test_a_bert_defaults_to_bert_smalls_shape(self, tmp_path):
        """A BERT without settings has BERT-Small's shape, 128 tokens and BertConfig's other defaults.

        BERT-Small has 4 layers of 512, with 8 heads, and a feed-forward part
        4 times the width. The length, 128 tokens, is the BERT paper's.
        """
        from transformers import BertConfig

        path = tmp_path / "abcd-run.yaml"
        path.write_text(
            yaml.safe_dump({"data": CONFIG["data"], "model": {"architecture": "bert"}})
        )
        values, _ = train.read_config(path, 0, ["--use_cpu"])
        settings = train.apply_config(values)["bert"]
        assert settings.num_hidden_layers == 4
        assert settings.hidden_size == 512
        assert settings.num_attention_heads == 8
        assert settings.intermediate_size == 4 * 512
        assert settings.max_position_embeddings == 128
        library = BertConfig()
        assert settings.hidden_dropout_prob == library.hidden_dropout_prob
        assert (
            settings.attention_probs_dropout_prob
            == library.attention_probs_dropout_prob
        )
        assert settings.initializer_range == library.initializer_range

    def test_a_given_learning_rate_is_kept(self, config):
        """The architecture's learning rate holds only when no config or flag gives one."""
        values, _ = train.read_config(config, 0, ["--learning_rate", "1e-3"])
        assert values["learning_rate"] == 1e-3

    @pytest.mark.parametrize(
        "sections, flags, message",
        [
            (
                {"model": {"num_hidden_layers": 1}},
                [],
                "num_hidden_layers is a setting of the bert architecture",
            ),
            (
                {},
                ["--num_hidden_layers", "1"],
                "--num_hidden_layers is a setting of the bert architecture",
            ),
            (
                {"model": {"architecture": "bert", "embedding_dim": 8}},
                [],
                "embedding_dim is a setting of the static architecture",
            ),
        ],
    )
    def test_another_architectures_settings_are_refused(
        self, tmp_path, sections, flags, message
    ):
        """A setting of another architecture, in the model section or as a flag, is refused."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(yaml.safe_dump({"data": CONFIG["data"], **sections}))
        with pytest.raises(ValueError, match=message):
            train.read_config(path, 0, flags)

    def test_each_setting_lives_in_one_section(self):
        """No two settings classes share a setting's name, since each setting has one flag."""
        names = [
            setting.name
            for classes in train.sections().values()
            for settings_class in classes.values()
            for setting in fields(settings_class)
        ]
        assert len(names) == len(set(names))

    def test_the_run_name_is_the_trainers_unless_the_config_names_one(
        self, config, tmp_path
    ):
        """The run's name becomes run_name, which W&B names the run after."""
        values, _ = train.read_config(config, 0, run_name="abcd-run_s0")
        assert values["run_name"] == "abcd-run_s0"
        values, _ = train.read_config(
            config, 0, ["--run_name", "efgh-run"], run_name="abcd-run_s0"
        )
        assert values["run_name"] == "efgh-run"

    def test_an_abbreviated_flag_is_refused(self, config):
        """--learning is not read as --learning_rate, which config.yaml would not record."""
        with pytest.raises(ValueError, match="--learning"):
            train.read_config(config, 0, ["--learning", "1e-2"])

    def test_bf16_is_off_unless_the_config_turns_it_on(self, config, tmp_path):
        """bf16 keeps the library's default, off; a config or a flag turns it on."""
        values, _ = train.read_config(config, 0)
        assert values["bf16"] is False
        values, given = train.read_config(config, 0, ["--bf16"])
        assert values["bf16"] is True
        assert "bf16" in given

    def test_reading_a_config_builds_no_settings(self, config, monkeypatch):
        """read_config builds no training settings, so it sets up no device."""
        from sentence_transformers import SentenceTransformerTrainingArguments

        monkeypatch.setattr(
            SentenceTransformerTrainingArguments,
            "__post_init__",
            lambda settings: pytest.fail("the training settings were built"),
        )
        values, _ = train.read_config(config, 0)
        assert values["corpus"] == CORPUS

    def test_the_settings_are_built_from_the_values(self, config):
        """Each settings class becomes settings, with the values read. Of the architectures, only the run's own, static, is built."""
        values, _ = train.read_config(config, 3, ["--use_cpu"])
        settings = train.apply_config(values)
        assert list(settings) == ["data", "model", "static", "loss", "training"]
        assert settings["data"].corpus == CORPUS
        assert settings["training"].seed == 3
        assert settings["training"].device.type == "cpu"


class TestFirstGpuOnly:
    """A run keeps its process to one GPU, before torch is imported."""

    def test_the_first_listed_gpu_is_kept(self, monkeypatch, capsys):
        """With CUDA_VISIBLE_DEVICES=2,3, the run keeps GPU 2, and names both."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert train.first_gpu_only() == ["2", "3"]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        assert "trains on GPU 2 only" in capsys.readouterr().out

    def test_without_a_list_gpu_0_is_kept(self, monkeypatch):
        """Without CUDA_VISIBLE_DEVICES, the run keeps GPU 0."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        assert train.first_gpu_only() == []
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    @pytest.mark.parametrize("listed", ["3", ""])
    def test_one_gpu_or_none_changes_nothing(self, monkeypatch, listed):
        """A list of one GPU, or an empty one, stays as it is."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", listed)
        assert train.first_gpu_only() == []
        assert os.environ["CUDA_VISIBLE_DEVICES"] == listed

    def test_the_gpu_is_kept_before_the_config_is_read(
        self, monkeypatch, config, tmp_path
    ):
        """train keeps to one GPU before it reads its config, which first imports torch."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        seen = []

        def read_config(*args, **kwargs):
            """Record the GPUs listed when the config is read, then stop the run."""
            seen.append(os.environ["CUDA_VISIBLE_DEVICES"])
            raise ValueError("stop here")

        monkeypatch.setattr(train, "read_config", read_config)
        assert (
            train.train(config, 0, [], tmp_path / "inputs", tmp_path / "outputs") == 1
        )
        assert seen == ["2"]


class TestTrain:
    """A run: the folder it writes and the inputs that stop it."""

    def test_a_run_writes_its_folder(self, inputs, config, tmp_path):
        """The model at the last step, its config, its record and the trainer's log."""
        outputs = tmp_path / "outputs"
        assert train.train(config, 0, [], inputs, outputs) == 0
        run = outputs / train.RUNS / "abcd-run_s0"
        assert (run / "model" / "model.safetensors").exists()
        assert not (run / "trainer").exists()
        record = json.loads((run / "manifest.json").read_text())
        assert record["run"] == "abcd-run_s0"
        assert record["counts"]["steps"] == 6
        assert record["device"] == "cpu"
        assert set(record["validation"]) == {"loss", "mrr@10", "accuracy@1",
                                             "accuracy@3", "accuracy@5",
                                             "accuracy@10"}  # fmt: skip
        corpus_source = record["sources"][0]
        assert corpus_source["name"] == CORPUS
        # write_corpus records one benchmark that the corpus was checked
        # against, topmed at version 1, with no files. The run copies it.
        assert corpus_source["benchmarks"] == [
            {"name": "topmed", "version": "1", "files": {}}
        ]
        assert record["files"]["model/model.safetensors"]["sha256"] == file_sha256(
            run / "model" / "model.safetensors"
        )
        recorded = yaml.safe_load((run / "config.yaml").read_text())
        assert list(recorded) == ["data", "model", "loss", "training"]
        assert recorded["model"] == {"architecture": "static", "embedding_dim": 16}
        assert recorded["training"]["seed"] == 0
        assert "output_dir" not in recorded["training"]
        logged = json.loads((run / "trainer_state.json").read_text())["log_history"]
        assert [
            log_entry["step"] for log_entry in logged if "loss" in log_entry
        ] == list(range(1, 7))
        assert any("eval_validation_cosine_mrr@10" in log_entry for log_entry in logged)

    def test_the_recorded_config_trains_the_same_model(self, inputs, config, tmp_path):
        """The run's own config.yaml, with the same seed, gives the same model."""
        first = tmp_path / "first"
        assert train.train(config, 0, [], inputs, first) == 0
        recorded = first / train.RUNS / "abcd-run_s0" / "config.yaml"
        again = tmp_path / "again" / "abcd-run.yaml"
        again.parent.mkdir()
        again.write_text(recorded.read_text())
        assert train.train(again, 0, [], inputs, tmp_path / "second") == 0
        weights = "runs/abcd-run_s0/model/model.safetensors"
        assert file_sha256(first / weights) == file_sha256(
            tmp_path / "second" / weights
        )

    def test_a_flag_overrides_the_config_and_is_recorded(
        self, inputs, config, tmp_path
    ):
        """--num_train_epochs 1 trains one epoch, and config.yaml says so."""
        outputs = tmp_path / "outputs"
        assert train.train(config, 0, ["--num_train_epochs", "1"], inputs, outputs) == 0
        run = outputs / train.RUNS / "abcd-run_s0"
        assert json.loads((run / "manifest.json").read_text())["counts"]["steps"] == 3
        assert (
            yaml.safe_load((run / "config.yaml").read_text())["training"][
                "num_train_epochs"
            ]
            == 1
        )

    def test_a_run_with_other_settings_keeps_the_earlier_run(
        self, inputs, config, tmp_path, monkeypatch, capsys
    ):
        """A flag keeps the run's name, so --learning_rate 0.05 is refused before it trains.

        The earlier run trained at the static architecture's rate, 0.2.
        """
        outputs = tmp_path / "outputs"
        assert train.train(config, 0, [], inputs, outputs) == 0
        manifest = outputs / train.RUNS / "abcd-run_s0" / "manifest.json"
        earlier_manifest = manifest.read_text()
        monkeypatch.setattr(train, "fit", lambda *args: pytest.fail("the run trained"))
        assert train.train(config, 0, ["--learning_rate", "0.05"], inputs, outputs) == 1
        error = capsys.readouterr().err
        assert "run abcd-run_s0 exists, trained with other settings" in error
        assert "learning_rate 0.2, here 0.05" in error
        assert manifest.read_text() == earlier_manifest

    def test_a_run_with_the_same_settings_replaces_the_earlier_run(
        self, inputs, tmp_path, monkeypatch
    ):
        """A run trained again with the same settings trains, and replaces the earlier one.

        config.yaml records report_to "none" as the list that the trainer
        makes of it, [], and the two still compare equal.
        """
        path = tmp_path / "abcd-run.yaml"
        config = {**CONFIG, "training": {**CONFIG["training"], "report_to": "none"}}
        path.write_text(yaml.safe_dump(config))
        outputs = tmp_path / "outputs"
        assert train.train(path, 0, [], inputs, outputs) == 0
        recorded = outputs / train.RUNS / "abcd-run_s0" / "config.yaml"
        assert yaml.safe_load(recorded.read_text())["training"]["report_to"] == []
        fit_calls = []
        fit = train.fit
        monkeypatch.setattr(
            train, "fit", lambda *args: fit_calls.append(True) or fit(*args)
        )
        assert train.train(path, 0, [], inputs, outputs) == 0
        assert fit_calls == [True]

    def test_an_earlier_run_whose_config_cannot_be_read_is_kept(
        self, inputs, config, tmp_path, capsys
    ):
        """An earlier run of the same name whose config.yaml is broken is not replaced."""
        outputs = tmp_path / "outputs"
        assert train.train(config, 0, [], inputs, outputs) == 0
        (outputs / train.RUNS / "abcd-run_s0" / "config.yaml").write_text("abcd: [")
        assert train.train(config, 0, [], inputs, outputs) == 1
        assert "its config.yaml cannot be read" in capsys.readouterr().err

    def test_the_last_evaluation_is_not_repeated(
        self, inputs, config, tmp_path, monkeypatch
    ):
        """When the trainer evaluated the last step, the run uses it, and evaluates no more."""
        from sentence_transformers import SentenceTransformerTrainer

        calls = []
        evaluate = SentenceTransformerTrainer.evaluate

        def counted(trainer, *args, **kwargs):
            """The trainer's evaluate, counted."""
            calls.append(trainer.state.global_step)
            return evaluate(trainer, *args, **kwargs)

        monkeypatch.setattr(SentenceTransformerTrainer, "evaluate", counted)
        assert train.train(config, 0, [], inputs, tmp_path / "every-2") == 0
        assert calls == [2, 4, 6]
        calls.clear()
        overrides = ["--eval_steps", "4"]
        assert train.train(config, 0, overrides, inputs, tmp_path / "every-4") == 0
        assert calls == [4, 6]

    def test_the_commit_is_read_before_training(
        self, inputs, config, tmp_path, monkeypatch
    ):
        """The run records the commit of the code that it started with."""
        from sentence_transformers import SentenceTransformerTrainer

        train_calls = []
        fit = SentenceTransformerTrainer.train

        def marked(trainer, *args, **kwargs):
            """The trainer's train, marked as done."""
            train_output = fit(trainer, *args, **kwargs)
            train_calls.append(True)
            return train_output

        monkeypatch.setattr(SentenceTransformerTrainer, "train", marked)
        monkeypatch.setattr(
            train,
            "code_commit",
            lambda: {
                "commit": "abcd456" if train_calls else "abcd123",
                "uncommitted_changes": False,
            },
        )
        outputs = tmp_path / "outputs"
        assert train.train(config, 0, [], inputs, outputs) == 0
        record = json.loads(
            (outputs / train.RUNS / "abcd-run_s0" / "manifest.json").read_text()
        )
        assert record["ai_harmonization"]["commit"] == "abcd123"

    def test_the_run_records_its_corpus_text_format(self, inputs, config, tmp_path):
        """The run copies the text format from the corpus's manifest."""
        write_corpus(inputs, corpus_rows(), text_format="name_description")
        outputs = tmp_path / "outputs"
        assert train.train(config, 0, [], inputs, outputs) == 0
        record = json.loads(
            (outputs / train.RUNS / "abcd-run_s0" / "manifest.json").read_text()
        )
        assert record["settings"]["format"] == "name_description"

    def test_a_corpus_without_a_text_format_is_refused(
        self, inputs, config, tmp_path, capsys
    ):
        """A corpus whose manifest names no text format asks for corpus build again."""
        write_corpus(inputs, corpus_rows(), text_format=None)
        assert train.train(config, 0, [], inputs, tmp_path / "outputs") == 1
        error = capsys.readouterr().err
        assert "holds no text format" in error
        assert f"corpus build --name {CORPUS}" in error

    def test_a_missing_tokenizer_names_the_command(self, inputs, tmp_path, capsys):
        """A tokenizer that was not trained stops the run, and the message says how to train it."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(
            yaml.safe_dump(
                {**CONFIG, "data": {"corpus": CORPUS, "tokenizer": "efgh-tokenizer"}}
            )
        )
        assert train.train(path, 0, [], inputs, tmp_path / "outputs") == 1
        assert "tokenizer train --name efgh-tokenizer" in capsys.readouterr().err

    def test_a_tokenizer_from_another_corpus_or_build_is_refused(self):
        """A tokenizer is refused when it learned from another corpus, or from another build of this one.

        Each message gives the command that trains one on this build, at the
        tokenizer's own size, 8,000 tokens. The build is told by the
        checksum of the pairs that the tokenizer learned from.
        """
        learned_files = {"pairs.parquet": {"sha256": "abcd1234"}}
        tokenizer_manifest = {
            "sources": [
                {"role": "corpus", "name": "efgh-corpus", "files": learned_files}
            ],
            "settings": {"vocab_size": 8000},
        }
        corpus_manifest = {"files": learned_files}
        with pytest.raises(
            ValueError, match="learned its vocabulary from efgh-corpus"
        ) as refused:
            train.check_tokenizer_corpus(
                "abcd-corpus", "efgh-corpus-8k", tokenizer_manifest, corpus_manifest
            )
        assert "tokenizer train --corpus abcd-corpus --vocab-size 8000" in str(
            refused.value
        )
        rebuilt_manifest = {"files": {"pairs.parquet": {"sha256": "efgh5678"}}}
        with pytest.raises(
            ValueError, match="an earlier build of corpus efgh-corpus"
        ) as refused:
            train.check_tokenizer_corpus(
                "efgh-corpus", "efgh-corpus-8k", tokenizer_manifest, rebuilt_manifest
            )
        assert "--vocab-size 8000 --name efgh-corpus-8k" in str(refused.value)
        train.check_tokenizer_corpus(
            "efgh-corpus", "efgh-corpus-8k", tokenizer_manifest, corpus_manifest
        )

    @pytest.mark.parametrize("seed", [-1, 2**32])
    def test_a_seed_out_of_range_is_refused(
        self, inputs, config, tmp_path, capsys, seed
    ):
        """A seed below 0, or past numpy's 32 bits, is refused before anything is read."""
        assert train.train(config, seed, [], inputs, tmp_path / "outputs") == 1
        assert "--seed must be from 0 to 4,294,967,295" in capsys.readouterr().err

    def test_a_config_without_a_corpus_is_refused(self, inputs, tmp_path, capsys):
        """The data section must name the corpus and the tokenizer."""
        path = tmp_path / "abcd-run.yaml"
        path.write_text(yaml.safe_dump({"training": CONFIG["training"]}))
        assert train.train(path, 0, [], inputs, tmp_path / "outputs") == 1
        assert "must name the corpus" in capsys.readouterr().err

    def test_a_corpus_without_validation_pairs_is_refused(
        self, inputs, config, tmp_path, capsys
    ):
        """A run evaluates on validation pairs, so a corpus needs them."""
        write_corpus(
            inputs, [pair(*pair_texts, corpus.TRAIN) for pair_texts in TRAINING]
        )
        assert train.train(config, 0, [], inputs, tmp_path / "outputs") == 1
        assert "no validation pairs" in capsys.readouterr().err

    def test_the_command(self, inputs, config, tmp_path):
        """python -m embedding_model train passes its other flags on as settings."""
        outputs = tmp_path / "outputs"
        assert main(["train", "--config", str(config), "--seed", "1",
                     "--inputs", str(inputs), "--outputs", str(outputs),
                     "--num_train_epochs", "1"]) == 0  # fmt: skip
        assert (outputs / train.RUNS / "abcd-run_s1" / "model").is_dir()

    def test_another_step_refuses_a_flag_that_it_does_not_have(self, inputs):
        """Only train takes flags beyond its own."""
        with pytest.raises(SystemExit):
            main(["tokenizer", "train", "--corpus", CORPUS, "--learning_rate", "1"])


class TestScoreRun:
    """score on a run's folder: its training format, its leakage check, its scores folder."""

    @pytest.fixture
    def run(self, inputs, config, tmp_path):
        """A trained run, and a benchmark named topmed at version 1."""
        write_benchmark(inputs)
        outputs = tmp_path / "outputs"
        assert train.train(config, 0, [], inputs, outputs) == 0
        return outputs / train.RUNS / "abcd-run_s0", outputs

    def test_a_checked_benchmark_is_scored_into_the_run(self, run, inputs):
        """A run whose corpus was checked against the benchmark is scored into its folder."""
        folder, outputs = run
        rows = scoring.score(str(folder), ["topmed"], inputs, outputs, device="cpu")
        assert rows[0]["format"] == "compact"
        assert (folder / "scores" / "topmed" / "compact" / "metrics.csv").exists()

    def test_the_run_is_found_from_its_model_folder(self, run, inputs):
        """The run's model folder also finds the run's record."""
        folder, outputs = run
        scoring.score(str(folder / "model"), ["topmed"], inputs, outputs, device="cpu")
        assert (folder / "scores" / "topmed" / "compact" / "metrics.csv").exists()

    def test_the_run_is_found_from_dot_inside_its_model_folder(
        self, run, inputs, monkeypatch
    ):
        """--model . inside the run's model folder finds the run's record too."""
        folder, outputs = run
        monkeypatch.chdir(folder / "model")
        scoring.score(".", ["topmed"], inputs, outputs, device="cpu")
        assert (folder / "scores" / "topmed" / "compact" / "metrics.csv").exists()

    def test_a_record_that_cannot_be_read_stops_scoring(self, run, inputs):
        """A broken record stops scoring, rather than skip the run's leakage check."""
        folder, outputs = run
        (folder / "manifest.json").write_text("{")
        with pytest.raises(ValueError, match="cannot be read"):
            scoring.score(str(folder), ["topmed"], inputs, outputs, device="cpu")

    @pytest.mark.parametrize(
        "field", ["not an object", "run", "files", "sources", "settings"]
    )
    def test_a_record_without_a_field_that_scoring_reads_stops_scoring(
        self, run, inputs, field
    ):
        """A record that is not an object, or lacks a field, stops scoring, rather than skip the leakage check."""
        folder, outputs = run
        path = folder / "manifest.json"
        if field == "not an object":
            path.write_text("[]")
        else:
            record = json.loads(path.read_text())
            del record[field]
            path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="is not a run's record"):
            scoring.score(str(folder), ["topmed"], inputs, outputs, device="cpu")

    def test_a_record_with_an_unknown_text_format_stops_scoring(self, run, inputs):
        """A run's text format must be one of FORMATS, rather than fail on lookup."""
        folder, outputs = run
        record = json.loads((folder / "manifest.json").read_text())
        record["settings"]["format"] = "abcd-format"
        (folder / "manifest.json").write_text(json.dumps(record))
        with pytest.raises(ValueError, match="which is not one of compact"):
            scoring.score(str(folder), ["topmed"], inputs, outputs, device="cpu")

    def test_a_model_that_differs_from_its_record_is_refused(self, run, inputs):
        """A run's model must be the one that its record names."""
        folder, outputs = run
        with (folder / "model" / "model.safetensors").open("ab") as weights:
            weights.write(b"abcd")
        with pytest.raises(ValueError, match="differs from the run's record"):
            scoring.score(str(folder), ["topmed"], inputs, outputs, device="cpu")

    def test_an_unchecked_benchmark_is_refused(self, run, inputs):
        """A benchmark that the corpus was not checked against stops scoring."""
        folder, outputs = run
        record = json.loads((folder / "manifest.json").read_text())
        record["sources"][0]["benchmarks"] = []
        (folder / "manifest.json").write_text(json.dumps(record))
        with pytest.raises(
            ValueError, match="was not checked against benchmark topmed"
        ):
            scoring.score(str(folder), ["topmed"], inputs, outputs, device="cpu")

    def test_a_new_version_with_the_same_files_is_scored(self, run, inputs):
        """The check compares the benchmark's files, so its version alone does not matter."""
        folder, outputs = run
        record = json.loads((folder / "manifest.json").read_text())
        record["sources"][0]["benchmarks"][0]["version"] = "0"
        (folder / "manifest.json").write_text(json.dumps(record))
        scoring.score(str(folder), ["topmed"], inputs, outputs, device="cpu")
        assert (folder / "scores" / "topmed" / "compact" / "metrics.csv").exists()

    def test_another_build_of_a_checked_version_is_refused(self, run, inputs):
        """A benchmark rebuilt with other files stops scoring, under the same version too."""
        folder, outputs = run
        built = inputs / "benchmarks" / "topmed"
        mappings = built / "topmed_mappings.csv"
        mappings.write_text(
            mappings.read_text().replace(
                "Mean arterial pressure", "Mean blood pressure"
            )
        )
        manifest = json.loads((built / "manifest.json").read_text())
        manifest["files"]["topmed_mappings.csv"]["sha256"] = file_sha256(mappings)
        (built / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(
            ValueError, match="this build of benchmark topmed"
        ) as raised:
            scoring.score(str(folder), ["topmed"], inputs, outputs, device="cpu")
        assert f"corpus build --name {CORPUS}" in str(raised.value)


# The smallest BERT that trains on a CPU in seconds: one layer of 32 dimensions.
BERT_SETTINGS = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
}
BERT_CONFIG = {**CONFIG, "model": {"architecture": "bert", **BERT_SETTINGS}}


class TestBert:
    """A transformer embedding model with BERT's architecture, trained from scratch on a CPU."""

    @pytest.fixture
    def bert_config(self, tmp_path):
        """The BERT run's config, as a YAML file named abcd-bert.yaml."""
        path = tmp_path / "abcd-bert.yaml"
        path.write_text(yaml.safe_dump(BERT_CONFIG))
        return path

    def test_a_bert_run_writes_its_folder(
        self, inputs, bert_config, tmp_path, monkeypatch
    ):
        """BERT trains on the texts: a BERT body with mean pooling, recorded as architecture bert.

        The run's config.yaml holds the BERT's settings in the model section,
        and reads back as the settings of the config that trained it.
        """
        from sentence_transformers import SentenceTransformer

        monkeypatch.setattr(
            train, "token_ids", lambda *args: pytest.fail("BERT trains on texts")
        )
        outputs = tmp_path / "outputs"
        assert train.train(bert_config, 0, [], inputs, outputs) == 0
        run = outputs / train.RUNS / "abcd-bert_s0"
        record = json.loads((run / "manifest.json").read_text())
        assert record["settings"]["architecture"] == "bert"
        recorded = yaml.safe_load((run / "config.yaml").read_text())
        assert list(recorded) == ["data", "model", "loss", "training"]
        assert recorded["model"]["architecture"] == "bert"
        assert "embedding_dim" not in recorded["model"]
        assert (
            train.read_config(run / "config.yaml", 0)[0]
            == train.read_config(bert_config, 0)[0]
        )
        assert recorded["training"]["learning_rate"] == 1e-4
        model = SentenceTransformer(str(run / "model"), device="cpu")
        assert [type(module).__name__ for module in model] == ["Transformer", "Pooling"]
        assert model.encode(["Body weight"]).shape == (1, 32)

    def test_a_bert_model_card_looks_up_nothing_on_the_hub(self, inputs):
        """The model card does not look up the temporary folder's path on the Hub."""
        settings = bert.Settings(**BERT_SETTINGS)
        model = bert.build(settings, trained_tokenizer(inputs))
        assert model.model_card_data.local_files_only

    def test_padding_leaves_a_texts_embedding_unchanged(self, inputs):
        """A text's embedding is the same alone and beside a longer text, which pads it.

        The attention mask keeps the padding out of the attention and out of
        the mean of the token vectors: the libraries' own work. Our part is
        the pad token that ``bert.build`` gives the model's tokenizer and
        config, which this checks.
        """
        settings = bert.Settings(**BERT_SETTINGS)
        model = bert.build(settings, trained_tokenizer(inputs))
        short = "Body weight"
        longer = (
            "Body weight in kilograms, measured at the visit after an overnight fast"
        )
        alone = model.encode([short], device="cpu")[0]
        padded = model.encode([short, longer], device="cpu")[0]
        assert padded.tolist() == pytest.approx(alone.tolist(), abs=1e-6)

    def test_a_run_sets_the_tokenizers_threads_only_when_unset(
        self, inputs, config, bert_config, tmp_path, monkeypatch
    ):
        """Unset, a BERT run turns the threads off and a static run on. A user's own setting is kept."""
        monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
        assert train.train(bert_config, 0, [], inputs, tmp_path / "bert") == 0
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        monkeypatch.delenv("TOKENIZERS_PARALLELISM")
        assert train.train(config, 0, [], inputs, tmp_path / "static") == 0
        assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
        assert train.train(bert_config, 0, [], inputs, tmp_path / "bert-again") == 0
        assert os.environ["TOKENIZERS_PARALLELISM"] == "true"

    def test_a_bert_run_trains_the_same_model_again(
        self, inputs, bert_config, tmp_path
    ):
        """Two BERT runs of one config and one seed, on a CPU, give the same weights."""
        weights = "runs/abcd-bert_s0/model/model.safetensors"
        for outputs in ("first", "second"):
            assert train.train(bert_config, 0, [], inputs, tmp_path / outputs) == 0
        assert file_sha256(tmp_path / "first" / weights) == file_sha256(
            tmp_path / "second" / weights
        )

    def test_a_bert_run_is_scored(self, inputs, bert_config, tmp_path):
        """score loads a BERT run with a plain SentenceTransformer, and ranks its benchmark."""
        write_benchmark(inputs)
        outputs = tmp_path / "outputs"
        assert train.train(bert_config, 0, [], inputs, outputs) == 0
        run = outputs / train.RUNS / "abcd-bert_s0"
        rows = scoring.score(str(run), ["topmed"], inputs, outputs, device="cpu")
        assert rows[0]["n"] == 1
        assert (run / "scores" / "topmed" / "compact" / "metrics.csv").exists()
