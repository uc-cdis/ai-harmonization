"""Tests for embedding_model.experiments: an experiment's file, its runs and its status.

Training a run is ``train``'s job, which its own tests check. Here each run's
process is replaced by a fake that writes the run's folder as ``train`` would,
so the tests check what an experiment adds: which runs it trains, where, with
which config, and on which GPU.
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import get_args

import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from embedding_model import experiments
from embedding_model.__main__ import main
from embedding_model.manifests import file_sha256
from embedding_model.train import validation_metrics

CORPUS = "abcd-corpus"
TOKENIZER = "abcd-tokenizer"
# The base of a first experiment: a small static model, on the CPU.
BASE = {
    "data": {"corpus": CORPUS, "tokenizer": TOKENIZER},
    "model": {"embedding_dim": 16},
    "training": {"num_train_epochs": 2, "use_cpu": True},
}
# Two arms that change the table's size, one smaller and one larger.
ARMS = {
    "dim8": {"test": "non-inferiority", "model": {"embedding_dim": 8}},
    "dim32": {"test": "superiority", "model": {"embedding_dim": 32}},
}


@pytest.fixture
def inputs(tmp_path):
    """An inputs folder with a built corpus's manifest and a trained tokenizer.

    ``experiment train`` checks that both are built before any run trains.
    It reads the corpus's manifest only, so the corpus has no pairs here.
    """
    folder = tmp_path / "inputs"
    corpus_folder = folder / "corpora" / CORPUS
    corpus_folder.mkdir(parents=True)
    (corpus_folder / "manifest.json").write_text(json.dumps({"corpus": CORPUS}))
    tokenizer_folder = folder / "tokenizers" / TOKENIZER
    tokenizer_folder.mkdir(parents=True)
    path = tokenizer_folder / "tokenizer.json"
    Tokenizer(WordLevel({"[UNK]": 0, "blood": 1}, unk_token="[UNK]")).save(str(path))
    (tokenizer_folder / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer": TOKENIZER,
                "files": {"tokenizer.json": {"sha256": file_sha256(path)}},
            }
        )
    )
    return folder


def experiment_file(folder, name="abcd-experiment", **fields):
    """Write an experiment's file, the base and two arms at seeds 0 and 1 unless fields say otherwise."""
    path = folder / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump({"base": BASE, "seeds": [0, 1], "arms": ARMS, **fields})
    )
    return path


class FakeTrain:
    """Stands in for ``subprocess.run``: each call writes its run's folder, as ``train`` would.

    It records each command and the GPU that the run saw. A run named in
    ``failing`` writes nothing and returns 1, as a failed ``train`` does.
    """

    def __init__(self, failing=()):
        """Start with no call, and fail the runs named in ``failing``."""
        self.calls = []
        self.failing = set(failing)

    def __call__(self, command, env, stdout, stderr, check):
        """Record the call, and write the run's folder unless the run fails.

        ``env`` is None when the run keeps this process's environment, as
        without a GPU; the GPU is then recorded as None.
        """
        flags = dict(zip(command[4::2], command[5::2]))
        run_name = f"{Path(flags['--config']).stem}_s{flags['--seed']}"
        gpu = None if env is None else env["CUDA_VISIBLE_DEVICES"]
        self.calls.append({"command": command, "gpu": gpu, "run": run_name})
        if run_name in self.failing:
            return subprocess.CompletedProcess(command, 1)
        run_folder = Path(flags["--outputs"]) / "runs" / run_name
        run_folder.mkdir(parents=True)
        (run_folder / "manifest.json").write_text(json.dumps({"run": run_name}))
        return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def fake_train(monkeypatch):
    """Replace each run's process with FakeTrain, and show the runs no GPU."""
    fake = FakeTrain()
    monkeypatch.setattr(experiments.subprocess, "run", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return fake


class TestReadExperiment:
    """An experiment's file, read and checked."""

    def test_the_readme_example_reads(self, tmp_path):
        """The README's experiment_example.yaml is an experiment whose base and arms give configs that train reads."""
        path = (
            Path(experiments.__file__).parent / "examples" / "experiment_example.yaml"
        )
        experiment = experiments.read_experiment(path)
        base_sections, _ = experiments.base_config(experiment["base"], tmp_path)
        arm_values = experiments.written_configs(
            experiments.arm_configs(experiment, base_sections), tmp_path, 0
        )
        assert sorted(arm_values) == ["base", "dim2048", "dim256"]
        assert [arm_values[arm]["embedding_dim"] for arm in ("dim256", "dim2048")] == [
            256,
            2048,
        ]

    def test_the_rule_takes_its_defaults(self, tmp_path):
        """A file without a decision or a question gets the default rule, and no question."""
        experiment = experiments.read_experiment(experiment_file(tmp_path))
        assert experiment["name"] == "abcd-experiment"
        assert experiment["question"] is None
        assert experiment["decision"] == {
            "scored_on": "validation",
            "metric": "mrr@10",
            "minimum_effect": 0.01,
            "alpha": 0.05,
        }

    def test_each_metric_that_a_run_records_can_decide(self):
        """The decision's metrics are those that a run's manifest records.

        train's validation_metrics names them: each key but the loss. The
        evaluator's results stand in as zeros, since only the names count.
        A metric that train adds, and the decision lacks, fails this test.
        """
        recorded = validation_metrics(defaultdict(float))
        metrics = get_args(experiments.Decision.model_fields["metric"].annotation)
        assert set(metrics) == set(recorded) - {"loss"}

    def test_the_file_keeps_its_own_rule(self, tmp_path):
        """A rule that the file gives replaces each default that it names."""
        path = experiment_file(
            tmp_path,
            question="Does a smaller table lose validation accuracy@1?",
            decision={"metric": "accuracy@1", "alpha": 0.01},
        )
        experiment = experiments.read_experiment(path)
        assert (
            experiment["question"] == "Does a smaller table lose validation accuracy@1?"
        )
        assert experiment["decision"]["metric"] == "accuracy@1"
        assert experiment["decision"]["alpha"] == 0.01
        assert experiment["decision"]["minimum_effect"] == 0.01

    @pytest.mark.parametrize(
        "fields, message",
        [
            (
                {"arm": ARMS},
                "the file has no field arm: its fields are question, base, seeds",
            ),
            (
                {"decision": {"score_on": "validation"}},
                "decision has no field score_on: its fields are scored_on, metric",
            ),
            (
                {"arms": {"dim8": {"model": {"embedding_dim": 8}}}},
                "arms.dim8.test is missing",
            ),
            (
                {"arms": {"dim8": {"test": "better"}}},
                "arms.dim8.test: Input should be 'superiority' or 'non-inferiority',"
                " not 'better'",
            ),
            ({"seeds": []}, "seeds: List should have at least 1 item"),
            (
                {"seeds": [0, -1]},
                "seeds.1: Input should be greater than or equal to 0, not -1",
            ),
            ({"seeds": [0, 0]}, "seeds: [0, 0] lists a seed twice"),
            ({"arms": {"base": ARMS["dim8"]}}, "'base' is not an arm's name"),
            ({"arms": {"dim 8": ARMS["dim8"]}}, "'dim 8' is not an arm's name"),
            (
                {
                    "arms": {
                        "efgh-corpus": {
                            "test": "superiority",
                            "data": {"corpus": "efgh-corpus"},
                        }
                    }
                },
                "arms.efgh-corpus: the arm changes data.corpus",
            ),
            (
                {"arms": {"seed": {"test": "superiority", "training": {"seed": 3}}}},
                "arms.seed: the arm sets training.seed",
            ),
            (
                {"base": {**BASE, "training": {"seed": 3}}},
                "base: the base sets training.seed",
            ),
            ({"base": "abcd-experiment"}, "<experiment>/<arm>, such as dim/dim128"),
            ({"base": ["data"]}, "base: it is neither the base's settings, by section"),
        ],
    )
    def test_a_field_that_cannot_be_used_is_refused(self, tmp_path, fields, message):
        """A file that breaks a rule of an experiment, or of its fields, is refused.

        The message names the file, the field, and what is wrong with it.
        """
        path = experiment_file(tmp_path, **fields)
        with pytest.raises(ValueError) as refused:
            experiments.read_experiment(path)
        assert str(refused.value).startswith(f"{path}: ")
        assert message in str(refused.value)

    def test_a_file_name_that_cannot_be_a_folder_name_is_refused(self, tmp_path):
        """A file whose name cannot be a folder's is refused, since the experiment's folder takes its name."""
        with pytest.raises(ValueError, match="not an experiment name"):
            experiments.read_experiment(experiment_file(tmp_path, name=".abcd"))

    def test_an_empty_section_is_read_as_empty(self, tmp_path):
        """A field with nothing in it, which YAML reads as None, is read as empty, or as its default.

        An empty section of settings stays empty, as train reads it. An empty
        question is None, and an empty decision rule is the default rule.
        """
        path = tmp_path / "abcd-experiment.yaml"
        path.write_text(
            "question:\nbase:\n  data: {corpus: abcd-corpus, tokenizer: abcd-tokenizer}\n"
            "  training:\nseeds: [0]\ndecision:\narms:\n  dim8:\n"
            "    test: superiority\n    model:\n"
        )
        experiment = experiments.read_experiment(path)
        assert experiment["base"]["training"] is None
        assert experiment["arms"]["dim8"]["model"] is None
        assert experiment["question"] is None
        assert experiment["decision"]["metric"] == "mrr@10"


class TestTrain:
    """Training an experiment's runs, with each run's process faked."""

    def test_each_arm_gets_its_full_config(self, tmp_path, inputs, fake_train):
        """Each arm's config is the base's, with only the arm's settings changed.

        The arm lr0.1 changes one of the base's three training settings, so
        its training section keeps the other two.
        """
        arms = {
            "dim8": ARMS["dim8"],
            "lr0.1": {"test": "superiority", "training": {"learning_rate": 0.1}},
        }
        path = experiment_file(tmp_path, arms=arms)
        assert experiments.train(path, inputs, tmp_path / "outputs") == 0
        configs = tmp_path / "outputs" / "experiments" / "abcd-experiment" / "configs"
        assert yaml.safe_load((configs / "base.yaml").read_text()) == BASE
        dim8_config = yaml.safe_load((configs / "dim8.yaml").read_text())
        assert dim8_config == {**BASE, "model": {"embedding_dim": 8}}
        lr_config = yaml.safe_load((configs / "lr0.1.yaml").read_text())
        assert lr_config["training"] == {
            "num_train_epochs": 2,
            "use_cpu": True,
            "learning_rate": 0.1,
        }

    def test_each_run_trains_on_its_own_gpu(
        self, tmp_path, inputs, fake_train, capsys, monkeypatch
    ):
        """With GPUs 2 and 3 visible, as SLURM shows a job, each run sees one of them alone."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert (
            experiments.train(experiment_file(tmp_path), inputs, tmp_path / "outputs")
            == 0
        )
        assert sorted(call["run"] for call in fake_train.calls) == [
            "base_s0",
            "base_s1",
            "dim32_s0",
            "dim32_s1",
            "dim8_s0",
            "dim8_s1",
        ]
        assert {call["gpu"] for call in fake_train.calls} == {"2", "3"}
        assert "2 GPUs: 2 runs at a time" in capsys.readouterr().out

    def test_without_a_gpu_the_runs_keep_the_environment(
        self, tmp_path, inputs, fake_train, capsys
    ):
        """With no CUDA GPU, the runs train one at a time, each with this process's environment, so train picks the device."""
        assert (
            experiments.train(experiment_file(tmp_path), inputs, tmp_path / "outputs")
            == 0
        )
        assert {call["gpu"] for call in fake_train.calls} == {None}
        assert "no CUDA GPU: one run at a time" in capsys.readouterr().out

    def test_each_run_is_the_public_train_command(self, tmp_path, inputs, fake_train):
        """A run is ``train``, with the arm's config and the seed, writing under the experiment's folder."""
        outputs = tmp_path / "outputs"
        experiments.train(experiment_file(tmp_path, seeds=[3]), inputs, outputs)
        folder = outputs / "experiments" / "abcd-experiment"
        command = next(
            call["command"] for call in fake_train.calls if call["run"] == "dim8_s3"
        )
        assert command == [
            sys.executable,
            "-m",
            "embedding_model",
            "train",
            "--config",
            str(folder / "configs" / "dim8.yaml"),
            "--seed",
            "3",
            "--inputs",
            str(inputs),
            "--outputs",
            str(folder),
        ]

    def test_training_again_retries_only_the_failed_runs(
        self, tmp_path, inputs, fake_train, capsys
    ):
        """A failed run does not stop the others. The next training of the same file trains only it.

        The base and two arms at two seeds make six runs.
        """
        path, outputs = experiment_file(tmp_path), tmp_path / "outputs"
        fake_train.failing = {"dim8_s1"}
        assert experiments.train(path, inputs, outputs) == 1
        assert "1 of 6 runs failed: dim8_s1" in capsys.readouterr().err
        fake_train.failing = set()
        fake_train.calls.clear()
        assert experiments.train(path, inputs, outputs) == 0
        assert [call["run"] for call in fake_train.calls] == ["dim8_s1"]

    def test_the_file_is_kept_as_it_was_first_trained(
        self, tmp_path, inputs, fake_train
    ):
        """The experiment keeps a copy of its file. A file of the same name whose seeds differ is refused."""
        path, outputs = experiment_file(tmp_path), tmp_path / "outputs"
        experiments.train(path, inputs, outputs)
        copy = outputs / "experiments" / "abcd-experiment" / "experiment.yaml"
        assert copy.read_bytes() == path.read_bytes()
        experiment_file(tmp_path, seeds=[0, 1, 2])
        with pytest.raises(ValueError, match="differs from") as refused:
            experiments.train(path, inputs, outputs)
        assert "in its seeds" in str(refused.value)
        assert "delete" in str(refused.value)
        assert "base: abcd-experiment/base" in str(refused.value)

    def test_a_file_that_reads_the_same_is_not_a_change(
        self, tmp_path, inputs, fake_train
    ):
        """A comment, another order of the fields, or a default written out leaves the experiment the same."""
        path, outputs = experiment_file(tmp_path), tmp_path / "outputs"
        assert experiments.train(path, inputs, outputs) == 0
        path.write_text(
            "# The table's size, smaller and larger.\n"
            + yaml.safe_dump(
                {
                    "arms": ARMS,
                    "decision": {"metric": "mrr@10"},
                    "seeds": [0, 1],
                    "base": BASE,
                }
            )
        )
        assert experiments.train(path, inputs, outputs) == 0

    def test_a_setting_that_train_refuses_stops_every_run(
        self, tmp_path, inputs, fake_train
    ):
        """An arm that names a setting that does not exist is refused by train's parser, before any run trains."""
        arms = {"dim8": {"test": "superiority", "model": {"embedding_dimension": 8}}}
        path, outputs = experiment_file(tmp_path, arms=arms), tmp_path / "outputs"
        with pytest.raises(ValueError, match="no setting embedding_dimension"):
            experiments.train(path, inputs, outputs)
        assert not fake_train.calls
        assert not (
            outputs / "experiments" / "abcd-experiment" / "experiment.yaml"
        ).exists()

    def test_a_corpus_that_is_not_built_names_its_command(
        self, tmp_path, inputs, fake_train
    ):
        """When the corpus is not built, the error gives the command that builds it."""
        (inputs / "corpora" / CORPUS / "manifest.json").unlink()
        with pytest.raises(FileNotFoundError, match="corpus build --name abcd-corpus"):
            experiments.train(experiment_file(tmp_path), inputs, tmp_path / "outputs")
        assert not fake_train.calls

    def test_an_arms_tokenizer_that_is_not_built_names_its_command(
        self, tmp_path, inputs, fake_train
    ):
        """An arm's own tokenizer is checked too. When it is not built, the error gives the command that trains it."""
        arms = {
            "efgh-tokenizer": {
                "test": "superiority",
                "data": {"tokenizer": "efgh-tokenizer"},
            }
        }
        with pytest.raises(
            FileNotFoundError, match="tokenizer train --name efgh-tokenizer"
        ):
            experiments.train(
                experiment_file(tmp_path, arms=arms), inputs, tmp_path / "outputs"
            )
        assert not fake_train.calls


class TestEarlierBase:
    """A base that is an arm of an earlier experiment, whose runs are reused."""

    def test_the_base_reuses_the_earlier_runs(self, tmp_path, inputs, fake_train):
        """The earlier arm's runs at seeds 0 and 1 are reused; the base at seed 2 trains here, with the arm's config."""
        outputs = tmp_path / "outputs"
        experiments.train(experiment_file(tmp_path), inputs, outputs)
        fake_train.calls.clear()
        arms = {"lr0.1": {"test": "superiority", "training": {"learning_rate": 0.1}}}
        later_path = experiment_file(
            tmp_path,
            name="abcd-later",
            base="abcd-experiment/dim8",
            seeds=[0, 1, 2],
            arms=arms,
        )
        assert experiments.train(later_path, inputs, outputs) == 0
        assert sorted(call["run"] for call in fake_train.calls) == [
            "base_s2",
            "lr0.1_s0",
            "lr0.1_s1",
            "lr0.1_s2",
        ]
        earlier_config = (
            outputs / "experiments" / "abcd-experiment" / "configs" / "dim8.yaml"
        )
        later_config = outputs / "experiments" / "abcd-later" / "configs" / "base.yaml"
        assert yaml.safe_load(later_config.read_text()) == yaml.safe_load(
            earlier_config.read_text()
        )

    def test_an_earlier_run_still_training_stops_the_experiment(
        self, tmp_path, inputs, fake_train
    ):
        """A base run of the earlier experiment that has a staging folder may be training now, so the later one waits."""
        outputs = tmp_path / "outputs"
        experiments.train(experiment_file(tmp_path, seeds=[0]), inputs, outputs)
        runs_folder = outputs / "experiments" / "abcd-experiment" / "runs"
        (runs_folder / ".dim8_s1~x1y2").mkdir()
        later_path = experiment_file(
            tmp_path, name="abcd-later", base="abcd-experiment/dim8", seeds=[0, 1]
        )
        with pytest.raises(ValueError, match="has started but not finished"):
            experiments.train(later_path, inputs, outputs)

    @pytest.mark.parametrize(
        "base, message",
        [
            ("efgh-experiment/base", "no experiment efgh-experiment"),
            (
                "abcd-experiment/dim64",
                "has no arm dim64; its arms are base, dim32, dim8",
            ),
        ],
    )
    def test_an_unknown_earlier_experiment_or_arm_is_refused(
        self, tmp_path, inputs, fake_train, base, message
    ):
        """A base that names no trained experiment, or no arm of it, is refused with what there is."""
        outputs = tmp_path / "outputs"
        experiments.train(experiment_file(tmp_path), inputs, outputs)
        later_path = experiment_file(tmp_path, name="abcd-later", base=base)
        with pytest.raises(ValueError, match=message):
            experiments.train(later_path, inputs, outputs)


class TestRunState:
    """A run's state, from its folder."""

    def test_a_run_is_done_started_or_not_started(self, tmp_path):
        """A manifest means done; a staging folder beside the run means started; nothing means not started."""
        done_run, started_run, not_started_run = (
            tmp_path / name for name in ("dim8_s0", "dim8_s1", "dim8_s2")
        )
        done_run.mkdir()
        (done_run / "manifest.json").write_text("{}")
        (tmp_path / ".dim8_s1~x1y2").mkdir()
        assert experiments.run_state(done_run) == experiments.DONE
        assert experiments.run_state(started_run) == experiments.STARTED
        assert experiments.run_state(not_started_run) == experiments.NOT_STARTED


class TestGpuIds:
    """The GPUs that the runs train on."""

    @pytest.mark.parametrize(
        "visible, ids", [("2,3", ["2", "3"]), ("0, 2", ["0", "2"]), ("", [])]
    )
    def test_cuda_visible_devices_decides(self, monkeypatch, visible, ids):
        """CUDA_VISIBLE_DEVICES gives the GPUs, as SLURM sets it for a job's GPUs; empty means none."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        assert experiments.gpu_ids() == ids

    def test_without_it_every_gpu_that_torch_finds(self, monkeypatch):
        """With CUDA_VISIBLE_DEVICES unset, the GPUs are those that torch counts."""
        import torch

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert experiments.gpu_ids() == ["0", "1"]


class TestStatus:
    """The status of an experiment's runs."""

    def test_each_arm_counts_its_runs_by_state(
        self, tmp_path, inputs, fake_train, capsys
    ):
        """After one failed run, the arm shows it as not started, and the others as done.

        The base and two arms at two seeds make six runs, and five are done.
        """
        path, outputs = experiment_file(tmp_path), tmp_path / "outputs"
        fake_train.failing = {"dim8_s1"}
        experiments.train(path, inputs, outputs)
        capsys.readouterr()
        assert experiments.status(path, outputs) == 0
        printed = capsys.readouterr().out
        assert "experiment abcd-experiment: 5 of 6 runs done" in printed
        assert "dim8        1 done, 1 not started" in printed


class TestCommandLine:
    """The experiment step on the command line."""

    def test_train_and_status(self, tmp_path, inputs, fake_train, capsys):
        """experiment train trains the runs, and experiment status counts them.

        The base and two arms at two seeds make six runs.
        """
        path, outputs = experiment_file(tmp_path), tmp_path / "outputs"
        flags = ["--config", str(path), "--outputs", str(outputs)]
        assert main(["experiment", "train", *flags, "--inputs", str(inputs)]) == 0
        assert main(["experiment", "status", *flags]) == 0
        assert "6 of 6 runs done" in capsys.readouterr().out

    def test_an_error_is_reported_and_exits_with_1(
        self, tmp_path, inputs, fake_train, capsys
    ):
        """A file that cannot be used is reported on stderr, and the command exits with 1."""
        path = experiment_file(tmp_path, seeds=[])
        flags = ["--config", str(path), "--outputs", str(tmp_path / "outputs")]
        assert main(["experiment", "train", *flags, "--inputs", str(inputs)]) == 1
        assert "experiment train: ValueError" in capsys.readouterr().err
