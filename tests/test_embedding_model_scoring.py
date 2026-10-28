"""Tests for the embedding model's scoring: texts, ranks, metrics and outputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace

from embedding_model import benchmarks, scoring, texts
from embedding_model.__main__ import main

WORDS = ["[UNK]", "age", "sex", "race", "gender"]


@pytest.fixture
def model():
    """A small static model that mistakes gender for race.

    age, sex and race are each their own direction. gender lies between race
    (0.8) and sex (0.6), so a gender variable sits closer to race than to sex.
    The mistake is on purpose: it lets the tests check a variable whose
    expected target is not ranked first. GENDER's expected target,
    demographic.sex, comes third.
    """
    from sentence_transformers import SentenceTransformer

    try:
        from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    except ImportError:
        from sentence_transformers.models import StaticEmbedding

    tokenizer = Tokenizer(
        WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]")
    )
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    weights = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.6, 0.8],
        ],
        dtype=np.float32,
    )
    return SentenceTransformer(
        modules=[StaticEmbedding(tokenizer, embedding_weights=weights)], device="cpu"
    )


def target(key, description, values=""):
    """A target row of a benchmark."""
    return {
        "target_property": key,
        "target_type": "string",
        "target_description": description,
        "target_values": values,
    }


def mapping(name, description, expected):
    """A row of a benchmark's mappings file, from a fictional study."""
    return {
        "source_study_id": "phs999999",
        "source_table_id": "pht999999.v1",
        "source_variable_id": f"phv9999990{len(name)}.v1",
        "source_variable_name": name,
        "source_type": "string",
        "source_description": description,
        "source_values": "",
        "expected_target": expected,
    }


TARGETS = [
    target("demographic.age", "Age"),
    target("demographic.sex", "Sex"),
    target("clinical.race", "Race"),
    target("demographic.race", "Race"),
]
MAPPINGS = [
    mapping("AGE01", "Age at exam", ["demographic.age"]),
    mapping("RACE", "Race", ["clinical.race", "demographic.race"]),
    mapping("GENDER", "Gender", ["demographic.sex"]),
]


def built(tmp_path, mappings, name="topmed"):
    """A benchmark built from TARGETS and the mappings given.

    Returns:
        Path: The inputs folder it was built into.
    """
    folder = tmp_path / "inputs" / "benchmarks" / name
    folder.mkdir(parents=True, exist_ok=True)
    benchmarks.write_benchmark(folder, name, "1", TARGETS, mappings, [], {})
    return tmp_path / "inputs"


@pytest.fixture
def inputs(tmp_path):
    """The benchmark built from TARGETS and MAPPINGS, in a temp inputs folder."""
    return built(tmp_path, MAPPINGS)


class TestRankTargets:
    def test_ranks_and_metrics(self, model, inputs):
        """A source variable's rank is the rank of its best placed expected target.

        The metrics come from those ranks.
        """
        targets, mappings, _ = benchmarks.read_benchmark(inputs, "topmed")
        source_texts, target_texts, expected = scoring.benchmark_texts(
            targets, mappings, texts.DEFAULT_FORMAT
        )
        metrics, ranked = scoring.rank_targets(
            model, source_texts, target_texts, expected
        )
        assert [ranked[q]["rank"] for q in ("0", "1", "2")] == [1, 1, 3]
        assert ranked["2"]["top1_target"] in {"clinical.race", "demographic.race"}
        assert metrics == {
            "mrr": round((1 + 1 + 1 / 3) / 3, 4),
            "accuracy@1": round(2 / 3, 4),
            "accuracy@5": 1.0,
            "accuracy@10": 1.0,
            "median_rank": 1.0,
        }

    def test_a_query_prompt_is_not_added(self, model, inputs):
        """Sources are embedded as the approaches embed them, without a query prompt."""
        model.prompts = {"query": "sex sex sex sex sex ", "document": ""}
        targets, mappings, _ = benchmarks.read_benchmark(inputs, "topmed")
        _, ranked = scoring.rank_targets(
            model, *scoring.benchmark_texts(targets, mappings, texts.DEFAULT_FORMAT)
        )
        assert [ranked[q]["rank"] for q in ("0", "1", "2")] == [1, 1, 3]

    def test_the_default_prompt_goes_on_both_sides(self, model, inputs):
        """A default prompt is added to sources and targets, as model.encode() adds it."""
        targets, mappings, _ = benchmarks.read_benchmark(inputs, "topmed")
        sources, target_texts, expected = scoring.benchmark_texts(
            targets, mappings, texts.DEFAULT_FORMAT
        )
        prompt = "sex sex "
        by_hand = scoring.rank_targets(
            model,
            {key: prompt + text for key, text in sources.items()},
            {key: prompt + text for key, text in target_texts.items()},
            expected,
        )
        model.prompts = {"abcd": prompt}
        model.default_prompt_name = "abcd"
        assert scoring.rank_targets(model, sources, target_texts, expected) == by_hand

    def test_targets_are_ranked_by_cosine(self, model, inputs):
        """Targets are ranked by cosine, even when the model's config names another similarity."""
        model.similarity_fn_name = "dot"
        targets, mappings, _ = benchmarks.read_benchmark(inputs, "topmed")
        metrics, ranked = scoring.rank_targets(
            model, *scoring.benchmark_texts(targets, mappings, texts.DEFAULT_FORMAT)
        )
        assert [ranked[q]["rank"] for q in ("0", "1", "2")] == [1, 1, 3]
        assert metrics["mrr"] == round((1 + 1 + 1 / 3) / 3, 4)


class TestScore:
    @pytest.fixture
    def saved(self, model, tmp_path):
        """The small static model, saved to a folder named abcd-model."""
        model.save(str(tmp_path / "abcd-model"))
        return str(tmp_path / "abcd-model")

    def test_writes_metrics_ranks_and_config(self, saved, inputs, tmp_path):
        """Scoring writes one metrics row, every source variable's rank, and the config."""
        (row,) = scoring.score(
            saved, ["topmed"], inputs, tmp_path / "outputs", device="cpu"
        )
        folder = tmp_path / "outputs" / "scores" / "abcd-model" / "topmed"
        folder = folder / texts.DEFAULT_FORMAT
        metrics = pd.read_csv(folder / "metrics.csv", dtype={"benchmark_version": str})
        assert metrics.loc[0, "mrr"] == row["mrr"]
        assert (metrics.loc[0, "n"], metrics.loc[0, "targets"]) == (3, 4)
        assert metrics.loc[0, "benchmark_version"] == "1"
        per_variable = pd.read_csv(folder / "per_variable.csv")
        assert list(per_variable["rank"]) == [1, 1, 3]
        assert list(per_variable["source_variable_name"]) == ["AGE01", "RACE", "GENDER"]
        config = yaml.safe_load((folder / "config.yaml").read_text())
        assert config["benchmark_manifest"]["version"] == "1"
        assert config["benchmark_manifest"]["files"]["topmed_mappings.csv"]["sha256"]
        assert config["format"] == texts.DEFAULT_FORMAT
        assert config["device"] == "cpu"
        assert set(config["ai_harmonization"]) == {"commit", "uncommitted_changes"}
        assert config["model_revision"] is None

    def test_another_model_of_the_same_name_is_a_warning(
        self, model, saved, inputs, capsys, tmp_path
    ):
        """Scores of another model folder with the same name are replaced, with a warning."""
        other = tmp_path / "other" / "abcd-model"
        model.save(str(other))
        for folder in (saved, str(other), str(other)):
            scoring.score(
                folder, ["topmed"], inputs, tmp_path / "outputs", device="cpu"
            )
        assert capsys.readouterr().err.count("They are replaced by those of") == 1

    def test_a_model_folder_is_loaded_without_asking_hugging_face(
        self, saved, inputs, monkeypatch, tmp_path
    ):
        """Loading a model folder sends no request to Hugging Face for its card."""
        from sentence_transformers.base import model_card

        asked = []
        monkeypatch.setattr(
            model_card, "get_model_info", lambda *args, **kwargs: asked.append(args)
        )
        scoring.score(saved, ["topmed"], inputs, tmp_path / "outputs", device="cpu")
        assert asked == []

    def test_a_relative_model_path_is_recorded_in_full(
        self, model, inputs, monkeypatch, capsys, tmp_path
    ):
        """Two folders given by the same relative path, from two places, are told apart."""
        for place in ("a", "b"):
            model.save(str(tmp_path / place / "abcd-model"))
            monkeypatch.chdir(tmp_path / place)
            scoring.score(
                "abcd-model", ["topmed"], inputs, tmp_path / "outputs", device="cpu"
            )
        folder = tmp_path / "outputs" / "scores" / "abcd-model" / "topmed"
        config = yaml.safe_load(
            (folder / texts.DEFAULT_FORMAT / "config.yaml").read_text()
        )
        assert config["model"] == str((tmp_path / "b" / "abcd-model").resolve())
        assert "They are replaced by those of" in capsys.readouterr().err

    @pytest.mark.parametrize("text", ["", "{", "[1, 2]"])
    def test_a_config_that_cannot_be_read_is_replaced_quietly(
        self, text, saved, inputs, capsys, tmp_path
    ):
        """A config.yaml that names no model, or is not YAML, is replaced with no warning."""
        folder = tmp_path / "outputs" / "scores" / "abcd-model" / "topmed"
        folder = folder / texts.DEFAULT_FORMAT
        folder.mkdir(parents=True)
        (folder / "config.yaml").write_text(text)
        scoring.score(saved, ["topmed"], inputs, tmp_path / "outputs", device="cpu")
        assert "warning" not in capsys.readouterr().err
        assert yaml.safe_load((folder / "config.yaml").read_text())["model"]

    def test_scores_each_benchmark(self, saved, inputs, tmp_path):
        """One call scores the model on each benchmark given, into its own folder."""
        built(tmp_path, MAPPINGS[:2], name="gdc")
        rows = scoring.score(
            saved, ["topmed", "gdc"], inputs, tmp_path / "outputs", device="cpu"
        )
        assert [(row["benchmark"], row["n"]) for row in rows] == [
            ("topmed", 3),
            ("gdc", 2),
        ]
        scores = tmp_path / "outputs" / "scores" / "abcd-model"
        for name in ("topmed", "gdc"):
            assert (scores / name / texts.DEFAULT_FORMAT / "metrics.csv").exists()

    def test_a_missing_benchmark_stops_before_any_scoring(
        self, saved, inputs, tmp_path
    ):
        """A benchmark not built yet stops scoring before any benchmark is scored."""
        with pytest.raises(FileNotFoundError, match="benchmarks build --gdc"):
            scoring.score(
                saved, ["topmed", "gdc"], inputs, tmp_path / "outputs", device="cpu"
            )
        assert not (tmp_path / "outputs").exists()


class TestScoresFolder:
    def test_a_hugging_face_model_keeps_its_owner(self, tmp_path):
        """A Hugging Face model's scores sit under its owner and name."""
        folder = scoring.scores_folder(
            tmp_path, "abcd-org/abcd-model", "gdc", "stdelement"
        )
        scores = tmp_path / "scores" / "abcd-org" / "abcd-model"
        assert folder == scores / "gdc" / "stdelement"

    def test_a_model_on_disk_is_named_after_its_folder(self, tmp_path):
        """A model on disk has its scores under its folder's name."""
        (tmp_path / "runs" / "abcd-run").mkdir(parents=True)
        folder = scoring.scores_folder(
            tmp_path, str(tmp_path / "runs" / "abcd-run"), "gdc", "stdelement"
        )
        assert folder == tmp_path / "scores" / "abcd-run" / "gdc" / "stdelement"

    def test_each_format_has_its_own_folder(self, tmp_path):
        """A model scored in two formats keeps both scores."""
        folders = {
            scoring.scores_folder(tmp_path, "abcd-org/abcd-model", "gdc", text_format)
            for text_format in ("stdelement", texts.DEFAULT_FORMAT)
        }
        assert len(folders) == 2


class TestWriteScores:
    """Writing one scoring's files as a whole folder."""

    def test_a_failed_write_keeps_the_previous_scores(self, tmp_path):
        """A scoring that fails while it writes leaves the previous files, all of them."""
        folder = tmp_path / "scores" / "abcd-model" / "topmed" / "compact"
        folder.mkdir(parents=True)
        (folder / "metrics.csv").write_text("previous")
        mappings = pd.DataFrame(
            {
                column: ["abcd"]
                for column in (
                    "source_study_id",
                    "source_table_id",
                    "source_variable_id",
                    "source_variable_name",
                    "expected_target",
                )
            }
        )

        to_csv = pd.DataFrame.to_csv

        def failing(frame, path, **keywords):
            """Write metrics.csv, then fail on per_variable.csv, as on a full disk."""
            if Path(path).name == "per_variable.csv":
                raise OSError("no space left on device")
            return to_csv(frame, path, **keywords)

        ranked = {"0": {"rank": 1, "top1_target": "a.b", "top1_similarity": 0.9}}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pd.DataFrame, "to_csv", failing)
            with pytest.raises(OSError):
                scoring.write_scores(folder, {"mrr": 1.0}, mappings, ranked, {})
        assert (folder / "metrics.csv").read_text() == "previous"
        assert [path.name for path in folder.parent.iterdir()] == ["compact"]


class TestModelRevision:
    def cached(self, folder, repo_id):
        """Put a fictional model in a Hugging Face cache, at commit abcd1234."""
        model = folder / ("models--" + repo_id.replace("/", "--"))
        (model / "snapshots" / "abcd1234").mkdir(parents=True)
        (model / "refs").mkdir()
        (model / "refs" / "main").write_text("abcd1234")

    def test_a_model_on_disk_has_none(self, tmp_path):
        """A model folder on disk has no Hugging Face commit."""
        assert scoring.model_revision(str(tmp_path)) is None

    def test_a_hugging_face_model_gives_its_cached_commit(self, monkeypatch, tmp_path):
        """A model loaded by id gives the commit that its cached copy was downloaded at."""
        self.cached(tmp_path, "abcd-org/abcd-model")
        monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path))
        assert scoring.model_revision("abcd-org/abcd-model") == "abcd1234"

    def test_a_model_not_in_the_cache_has_none(self, monkeypatch, tmp_path):
        """A model that the cache does not hold gives None, and scoring goes on."""
        monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path))
        assert scoring.model_revision("abcd-org/abcd-model") is None


class TestCommandLine:
    def test_score_command(self, model, inputs, tmp_path):
        """`score` scores a model folder on two built benchmarks."""
        built(tmp_path, MAPPINGS, name="gdc")
        model.save(str(tmp_path / "abcd-model"))
        status = main(
            [
                "score",
                "--model",
                str(tmp_path / "abcd-model"),
                "--benchmark",
                "topmed",
                "gdc",
                "--inputs",
                str(inputs),
                "--outputs",
                str(tmp_path / "outputs"),
                "--device",
                "cpu",
            ]
        )
        assert status == 0
        scores = tmp_path / "outputs" / "scores" / "abcd-model"
        for name in ("topmed", "gdc"):
            assert (scores / name / texts.DEFAULT_FORMAT / "metrics.csv").exists()

    def test_an_unbuilt_benchmark_is_a_clean_error(self, capsys, tmp_path):
        """An unbuilt benchmark exits 1 with the build command, not a traceback."""
        status = main(
            [
                "score",
                "--model",
                "abcd",
                "--benchmark",
                "gdc",
                "--inputs",
                str(tmp_path),
            ]
        )
        assert status == 1
        assert "benchmarks build --gdc" in capsys.readouterr().err

    def test_a_model_that_cannot_be_loaded_is_named(self, inputs, capsys, tmp_path):
        """A folder that holds no model is named, and score exits 1."""
        empty = tmp_path / "abcd-model"
        empty.mkdir()
        status = main(
            [
                "score",
                "--model",
                str(empty),
                "--benchmark",
                "topmed",
                "--inputs",
                str(inputs),
                "--outputs",
                str(tmp_path / "outputs"),
                "--device",
                "cpu",
            ]
        )
        assert status == 1
        assert f"the model {empty} cannot be loaded" in capsys.readouterr().err

    def test_score_needs_a_known_benchmark(self):
        """An unknown benchmark is a usage error."""
        with pytest.raises(SystemExit) as raised:
            main(["score", "--model", "abcd", "--benchmark", "abcd"])
        assert raised.value.code == 2
