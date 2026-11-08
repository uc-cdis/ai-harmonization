"""Tests for embedding_model.compare: intervals, paired tests, and the comparison's folder."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from matplotlib import rc_context
from matplotlib.container import BarContainer
from scipy import stats

from embedding_model import benchmarks, compare, scoring
from embedding_model.__main__ import main

# A fictional benchmark of six vital signs and measures, with one expected
# target each: (source variable, its description, its expected target).
VARIABLES = [
    ("SBP01", "Systolic blood pressure", "vital_signs.systolic_blood_pressure"),
    ("DBP01", "Diastolic blood pressure", "vital_signs.diastolic_blood_pressure"),
    ("HR01", "Heart rate", "vital_signs.heart_rate"),
    ("BMI01", "Body mass index", "anthropometry.body_mass_index"),
    ("WAIST01", "Waist circumference", "anthropometry.waist_circumference"),
    ("GLU01", "Fasting glucose", "lab.fasting_glucose"),
]
# Two fictional models, by the rank at which each one puts each variable's
# expected target. The reference ranks four first. The other ranks one first,
# and WAIST01's twelfth.
REFERENCE = "abcd-org/abcd-model"
OTHER = "efgh-org/efgh-model"
RANKS = {REFERENCE: [1, 1, 1, 2, 1, 3], OTHER: [2, 1, 4, 2, 12, 3]}
# Their sizes: the reference a static model of 8,000 tokens by 256 dimensions,
# the other a transformer of 110 million parameters.
PARAMETERS = {REFERENCE: 8_000 * 256, OTHER: 110_000_000}
# The target that a model puts first when it misses a variable's expected
# target: the closest other measure, such as diastolic for systolic blood
# pressure.
NEAR_MISSES = {
    "SBP01": "vital_signs.diastolic_blood_pressure",
    "DBP01": "vital_signs.systolic_blood_pressure",
    "HR01": "vital_signs.systolic_blood_pressure",
    "BMI01": "anthropometry.waist_circumference",
    "WAIST01": "anthropometry.body_mass_index",
    "GLU01": "anthropometry.body_mass_index",
}


def committed_version(name):
    """The version of a benchmark's committed manifest, which a test build takes."""
    committed = benchmarks.MANIFESTS / f"{name}.json"
    return json.loads(committed.read_text(encoding="utf-8"))["version"]


def mappings():
    """The benchmark's mapping rows, from a fictional study."""
    return [
        {
            "source_study_id": "phs999999",
            "source_table_id": "pht999999.v1",
            "source_variable_id": f"phv999999{number:02d}.v1",
            "source_variable_name": name,
            "source_type": "float",
            "source_description": description,
            "source_values": "",
            "expected_target": [target],
        }
        for number, (name, description, target) in enumerate(VARIABLES, start=1)
    ]


def targets():
    """The benchmark's targets: one per variable, described as the variable is."""
    return [
        {
            "target_property": target,
            "target_type": "float",
            "target_description": description,
            "target_values": "",
        }
        for _, description, target in VARIABLES
    ]


def built_manifest(inputs):
    """The manifest of the topmed benchmark built in ``inputs``."""
    return benchmarks.read_benchmark(inputs, "topmed")[2]


def score_means(ranks):
    """The means that score writes to metrics.csv, from each variable's rank.

    Each variable has one expected target, so its nDCG@10 is
    1 / log2(rank + 1) within the first 10, as MRR@10 is 1 / rank there.
    """
    ranks = np.asarray(ranks)
    within_10 = ranks <= 10
    means = {
        "mrr": np.mean(1 / ranks),
        "mrr@10": np.mean(np.where(within_10, 1 / ranks, 0)),
        "ndcg@10": np.mean(np.where(within_10, 1 / np.log2(ranks + 1), 0)),
        **{f"accuracy@{k}": np.mean(ranks <= k) for k in (1, 3, 5, 10)},
    }
    return {metric: round(float(mean), 4) for metric, mean in means.items()}


def scores(
    inputs,
    outputs,
    model,
    ranks,
    reverse=False,
    parameters=None,
    blank_ids=False,
    folder=None,
):
    """Write a model's scores on the topmed benchmark built in ``inputs``, as score writes them.

    ``model`` is a Hugging Face id, a model folder's full path or a run's
    folder, as score records them. config.yaml records the benchmark's version
    and files. A variable whose expected target is not first has its near
    miss, from NEAR_MISSES, as its first suggestion. ``reverse`` lists the
    variables in reverse order. ``parameters`` defaults to the model's in
    PARAMETERS. ``blank_ids`` leaves every variable's id empty, as GDC gives
    none. ``folder`` defaults to the model's scores folder under ``outputs``.
    """
    manifest = built_manifest(inputs)
    order = list(range(len(VARIABLES)))
    if reverse:
        order.reverse()
    table = (
        pd.DataFrame(mappings())
        .assign(expected_target=[target for _, _, target in VARIABLES])
        .iloc[order]
        .reset_index(drop=True)
    )
    if blank_ids:
        table["source_variable_id"] = ""
    ranked = {
        str(position): {
            "rank": ranks[index],
            "top1_target": (
                VARIABLES[index][2]
                if ranks[index] == 1
                else NEAR_MISSES[VARIABLES[index][0]]
            ),
            "top1_similarity": 0.9,
        }
        for position, index in enumerate(order)
    }
    row = {
        "model": model,
        "benchmark": "topmed",
        "benchmark_version": manifest["version"],
        "format": "compact",
        "n": len(ranks),
        "targets": len(VARIABLES),
        "parameters": PARAMETERS[model] if parameters is None else parameters,
        **score_means(ranks),
        "median_rank": float(np.median(ranks)),
    }
    if folder is None:
        folder = scoring.scores_folder(outputs, model, "topmed", "compact")
    config = {
        "model": model,
        "benchmark_manifest": {
            "version": manifest["version"],
            "files": manifest["files"],
        },
    }
    scoring.write_scores(folder, row, table, ranked, config)
    return folder


@pytest.fixture
def built(tmp_path):
    """The inputs and outputs folders: the benchmark built, and both models scored."""
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    folder = inputs / "benchmarks" / "topmed"
    folder.mkdir(parents=True)
    benchmarks.write_benchmark(
        folder, "topmed", committed_version("topmed"), targets(), mappings(), [], {}
    )
    for model, ranks in RANKS.items():
        scores(inputs, outputs, model, ranks)
    return inputs, outputs


def scored_run(inputs, outputs, run):
    """A run of train in ``run``, a static model, with its own scores on topmed.

    Its scores are in the run's folder, and name the run's folder as the
    model, as score writes a run's scores. It ranks as the reference does.
    """
    (run / "model").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"run": run.name, "files": {}, "sources": [], "settings": {}})
    )
    scores(
        inputs,
        outputs,
        str(run.resolve()),
        RANKS[REFERENCE],
        parameters=PARAMETERS[REFERENCE],
        folder=run / "scores" / "topmed" / "compact",
    )
    return run


class TestVariableValues:
    """Each source variable's value of each metric, from its rank."""

    def test_each_value_comes_from_the_rank(self):
        """MRR takes 1 / rank, MRR@10 counts a rank past 10 as 0, accuracy@k is 1 or 0.

        Rank 10 is the edge: it still counts for MRR@10 and accuracy@10.
        """
        values = compare.variable_values(np.array([1, 4, 10, 12]))
        assert values["mrr"].tolist() == [1, 1 / 4, 1 / 10, 1 / 12]
        assert values["mrr@10"].tolist() == [1, 1 / 4, 1 / 10, 0]
        assert values["accuracy@1"].tolist() == [1, 0, 0, 0]
        assert values["accuracy@3"].tolist() == [1, 0, 0, 0]
        assert values["accuracy@5"].tolist() == [1, 1, 0, 0]
        assert values["accuracy@10"].tolist() == [1, 1, 1, 0]


def paired_values(both, only_model, only_reference, neither):
    """Two models' values, 1 or 0, from the four counts of their 2 x 2 table."""
    model = [1] * both + [1] * only_model + [0] * only_reference + [0] * neither
    reference = [1] * both + [0] * only_model + [1] * only_reference + [0] * neither
    return np.array(model, dtype=float), np.array(reference, dtype=float)


class TestInterval:
    """The t interval for MRR, and Newcombe's for a difference of accuracy@k."""

    def test_the_t_interval_is_a_95_percent_one(self):
        """The interval is the 95% t interval around the mean: on 200 variables, 0.75 +/- 0.0349.

        Of the 200 variables, 100 rank their target first and 100 second: MRR
        values of 1 and 1/2, with a mean of 0.75. Their standard deviation is
        0.2506, and t at 97.5% with 199 degrees of freedom is 1.972. So each
        end lies 1.972 x 0.2506 / sqrt(200) = 0.0349 from the mean.
        """
        ranks = np.array([1] * 100 + [2] * 100)
        low, high = compare.t_interval(compare.variable_values(ranks)["mrr"])
        assert (low, high) == pytest.approx((0.75 - 0.0349, 0.75 + 0.0349), abs=1e-4)

    @pytest.mark.filterwarnings("error")
    def test_equal_values_give_their_value(self):
        """When every variable has the same MRR, such as 1, the interval is that value, with no warning.

        The values have no spread, so the interval has no width. scipy gives
        that interval too, but warns of a loss of precision. The code returns
        the shared value without asking scipy, so no warning is raised.
        """
        assert compare.t_interval(np.ones(6)) == (1.0, 1.0)

    def test_the_interval_and_the_test_agree(self):
        """A difference's interval excludes 0 exactly when the paired t-test's p is below 0.05.

        Both come from one t distribution. Of 20 variables, the model finds
        8 expected targets first that the reference finds second, and the two
        tie on the other 12, at ranks 1 to 12: a difference, with p 0.002.
        With the two models swapped on half of the 8, the means cancel: no
        difference, and p 1.
        """
        tied = list(range(1, 13))
        reference = 1 / np.array([2] * 8 + tied)
        better = 1 / np.array([1] * 8 + tied)
        mixed = 1 / np.array([1] * 4 + [2] * 4 + tied)
        mixed_reference = 1 / np.array([2] * 4 + [1] * 4 + tied)
        for model, base in ((better, reference), (mixed, mixed_reference)):
            low, high = compare.t_interval(model - base)
            _, p = compare.paired_test("mrr", model, base)
            assert (low > 0 or high < 0) == (p < 0.05)

    def test_newcombes_interval_uses_the_pairing(self):
        """Two models with the same shares get a narrower interval when they agree.

        On 20 variables, each model finds 10 expected targets. Models that
        agree on every variable get a narrower interval of their difference
        than models that agree only as often as chance gives: 5 found by
        both, 5 by each alone, and 5 by neither. Models that agree on every
        variable still keep an interval of some width, around 0.
        """
        agreeing = compare.newcombe_interval(*paired_values(10, 0, 0, 10))
        independent = compare.newcombe_interval(*paired_values(5, 5, 5, 5))
        assert agreeing[1] - agreeing[0] < independent[1] - independent[0]
        assert agreeing[0] < 0 < agreeing[1]

    def test_swapping_the_models_negates_the_interval(self):
        """The reference minus the model is the interval of the model minus the reference, negated."""
        model, reference = paired_values(10, 5, 2, 20)
        low, high = compare.newcombe_interval(model, reference)
        assert compare.newcombe_interval(reference, model) == pytest.approx(
            (-high, -low)
        )


class TestPairedTest:
    """The paired tests of two models on the same source variables."""

    def test_accuracy_is_compared_by_mcnemars_exact_test(self):
        """At k = 1, the models differ on SBP01, HR01 and WAIST01, all for the reference.

        Under no difference, each is a coin flip, so p = 2 x 0.5^3 = 0.25.
        """
        reference = compare.variable_values(np.array(RANKS[REFERENCE]))["accuracy@1"]
        other = compare.variable_values(np.array(RANKS[OTHER]))["accuracy@1"]
        test, p = compare.paired_test("accuracy@1", other, reference)
        assert (test, p) == (compare.MCNEMAR, 0.25)

    def test_models_that_never_differ_have_p_1(self):
        """Two models with the same accuracy on every variable have nothing to test."""
        values = np.array([1.0, 0.0, 1.0])
        assert compare.paired_test("accuracy@1", values, values) == (
            compare.MCNEMAR,
            1.0,
        )

    def test_mrr_is_compared_by_a_paired_t_test(self):
        """The paired t-test finds a difference that the spread of ranks hides from an unpaired test.

        Of 20 variables, the model finds 8 expected targets first that the
        reference finds second: a difference of 8 x (1 - 1/2) / 20 = 0.2 in
        MRR. The other 12 tie, at ranks 1 to 12. The paired test sees each
        variable's difference, and gives p 0.002. An unpaired test sees the
        spread of the 12 tied ranks too, and gives p 0.07.
        """
        tied = list(range(1, 13))
        model = 1 / np.array([1] * 8 + tied)
        reference = 1 / np.array([2] * 8 + tied)
        test, p = compare.paired_test("mrr", model, reference)
        assert test == compare.PAIRED_T
        assert p == pytest.approx(0.0021, abs=1e-4)
        assert stats.ttest_ind(model, reference).pvalue > 0.05

    @pytest.mark.filterwarnings("error")
    def test_differences_without_spread_have_p_0_or_1(self):
        """When every variable differs by the same amount, p is 0, or 1 when that amount is 0, with no warning.

        The reference ranks each of six targets second, and the model first:
        each variable gains 1 - 1/2. The t statistic is infinite, so p is 0.
        Two models that rank alike have nothing to test, so p is 1, where
        scipy gives NaN. scipy warns in both cases; the code asks it in
        neither.
        """
        model, reference = np.ones(6), np.full(6, 1 / 2)
        assert compare.paired_test("mrr", model, reference) == (compare.PAIRED_T, 0.0)
        assert compare.paired_test("mrr", model, model) == (compare.PAIRED_T, 1.0)

    def test_the_differences_keep_p_whole(self):
        """The reference finds eleven targets first, and the other model none.

        McNemar's exact p is 2 x 0.5^11 = 0.0009765625, below 0.001, so three
        stars. Rounded to four decimals, it would be 0.0010, with two.
        """
        values = {
            REFERENCE: compare.variable_values(np.ones(11, dtype=int)),
            OTHER: compare.variable_values(np.full(11, 2)),
        }
        rows = compare.difference_rows("topmed", REFERENCE, [OTHER], values)
        at_1 = next(row for row in rows if row["metric"] == "accuracy@1")
        assert at_1["p"] == 2 * 0.5**11
        assert compare.stars(at_1["p"]) == "***"

    def test_stars_mark_the_p_below_each_level(self):
        """One star below 0.05, two below 0.01, three below 0.001, none from 0.05 up.

        A p of exactly 0.05 gets no star, as the note under the legend says.
        """
        assert [compare.stars(p) for p in (0.2, 0.04, 0.009, 0.0009)] == [
            "",
            "*",
            "**",
            "***",
        ]
        assert [compare.stars(p) for p in (0.05, 0.01, 0.001)] == ["", "*", "**"]


class TestCompare:
    """A comparison of two scored models."""

    def test_metrics_differences_figures_and_manifest(self, built, capsys):
        """The folder holds both tables, the four figures in each format, and the manifest.

        The reference's MRR is (1 + 1 + 1 + 1/2 + 1 + 1/3) / 6, the other's
        (1/2 + 1 + 1/4 + 1/2 + 1/12 + 1/3) / 6. The other's nDCG@10 takes
        1 / log2(rank + 1) for each rank within 10. The reference ranks every
        expected target within 5, so its accuracy@5 is 1, and its Wilson
        interval starts at 6 / (6 + z^2).

        On accuracy@1, both models find DBP01's target first, only the
        reference finds SBP01's, HR01's and WAIST01's, and neither finds
        BMI01's or GLU01's. Newcombe lowers that table's covariance,
        1 x 2 - 0 x 3 = 2, by n / 2 = 3, to 0. So the interval of 1/6 - 4/6
        combines the two Wilson intervals as if apart: 0.0301 to 0.5635 for
        1/6, and 0.3000 to 0.9032 for 4/6. It runs from 1/6 - 4/6 minus
        sqrt((1/6 - 0.0301)^2 + (0.9032 - 4/6)^2) to 1/6 - 4/6 plus
        sqrt((0.5635 - 1/6)^2 + (4/6 - 0.3000)^2): -0.7732 to 0.0403.
        """
        inputs, outputs = built
        folder = compare.compare(
            "abcd-comparison",
            [REFERENCE, OTHER],
            ["topmed"],
            inputs,
            outputs,
            reference=REFERENCE,
        )
        assert folder == outputs / "comparisons" / "abcd-comparison"
        metrics = pd.read_csv(folder / "metrics.csv").set_index("model")
        assert metrics.loc[REFERENCE, "mrr"] == round(
            (1 + 1 + 1 + 1 / 2 + 1 + 1 / 3) / 6, 4
        )
        assert metrics.loc[OTHER, "mrr"] == round(
            (1 / 2 + 1 + 1 / 4 + 1 / 2 + 1 / 12 + 1 / 3) / 6, 4
        )
        for model in RANKS:
            row = metrics.loc[model]
            assert row["mrr_low"] <= row["mrr"] <= row["mrr_high"]
        log2 = np.log2
        assert metrics.loc[OTHER, "ndcg@10"] == round(
            (1 / log2(3) + 1 + 1 / log2(5) + 1 / log2(3) + 0 + 1 / log2(4)) / 6, 4
        )
        assert metrics.loc[REFERENCE, "parameters"] == 8_000 * 256
        differences = pd.read_csv(folder / "differences.csv")
        # One other model, on MRR, MRR@10 and accuracy@1, @3, @5 and @10.
        assert len(differences) == 6
        at_1 = differences.set_index("metric").loc["accuracy@1"]
        assert (at_1.test, at_1.p) == (compare.MCNEMAR, 0.25)
        assert at_1.difference == round(1 / 6 - 4 / 6, 4)
        assert (at_1.low, at_1.high) == (-0.7732, 0.0403)
        z = stats.norm.ppf(0.975)
        assert metrics.loc[REFERENCE, "accuracy@5_low"] == round(6 / (6 + z**2), 4)
        for stem in (
            "accuracy_and_mrr",
            "accuracy_by_k",
            "mrr_by_size",
            "differences_from_reference",
        ):
            for suffix in ("png", "pdf", "eps"):
                assert (folder / "figures" / f"{stem}.{suffix}").stat().st_size > 0
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["models"] == [REFERENCE, OTHER]
        assert manifest["reference"] == REFERENCE
        assert manifest["benchmarks"] == {"topmed": committed_version("topmed")}
        assert manifest["settings"]["format"] == "compact"
        assert manifest["settings"]["intervals"]["accuracy@1"] == compare.WILSON
        assert manifest["settings"]["difference_intervals"]["mrr"] == compare.T_INTERVAL
        assert manifest["settings"]["tests"]["mrr"] == compare.PAIRED_T
        assert manifest["built"] == date.today().isoformat()
        assert [source["name"] for source in manifest["sources"]] == [REFERENCE, OTHER]
        assert "-> " in capsys.readouterr().out

    def test_without_a_reference_no_differences_are_written(self, built, capsys):
        """A comparison with no reference writes the metrics and three figures, no differences, and says why."""
        inputs, outputs = built
        folder = compare.compare(
            "abcd-comparison", [REFERENCE, OTHER], ["topmed"], inputs, outputs
        )
        assert (folder / "metrics.csv").exists()
        assert not (folder / "differences.csv").exists()
        figures = sorted(path.name for path in (folder / "figures").glob("*.png"))
        assert figures == [
            "accuracy_and_mrr.png",
            "accuracy_by_k.png",
            "mrr_by_size.png",
        ]
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["reference"] is None
        assert "no --reference: no paired tests" in capsys.readouterr().out

    def test_a_run_is_read_from_its_folder_by_its_name(self, built):
        """A run of train is named after its run, and its scores are read from its own folder."""
        inputs, outputs = built
        run = scored_run(inputs, outputs, outputs / "runs" / "abcd-run_s0")
        assert compare.label_of(str(run)) == "abcd-run_s0"
        folder = compare.compare(
            "abcd-comparison", [str(run), OTHER], ["topmed"], inputs, outputs
        )
        assert pd.read_csv(folder / "metrics.csv").model.tolist() == [
            "abcd-run_s0",
            OTHER,
        ]

    def test_a_moved_run_is_still_read(self, built, tmp_path):
        """A run's scores are in its own folder, so they are read after the run has moved.

        Their config.yaml still names the run's first folder.
        """
        inputs, outputs = built
        first = scored_run(inputs, outputs, tmp_path / "elsewhere" / "abcd-run_s0")
        (outputs / "runs").mkdir()
        run = first.rename(outputs / "runs" / "abcd-run_s0")
        folder = compare.compare(
            "abcd-comparison", [str(run), OTHER], ["topmed"], inputs, outputs
        )
        assert pd.read_csv(folder / "metrics.csv").model.tolist() == [
            "abcd-run_s0",
            OTHER,
        ]

    @pytest.mark.parametrize("blank_ids", [False, True])
    def test_scores_of_other_variables_are_refused(self, built, blank_ids):
        """Two models' scores must list the same variables in the same order, or the paired tests cannot match them.

        Without ids, as GDC's variables have none, the names still tell the
        variables apart.
        """
        inputs, outputs = built
        scores(inputs, outputs, REFERENCE, RANKS[REFERENCE], blank_ids=blank_ids)
        scores(inputs, outputs, OTHER, RANKS[OTHER], reverse=True, blank_ids=blank_ids)
        with pytest.raises(ValueError, match="list other source variables"):
            compare.compare(
                "abcd-comparison", [REFERENCE, OTHER], ["topmed"], inputs, outputs
            )

    def test_the_reference_can_be_named_another_way(self, built, tmp_path, monkeypatch):
        """A reference given by a relative path is the model given by its full path."""
        inputs, outputs = built
        folder = tmp_path / "abcd-model"
        folder.mkdir()
        scores(
            inputs,
            outputs,
            str(folder.resolve()),
            RANKS[REFERENCE],
            parameters=PARAMETERS[REFERENCE],
        )
        monkeypatch.chdir(tmp_path)
        result = compare.compare(
            "abcd-comparison",
            [str(folder.resolve()), OTHER],
            ["topmed"],
            inputs,
            outputs,
            reference="./abcd-model",
        )
        differences = pd.read_csv(result / "differences.csv")
        assert set(differences.reference) == {"abcd-model"}

    def test_the_reference_is_matched_by_its_folder(self, built, tmp_path):
        """A reference is the model of its folder, not any model of its folder's name.

        Two folders of one name, abcd-model, take one name in the tables. The
        reference in the second folder is not the model in the first.
        """
        inputs, outputs = built
        first = tmp_path / "first" / "abcd-model"
        second = tmp_path / "second" / "abcd-model"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        scores(
            inputs,
            outputs,
            str(first.resolve()),
            RANKS[REFERENCE],
            parameters=PARAMETERS[REFERENCE],
        )
        with pytest.raises(ValueError, match="is not one of the models"):
            compare.compare(
                "abcd-comparison",
                [str(first), OTHER],
                ["topmed"],
                inputs,
                outputs,
                reference=str(second),
            )

    def test_one_model_named_twice_is_refused(self, built, tmp_path, monkeypatch):
        """A model folder given by its full path and by a relative path is one model, named twice."""
        inputs, outputs = built
        folder = tmp_path / "abcd-model"
        folder.mkdir()
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="names one model twice") as raised:
            compare.compare(
                "abcd-comparison",
                [str(folder.resolve()), "./abcd-model"],
                ["topmed"],
                inputs,
                outputs,
            )
        assert "./abcd-model" in str(raised.value)

    def test_two_models_of_one_name_are_refused(self, built, tmp_path):
        """Two model folders of one name would take one name in the tables and the legends."""
        inputs, outputs = built
        first = tmp_path / "first" / "abcd-model"
        second = tmp_path / "second" / "abcd-model"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        with pytest.raises(
            ValueError, match="two models have the same name: abcd-model"
        ):
            compare.compare(
                "abcd-comparison",
                [str(first), str(second)],
                ["topmed"],
                inputs,
                outputs,
            )

    def test_a_benchmark_named_twice_is_compared_once(self, built, monkeypatch):
        """A benchmark named twice is read and computed once, as score scores it once."""
        inputs, outputs = built
        read = []
        benchmark_result = compare.benchmark_result

        def counted(comparison, benchmark, *paths):
            """benchmark_result, which also records each benchmark that it reads."""
            read.append(benchmark)
            return benchmark_result(comparison, benchmark, *paths)

        monkeypatch.setattr(compare, "benchmark_result", counted)
        folder = compare.compare(
            "abcd-comparison", [REFERENCE, OTHER], ["topmed", "topmed"], inputs, outputs
        )
        assert read == ["topmed"]
        assert pd.read_csv(folder / "metrics.csv").model.tolist() == [REFERENCE, OTHER]

    @pytest.mark.parametrize(
        "name, models, reference, message",
        [
            ("abcd comparison", [REFERENCE, OTHER], None, "is not a comparison name"),
            ("abcd-comparison", [REFERENCE, REFERENCE], None, "names one model twice"),
            (
                "abcd-comparison",
                [REFERENCE, OTHER],
                "ijkl-org/ijkl-model",
                "not one of the models",
            ),
            (
                "abcd-comparison",
                [f"abcd-org/model-{n}" for n in range(9)],
                None,
                "at most 8",
            ),
            ("abcd-comparison", [REFERENCE], REFERENCE, "no other model"),
        ],
    )
    def test_a_comparison_that_cannot_be_drawn_is_refused(
        self, built, name, models, reference, message
    ):
        """A bad name, a model named twice, a reference not among the models or alone, or too many models are refused."""
        inputs, outputs = built
        with pytest.raises(ValueError, match=message):
            compare.compare(
                name, models, ["topmed"], inputs, outputs, reference=reference
            )
        assert not (outputs / "comparisons").exists()


def figure_results(count, reference_ranks=(1,) * 6, other_ranks=(2,) * 6):
    """Results on two benchmarks for the reference and count - 1 other models.

    By default the reference ranks each of six variables' expected target
    first, and each other model second. On accuracy@1 the two then differ on
    all six variables, and McNemar's p is 2 * 0.5 ** 6 = 0.03125: one star. On
    MRR each variable differs by 1 - 1/2, with no spread, so the paired
    t-test's p is 0: three stars. So each gets a bracket. At k of 5 and 10
    the models agree.
    """
    models = [REFERENCE] + [f"efgh-org/model-{n}" for n in range(2, count + 1)]
    benchmark_results, differences = {}, []
    for benchmark in ("topmed", "gdc"):
        result = {
            "version": "1",
            "n": len(reference_ranks),
            "metrics": {},
            "values": {},
        }
        for model in models:
            ranks = np.array(reference_ranks if model == REFERENCE else other_ranks)
            values = compare.variable_values(ranks)
            row = {
                **score_means(ranks),
                "median_rank": float(np.median(ranks)),
                "parameters": 1,
            }
            result["metrics"][model] = compare.metrics_row(
                model, benchmark, "1", "compact", row, values
            )
            result["values"][model] = values
        differences += compare.difference_rows(
            benchmark, REFERENCE, models[1:], result["values"]
        )
        benchmark_results[benchmark] = result
    return models, benchmark_results, differences


class TestFigures:
    """The figures, drawn as compare draws them."""

    @pytest.mark.parametrize("count", [2, 5])
    def test_every_bracket_sits_above_its_bars(self, count):
        """Each bracket starts above the intervals of its group, and its stars stay in the panel.

        Each model gives every variable the same rank. A Wilson interval still
        reaches above its bar: the other models' 0 of 6 at k of 1 runs up to
        0.39. With 5 models, four brackets stack over accuracy@1 and over MRR.
        """
        models, benchmark_results, differences = figure_results(count)
        with rc_context(compare.STYLE):
            figure = compare.overview_figure(
                benchmark_results, models, REFERENCE, differences
            )
            figure.draw_without_rendering()
        panels = figure.get_axes()
        assert len(panels) == 4
        for index, axes in enumerate(panels):
            metrics = benchmark_results[("topmed", "gdc")[index // 2]]["metrics"]
            group_metrics = (
                [f"accuracy@{k}" for k in compare.FIGURE_K]
                if index % 2 == 0
                else ["mrr"]
            )
            lines = [line for line in axes.lines if line.get_gid() == "bracket"]
            assert len(lines) == len(axes.texts) == count - 1
            for line in lines:
                metric = group_metrics[round(float(np.mean(line.get_xdata())))]
                highest = max(metrics[model][f"{metric}_high"] for model in models)
                assert min(line.get_ydata()) > highest
            top = axes.get_window_extent().y1
            assert all(mark.get_window_extent().y1 <= top for mark in axes.texts)

    def test_brackets_start_over_their_own_bars(self):
        """A bracket over low bars starts just above their intervals, below the top of higher bars.

        Of twelve variables, the reference ranks six first and six third, and
        the other model ranks all second. At k of 1 the bars are 0.5 and 0,
        and McNemar's p is 2 * 0.5 ** 6, so a bracket. The reference's Wilson
        interval for 6 of 12 reaches 0.7462. At k of 5 both bars are 1, with
        no bracket.
        """
        models, benchmark_results, differences = figure_results(
            2, reference_ranks=(1,) * 6 + (3,) * 6, other_ranks=(2,) * 12
        )
        with rc_context(compare.STYLE):
            figure = compare.overview_figure(
                benchmark_results, models, REFERENCE, differences
            )
            figure.draw_without_rendering()
        accuracy = figure.get_axes()[0]
        (line,) = [line for line in accuracy.lines if line.get_gid() == "bracket"]
        assert 0.7462 < min(line.get_ydata()) < 1.0

    def test_the_reference_comes_first_in_its_own_color(self):
        """The reference is the first bar of each group, and every model keeps the color of its place in --model.

        The other model is given first and the reference second, so the
        reference takes the second color, though it is drawn first.
        """
        (reference, other), benchmark_results, differences = figure_results(2)
        with rc_context(compare.STYLE):
            figure = compare.overview_figure(
                benchmark_results, [other, reference], reference, differences
            )
        accuracy = figure.get_axes()[0]
        first, second = [
            bars for bars in accuracy.containers if isinstance(bars, BarContainer)
        ]
        assert first.patches[0].get_facecolor() == pytest.approx(
            (*compare.colors()[1], 1)
        )
        assert second.patches[0].get_facecolor() == pytest.approx(
            (*compare.colors()[0], 1)
        )
        assert first.patches[0].get_x() < second.patches[0].get_x()
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert labels == [f"{reference} (reference)", other]
        assert figure.legends[0].get_title().get_text() == compare.STARS_NOTE

    def test_the_differences_figure_draws_each_row(self):
        """differences_from_reference puts the model at its difference on each metric's row.

        The other model ranks every expected target second, and the reference
        first. So on accuracy@1 it is 0 - 1 = -1 behind; at k of 5 and 10,
        both find every target, 1 - 1 = 0; on MRR it is 1/2 - 1 = -0.5. The
        rows run from accuracy@1 at the top, row 0, down to MRR, row 3.
        """
        models, benchmark_results, differences = figure_results(2)
        with rc_context(compare.STYLE):
            figure = compare.differences_figure(
                benchmark_results, models, REFERENCE, differences
            )
        axes = figure.get_axes()[0]
        points = [
            line for line in axes.lines if line.get_marker() == compare.MARKERS[1]
        ]
        assert [float(point.get_xdata()[0]) for point in points] == [-1, 0, 0, -0.5]
        assert [float(point.get_ydata()[0]) for point in points] == [0, 1, 2, 3]

    def test_the_bars_fill_their_panel(self):
        """Without brackets too, each bar panel ends half a group beyond its groups."""
        models, benchmark_results, _ = figure_results(2)
        with rc_context(compare.STYLE):
            figure = compare.overview_figure(benchmark_results, models, None, [])
            accuracy, mrr = figure.get_axes()[:2]
            assert accuracy.get_xlim() == (-0.5, 2.5)
            assert mrr.get_xlim() == (-0.5, 0.5)

    def test_the_curve_is_accuracy_at_every_k(self):
        """accuracy_by_k's line at k: the share of variables with a target in the first k.

        The reference's ranks are 1, 1, 1, 2, 1 and 3: four variables in the
        first one, five in the first two, and all six from k of 3.
        """
        benchmark_results = {
            "topmed": {
                "version": "1",
                "n": 6,
                "ranks": {REFERENCE: np.array(RANKS[REFERENCE])},
            }
        }
        with rc_context(compare.STYLE):
            figure = compare.curve_figure(benchmark_results, [REFERENCE], None)
        (line,) = figure.get_axes()[0].get_lines()
        assert line.get_ydata()[:4] == pytest.approx([4 / 6, 5 / 6, 1, 1])

    def test_each_line_carries_its_models_marker(self):
        """accuracy_by_k marks each line with the marker of its model's place in --model.

        The models are given as the other model, then the reference. The
        reference's line is drawn first, but keeps the marker of place 1.
        """
        benchmark_results = {
            "topmed": {
                "version": "1",
                "n": 6,
                "ranks": {model: np.array(ranks) for model, ranks in RANKS.items()},
            }
        }
        with rc_context(compare.STYLE):
            figure = compare.curve_figure(
                benchmark_results, [OTHER, REFERENCE], REFERENCE
            )
        lines = figure.get_axes()[0].get_lines()
        assert [line.get_marker() for line in lines] == [
            compare.MARKERS[1],
            compare.MARKERS[0],
        ]

    def test_the_size_axis_ends_on_whole_decades(self):
        """mrr_by_size's log axis ends on a power of ten on each side of the models.

        The models have 2.048 and 110 million parameters, so the axis runs
        from 1 to 1,000 million.
        """
        metrics = {
            model: {
                "parameters": PARAMETERS[model],
                "mrr": 0.7,
                "mrr_low": 0.6,
                "mrr_high": 0.8,
            }
            for model in (REFERENCE, OTHER)
        }
        benchmark_results = {"topmed": {"version": "1", "n": 6, "metrics": metrics}}
        with rc_context(compare.STYLE):
            figure = compare.size_figure(benchmark_results, [REFERENCE, OTHER], None)
            assert figure.get_axes()[0].get_xlim() == pytest.approx((1, 1000))


class TestCommandLine:
    """The `compare` command."""

    def test_compare_command(self, built):
        """compare writes the comparison's folder and returns 0."""
        inputs, outputs = built
        code = main(
            [
                "compare",
                "--name", "abcd-comparison",
                "--model", REFERENCE, OTHER,
                "--reference", REFERENCE,
                "--benchmark", "topmed",
                "--inputs", str(inputs),
                "--outputs", str(outputs),
            ]
        )  # fmt: skip
        assert code == 0
        assert (
            outputs / "comparisons" / "abcd-comparison" / "differences.csv"
        ).exists()

    def test_a_model_not_scored_is_a_clean_error(self, built, capsys):
        """A model without scores stops compare with the score command, not a traceback."""
        inputs, outputs = built
        code = main(
            [
                "compare",
                "--name", "abcd-comparison",
                "--model", REFERENCE, "ijkl-org/ijkl-model",
                "--benchmark", "topmed",
                "--inputs", str(inputs),
                "--outputs", str(outputs),
            ]
        )  # fmt: skip
        assert code == 1
        assert (
            "embedding_model score --model ijkl-org/ijkl-model"
            in capsys.readouterr().err
        )
