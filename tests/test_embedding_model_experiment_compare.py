"""Tests for embedding_model.experiment_compare: each arm of an experiment against its base.

Each run's manifest records a validation score, as ``train`` writes it. So
the tests check what the comparison adds: the statistics, the verdicts, the
recommendation, the figure and the files that it writes.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from embedding_model import experiment_compare, experiments
from embedding_model.__main__ import main

CORPUS = "abcd-corpus"
TOKENIZER = "abcd-tokenizer"
# The base of a first experiment: a small static model, on the CPU.
BASE = {
    "data": {"corpus": CORPUS, "tokenizer": TOKENIZER},
    "model": {"embedding_dim": 16},
    "training": {"num_train_epochs": 2, "use_cpu": True},
}
# Arms that change the table's size: one smaller and one larger, and one
# with a long name.
ARMS = {
    "dim8": {"test": "non-inferiority", "model": {"embedding_dim": 8}},
    "dim32": {"test": "superiority", "model": {"embedding_dim": 32}},
    "embedding_dim_64": {"test": "superiority", "model": {"embedding_dim": 64}},
}


def experiment_file(folder, name="abcd-experiment", **fields):
    """Write an experiment's file, the base and two arms at seeds 0 and 1 unless fields say otherwise."""
    path = folder / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump({"base": BASE, "seeds": [0, 1], "arms": ARMS, **fields})
    )
    return path


# The base's validation MRR@10 at five seeds: mean 0.801, seed sd 0.0016.
BASE_SCORES = [0.800, 0.802, 0.801, 0.799, 0.803]


def trained_experiment(
    folder, arm_scores, name="abcd-experiment", base=None, decision=None
):
    """An experiment trained at seeds 0 to 4: its file, the copy that training keeps, its configs and each run's manifest.

    The configs are written as ``experiment train`` writes them, by
    written_configs. ``arm_scores`` gives the five scores of the base and of each arm, by its
    name in ARMS. Each run's manifest records its score under the
    decision's metric, MRR@10 unless ``decision`` names another, and the
    benchmark that its corpus was checked against, TOPMed, as ``train``
    writes them. With ``base``, an earlier experiment's arm, the base's runs
    are that arm's, so ``arm_scores`` gives no base.
    """
    arms = {arm: ARMS[arm] for arm in arm_scores if arm != "base"}
    fields = {"seeds": [0, 1, 2, 3, 4], "arms": arms}
    if base is not None:
        fields["base"] = base
    if decision is not None:
        fields["decision"] = decision
    metric = (decision or {}).get("metric", "mrr@10")
    path = experiment_file(folder, name=name, **fields)
    experiment_folder = folder / "outputs" / "experiments" / path.stem
    experiment_folder.mkdir(parents=True)
    shutil.copy(path, experiment_folder / "experiment.yaml")
    experiment = experiments.read_experiment(path)
    base_sections, _ = experiments.base_config(experiment["base"], folder / "outputs")
    experiments.written_configs(
        experiments.arm_configs(experiment, base_sections), experiment_folder, 0
    )
    for arm, scores in arm_scores.items():
        for seed, score in enumerate(scores):
            run_folder = experiment_folder / "runs" / f"{arm}_s{seed}"
            run_folder.mkdir(parents=True)
            (run_folder / "manifest.json").write_text(
                json.dumps(
                    {
                        "run": run_folder.name,
                        "sources": [
                            {
                                "role": "corpus",
                                "name": CORPUS,
                                "benchmarks": [{"name": "topmed", "version": "3"}],
                            }
                        ],
                        "validation": {metric: score},
                    }
                )
            )
    return path


def comparison_table(folder, file_name, name="abcd-experiment"):
    """The rows of a table that experiment compare wrote, as dicts."""
    path = folder / "outputs" / "experiments" / name / "comparison"
    return pd.read_csv(path / file_name).to_dict("records")


class TestCompare:
    """experiment compare: each arm against the base, by its seeds' validation MRR@10."""

    def test_a_gain_of_at_least_the_minimum_effect_passes_superiority(
        self, tmp_path, capsys
    ):
        """dim32 gains 0.02 on the base, so it is better by at least 0.01, and it is recommended.

        Its scores are the base's five plus 0.02, in another order. With one
        arm, alpha stays 0.05, and Welch's t-test gives p near 4e-08. Its best
        run is its seed 3, at 0.823, which has no benchmark scores yet. Its
        corpus was checked against TOPMed alone, so the score command names
        TOPMed alone: score would refuse GDC.
        """
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim32": [0.821, 0.822, 0.820, 0.823, 0.819]},
        )
        assert experiment_compare.compare(path, tmp_path / "outputs") == 0
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert row["difference"] == 0.02
        assert row["p"] < 1e-6
        assert row["verdict"] == "better by at least 0.01"
        printed = capsys.readouterr().out
        assert "recommended: dim32, as better by at least 0.01" in printed
        assert "    base: abcd-experiment/dim32" in printed
        assert "Its best run is dim32_s3, at validation MRR@10 0.823." in printed
        assert (
            "score --model " in printed and "dim32_s3 --benchmark topmed\n" in printed
        )

    def test_a_sure_gain_under_the_minimum_effect_does_not_pass(self, tmp_path, capsys):
        """dim32's mean is 0.005 above the base's: significant, but under 0.01, so superiority does not pass.

        Superiority needs both: p under alpha, and a mean gain of at least the
        minimum effect. Here p is about 0.001, under 0.05. The interval ends
        below 0.01, so the gain is settled as too small: "not better", not
        "not shown better". No arm passes, and the base is recommended. dim32
        varies as the base does, by 0.0016, so 2 seeds would show a gain of
        0.01: t(0.975, 2) * 0.0016 = 0.0068.
        """
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim32": [0.807, 0.805, 0.808, 0.806, 0.804]},
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert row["difference"] == 0.005
        assert row["p"] < 0.05
        assert row["seeds_needed"] == 2
        assert row["high"] < 0.01
        assert row["verdict"] == "not better by at least 0.01"
        printed = capsys.readouterr().out
        assert "recommended: base, as no arm passes its test" in printed
        assert "    base: abcd-experiment/base" in printed

    def test_a_small_loss_passes_non_inferiority(self, tmp_path):
        """dim8 loses 0.003, and its interval's low end, about -0.005, stays above -0.01.

        Its scores are the base's five minus 0.003, in another order.
        """
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim8": [0.797, 0.799, 0.796, 0.800, 0.798]},
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert row["difference"] == -0.003
        assert -0.01 < row["low"] < -0.003
        assert row["verdict"] == "no more than 0.01 worse"

    def test_two_arms_are_each_tested_at_half_alpha(self, tmp_path):
        """With two arms, Bonferroni tests each at 0.05 / 2 = 0.025, with a 97.5% interval, and the manifest says so."""
        path = trained_experiment(
            tmp_path,
            {
                "base": BASE_SCORES,
                "dim8": [0.797, 0.799, 0.796, 0.800, 0.798],
                "dim32": [0.821, 0.822, 0.820, 0.823, 0.819],
            },
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        rows = comparison_table(tmp_path, "differences.csv")
        assert [(row["alpha"], row["confidence"]) for row in rows] == [
            (0.025, 0.975),
            (0.025, 0.975),
        ]
        folder = tmp_path / "outputs" / "experiments" / "abcd-experiment" / "comparison"
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["settings"]["alpha_per_arm"] == 0.025

    def test_a_loss_that_may_pass_the_margin_is_welchs_and_underpowered(self, tmp_path):
        """dim8 loses 0.009, with seeds that vary 3.6 times as much as the base's.

        dim32 makes two arms, so Bonferroni tests each at 0.05 / 2 = 0.025.
        dim8's interval is Welch's: the difference, minus and plus the t
        quantile at Welch-Satterthwaite's degrees of freedom, 4.6, times the
        standard error. Student's pooled interval would take 8. The low end,
        about -0.018, is past -0.01, so non-inferiority is not shown. At these
        spreads, Welch's power is 0.69 at 5 seeds and 0.83 at 6, so 6 give
        80%, and 5 are too few.
        """
        dim8_scores = [0.785, 0.800, 0.790, 0.795, 0.790]
        path = trained_experiment(
            tmp_path,
            {
                "base": BASE_SCORES,
                "dim8": dim8_scores,
                "dim32": [0.821, 0.822, 0.820, 0.823, 0.819],
            },
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        rows = comparison_table(tmp_path, "differences.csv")
        (row,) = [row for row in rows if row["arm"] == "dim8"]
        arm_variance = np.var(dim8_scores, ddof=1) / 5
        base_variance = np.var(BASE_SCORES, ddof=1) / 5
        standard_error = np.sqrt(arm_variance + base_variance)
        degrees_of_freedom = (arm_variance + base_variance) ** 2 / (
            arm_variance**2 / 4 + base_variance**2 / 4
        )
        low = -0.009 - stats.t.ppf(1 - 0.025 / 2, degrees_of_freedom) * standard_error
        assert row["df"] == 4.6
        assert row["low"] == round(low, 5)
        assert row["low"] < -0.01
        assert row["seeds_needed"] == 6
        assert row["verdict"] == (
            "may be more than 0.01 worse; underpowered: 6 seeds needed"
        )

    def test_a_loss_wholly_past_the_margin_is_not_underpowered(self, tmp_path):
        """dim8 loses 0.041, and its whole interval, about -0.061 to -0.021, lies past -0.01.

        Its seeds vary 10 times as much as the base's, so its test would need
        far more than its 5 seeds for 80% power. But the interval has settled
        the question: the arm is more than 0.01 worse, and more seeds would
        not make it pass. So it is not underpowered.
        """
        path = trained_experiment(
            tmp_path, {"base": BASE_SCORES, "dim8": [0.74, 0.78, 0.76, 0.75, 0.77]}
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert row["high"] < -0.01
        assert row["seeds_needed"] > 5
        assert not row["underpowered"]
        assert row["verdict"] == "more than 0.01 worse"

    def test_several_passing_arms_recommend_none(self, tmp_path, capsys):
        """dim8 is no more than 0.01 worse, and dim32 is better: which is worth its cost is the user's choice.

        Both base lines are printed, one for each arm.
        """
        path = trained_experiment(
            tmp_path,
            {
                "base": BASE_SCORES,
                "dim8": [0.797, 0.799, 0.796, 0.800, 0.798],
                "dim32": [0.821, 0.822, 0.820, 0.823, 0.819],
            },
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        printed = capsys.readouterr().out
        assert "recommended: none, as 2 arms pass their tests" in printed
        assert (
            "To build on one, write its line in the next experiment's file:" in printed
        )
        assert "    base: abcd-experiment/dim8\n" in printed
        assert "    base: abcd-experiment/dim32\n" in printed
        assert "Its best run" not in printed

    def test_it_writes_its_folder_whole(self, tmp_path):
        """comparison/ holds arms.csv, with each seed's score, differences.csv, the figure and the manifest.

        The base's sd is 0.0016, and its 95% interval 0.801 plus and minus
        t(0.975, 4) * 0.0016 / sqrt(5) = 0.0020. The manifest records the
        experiment's file and each of the 10 runs' manifests by checksum, the
        settings and the recommendation.
        """
        dim32_scores = [0.821, 0.822, 0.820, 0.823, 0.819]
        path = trained_experiment(
            tmp_path, {"base": BASE_SCORES, "dim32": dim32_scores}
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        folder = tmp_path / "outputs" / "experiments" / "abcd-experiment" / "comparison"
        arms = comparison_table(tmp_path, "arms.csv")
        assert [row["arm"] for row in arms] == ["base", "dim32"]
        assert [arms[1][f"s{seed}"] for seed in range(5)] == dim32_scores
        assert (arms[0]["mean"], arms[1]["mean"]) == (0.801, 0.821)
        assert (arms[0]["sd"], arms[0]["low"], arms[0]["high"]) == (
            0.00158,
            0.79904,
            0.80296,
        )
        assert sorted(path.name for path in (folder / "figures").iterdir()) == [
            "arms_and_differences.eps",
            "arms_and_differences.pdf",
            "arms_and_differences.png",
        ]
        manifest = json.loads((folder / "manifest.json").read_text())
        roles = [source["role"] for source in manifest["sources"]]
        assert roles == ["experiment"] + ["run"] * 10
        experiment_entry, first_run = manifest["sources"][0], manifest["sources"][1]
        assert experiment_entry["files"]["experiment.yaml"]["sha256"] == (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )
        run_manifest = Path(first_run["location"]) / "manifest.json"
        assert first_run["files"]["manifest.json"]["sha256"] == (
            hashlib.sha256(run_manifest.read_bytes()).hexdigest()
        )
        assert manifest["settings"]["alpha_per_arm"] == 0.05
        assert manifest["settings"]["test"] == "Welch's t-test"
        assert manifest["recommended"] == "dim32"

    def test_the_differences_center_on_0_with_both_margins(self, tmp_path):
        """The differences' axis runs as far left of 0 as right, and both panels shade the margin.

        dim32's interval reaches 0.02 + t(0.975, 8) * 0.001 = 0.0223, so the
        axis runs past -0.0223 and 0.0223 alike, beyond -0.01 and 0.01. The right panel's band spans
        -0.01 to 0.01, and the left panel's the base's mean, 0.801, plus and
        minus 0.01. Diamonds, which mark whether an arm passes, are drawn in
        the right panel only.
        """
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim32": [0.821, 0.822, 0.820, 0.823, 0.819]},
        )
        experiment = experiments.read_experiment(path)
        arm_scores = {
            "base": BASE_SCORES,
            "dim32": [0.821, 0.822, 0.820, 0.823, 0.819],
        }
        figure = experiment_compare.arms_figure(
            experiment,
            arm_scores,
            experiment_compare.arm_rows(experiment, arm_scores),
            experiment_compare.difference_rows(experiment, arm_scores),
        )
        left_end, right_end = figure.axes[1].get_xlim()
        assert left_end == -right_end
        assert right_end > 0.0223
        (band,) = figure.axes[1].patches
        assert band.get_x() == pytest.approx(-0.01)
        assert band.get_width() == pytest.approx(0.02)
        (base_band,) = figure.axes[0].patches
        assert base_band.get_x() == pytest.approx(0.801 - 0.01)
        markers = [[line.get_marker() for line in axes.lines] for axes in figure.axes]
        assert "D" not in markers[0]
        assert "D" in markers[1]

    @pytest.mark.filterwarnings("error")
    def test_scores_that_do_not_vary_give_an_exact_difference(self, tmp_path):
        """Every run of the base and of dim32 ranks every validation positive first: an MRR@10 of 1.

        Welch's t-test has no spread to measure, so scipy would give p as nan,
        with a warning. The difference, 0, is exact: its interval is [0, 0],
        and p is 1, as compare's paired t-test takes the same case. With no
        spread, 2 seeds would suffice, so the result is not underpowered.
        """
        path = trained_experiment(tmp_path, {"base": [1.0] * 5, "dim32": [1.0] * 5})
        experiment_compare.compare(path, tmp_path / "outputs")
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert (row["difference"], row["low"], row["high"], row["p"]) == (0, 0, 0, 1)
        assert row["seeds_needed"] == 2
        assert row["verdict"] == "not better by at least 0.01"

    def test_scores_that_do_not_vary_need_2_seeds_at_any_score(self, tmp_path):
        """Every run of the base scores 0.8123, and every run of dim8 0.8101.

        numpy's sd of five scores of 0.8123 is about 1e-16, a rounding error.
        The sd is exact, 0, so neither arm varies: 2 seeds suffice. dim8's
        loss of 0.0022 is exact too, and passes non-inferiority.
        """
        path = trained_experiment(
            tmp_path, {"base": [0.8123] * 5, "dim8": [0.8101] * 5}
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert row["seeds_needed"] == 2
        assert row["verdict"] == "no more than 0.01 worse"

    def test_the_numbers_line_up_after_the_longest_arm_name(self, tmp_path, capsys):
        """An arm named embedding_dim_64, 16 characters, pads each name to 18, so each number starts in one column.

        The base's mean is 0.801, and the arm scores 0.002 more at each seed.
        """
        arm_scores = [round(score + 0.002, 4) for score in BASE_SCORES]
        path = trained_experiment(
            tmp_path, {"base": BASE_SCORES, "embedding_dim_64": arm_scores}
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        printed = capsys.readouterr().out
        assert f"  {'base':<18}0.80100," in printed
        assert "  embedding_dim_64  +0.00200," in printed

    def test_a_gain_shown_as_the_minimum_effect_passes(self, tmp_path):
        """dim32 scores 0.57 at every seed, and the base 0.56: a gain of 0.01, which passes.

        In floating point, 0.57 - 0.56 is just under 0.01. The verdict reads
        the difference as differences.csv shows it, rounded, so the two agree.
        Neither arm varies, so the difference is exact, with p 0.
        """
        path = trained_experiment(tmp_path, {"base": [0.56] * 5, "dim32": [0.57] * 5})
        experiment_compare.compare(path, tmp_path / "outputs")
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert row["difference"] == 0.01
        assert row["verdict"] == "better by at least 0.01"

    def test_bonferroni_can_leave_a_gain_not_shown(self, tmp_path):
        """dim32 gains 0.012, at p = 0.043. Alone it passes; beside a second arm, at alpha 0.025, it does not.

        Here the base's seeds and dim32's vary by 0.0079 each, five times as
        much as BASE_SCORES. So the standard error is 0.005, t = 0.012 /
        0.005 = 2.4 at 8 degrees of freedom, and p is 0.043. Alone, dim32 is
        tested at 0.05, and passes. Beside dim8, it is tested at 0.025, and its
        97.5% interval, about -0.002 to 0.026, still reaches 0.01: not shown.
        A gain of 0.01 would be significant at 8 seeds, with 0.0099, not at
        7, with 0.0108, so 5 seeds are too few.
        """
        base_scores = [0.795, 0.805, 0.800, 0.790, 0.810]
        dim32_scores = [0.812, 0.802, 0.817, 0.807, 0.822]
        one_arm_path = trained_experiment(
            tmp_path, {"base": base_scores, "dim32": dim32_scores}
        )
        two_arm_path = trained_experiment(
            tmp_path,
            {
                "base": base_scores,
                "dim8": [0.797, 0.799, 0.796, 0.800, 0.798],
                "dim32": dim32_scores,
            },
            name="abcd-two-arm-experiment",
        )
        experiment_compare.compare(one_arm_path, tmp_path / "outputs")
        experiment_compare.compare(two_arm_path, tmp_path / "outputs")
        (alone,) = comparison_table(tmp_path, "differences.csv")
        (beside,) = [
            row
            for row in comparison_table(
                tmp_path, "differences.csv", "abcd-two-arm-experiment"
            )
            if row["arm"] == "dim32"
        ]
        assert alone["verdict"] == "better by at least 0.01"
        assert alone["p"] == beside["p"] == pytest.approx(0.043, abs=0.001)
        assert beside["verdict"] == (
            "not shown better by at least 0.01; underpowered: 8 seeds needed"
        )

    def test_an_open_question_with_enough_seeds_is_not_underpowered(self, tmp_path):
        """dim32 gains 0.008, and its interval, about 0.0057 to 0.0103, still reaches 0.01: not shown better.

        dim32 varies as the base does, by 0.0016, so 2 seeds would show a
        gain of 0.01: t(0.975, 2) * 0.0016 = 0.0068. Its 5 seeds are enough,
        so it is not underpowered: the gain may simply be under 0.01.
        """
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim32": [0.810, 0.808, 0.811, 0.807, 0.809]},
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert (row["low"], row["high"]) == (0.00569, 0.01031)
        assert row["seeds_needed"] == 2
        assert not row["underpowered"]
        assert row["verdict"] == "not shown better by at least 0.01"

    def test_the_decision_sets_the_metric_and_the_minimum_effect(
        self, tmp_path, capsys
    ):
        """A decision on validation accuracy@1, with a minimum effect of 0.02: dim32's gain of 0.015 is not enough.

        dim32 varies as the base does, so its interval is 0.015 plus and minus
        t(0.975, 8) * 0.001 = 0.0023, about 0.0127 to 0.0173: wholly under
        0.02, which settles it. At the default minimum effect, 0.01, it would
        pass.
        """
        decision = {"metric": "accuracy@1", "minimum_effect": 0.02}
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim32": [0.816, 0.817, 0.815, 0.818, 0.814]},
            decision=decision,
        )
        experiment_compare.compare(path, tmp_path / "outputs")
        (row,) = comparison_table(tmp_path, "differences.csv")
        assert (row["metric"], row["minimum_effect"]) == ("accuracy@1", 0.02)
        assert row["difference"] == 0.015
        assert row["verdict"] == "not better by at least 0.02"
        assert "validation accuracy@1, the base and 1 arm" in capsys.readouterr().out

    def test_the_figure_fills_the_diamond_of_an_arm_that_passes(self, tmp_path):
        """dim32 passes superiority, and dim8, 0.012 worse, does not pass non-inferiority.

        In the right panel, dim32's diamond is filled with its color, and
        dim8's is hollow: filled with the figure's surface. With two arms, the
        panel's label names Bonferroni. The scores come in the experiment's
        order of its arms, as compare reads them.
        """
        scores = {
            "base": BASE_SCORES,
            "dim8": [0.788, 0.790, 0.789, 0.787, 0.791],
            "dim32": [0.821, 0.822, 0.820, 0.823, 0.819],
        }
        experiment = experiments.read_experiment(trained_experiment(tmp_path, scores))
        arm_scores = {arm: scores[arm] for arm in ["base", *experiment["arms"]]}
        figure = experiment_compare.arms_figure(
            experiment,
            arm_scores,
            experiment_compare.arm_rows(experiment, arm_scores),
            experiment_compare.difference_rows(experiment, arm_scores),
        )
        names = [label.get_text() for label in figure.axes[0].get_yticklabels()]
        diamonds = {
            names[int(line.get_ydata()[0])]: line
            for line in figure.axes[1].lines
            if line.get_marker() == "D"
        }
        assert diamonds["dim32"].get_markerfacecolor() == diamonds["dim32"].get_color()
        assert diamonds["dim8"].get_markerfacecolor() == "#ffffff"
        assert "Bonferroni over 2 arms" in figure.axes[1].get_xlabel()

    def test_a_base_from_an_earlier_experiment_is_read_from_its_runs(
        self, tmp_path, capsys
    ):
        """abcd-later-experiment builds on abcd-experiment's dim8, so its base's scores are dim8's runs.

        dim8's five scores have a mean of 0.798. dim32 gains 0.002 on it, too
        little to pass, so the base is recommended. Its line names dim8 in
        abcd-experiment, where its runs are, so the next experiment reuses
        them.
        """
        trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim8": [0.797, 0.799, 0.796, 0.800, 0.798]},
        )
        later_path = trained_experiment(
            tmp_path,
            {"dim32": [0.799, 0.801, 0.798, 0.802, 0.800]},
            name="abcd-later-experiment",
            base="abcd-experiment/dim8",
        )
        experiment_compare.compare(later_path, tmp_path / "outputs")
        arms = comparison_table(tmp_path, "arms.csv", "abcd-later-experiment")
        assert (arms[0]["arm"], arms[0]["mean"]) == ("base", 0.798)
        printed = capsys.readouterr().out
        assert "recommended: base" in printed
        assert "    base: abcd-experiment/dim8\n" in printed

    def test_an_earlier_run_still_training_stops_the_comparison(self, tmp_path):
        """A base run of the earlier experiment with a staging folder may be training, so the comparison waits for it.

        The message is the one that experiment train gives: wait, or delete
        the staging folder.
        """
        trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim8": [0.797, 0.799, 0.796, 0.800, 0.798]},
        )
        runs_folder = tmp_path / "outputs" / "experiments" / "abcd-experiment" / "runs"
        shutil.rmtree(runs_folder / "dim8_s4")
        (runs_folder / ".dim8_s4~x1y2").mkdir()
        later_path = trained_experiment(
            tmp_path,
            {"dim32": [0.799, 0.801, 0.798, 0.802, 0.800]},
            name="abcd-later-experiment",
            base="abcd-experiment/dim8",
        )
        with pytest.raises(ValueError, match="has started but not finished") as refused:
            experiment_compare.compare(later_path, tmp_path / "outputs")
        assert "Wait for it to finish" in str(refused.value)
        assert f"delete its staging folder, {runs_folder / '.dim8_s4~x1y2'}" in str(
            refused.value
        )

    def test_a_corpus_checked_against_no_benchmark_says_what_to_build(
        self, tmp_path, capsys
    ):
        """A corpus built before any benchmark was checked against none, so score would refuse every one.

        The message then says to build the benchmarks, and the corpus again,
        rather than give a score command that fails.
        """
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim32": [0.821, 0.822, 0.820, 0.823, 0.819]},
        )
        runs_folder = tmp_path / "outputs" / "experiments" / "abcd-experiment" / "runs"
        for record_path in runs_folder.glob("*/manifest.json"):
            record = json.loads(record_path.read_text())
            record["sources"][0]["benchmarks"] = []
            record_path.write_text(json.dumps(record))
        experiment_compare.compare(path, tmp_path / "outputs")
        printed = capsys.readouterr().out
        assert "checked against no benchmark" in printed
        assert "score --model" not in printed

    def test_a_run_that_is_not_done_stops_the_comparison(self, tmp_path):
        """An arm is read only when all its seeds are done. The message names the run and the command."""
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim32": [0.821, 0.822, 0.820, 0.823, 0.819]},
        )
        run_folder = (
            tmp_path
            / "outputs"
            / "experiments"
            / "abcd-experiment"
            / "runs"
            / "dim32_s4"
        )
        shutil.rmtree(run_folder)
        with pytest.raises(ValueError, match=r"dim32_s4 \(not started\)") as error:
            experiment_compare.compare(path, tmp_path / "outputs")
        assert f"experiment train --config {path}" in str(error.value)

    def test_one_seed_is_refused(self, tmp_path):
        """One seed has no seed-to-seed spread, so its arms cannot be compared."""
        path = experiment_file(tmp_path, seeds=[0])
        with pytest.raises(ValueError, match="has one seed"):
            experiment_compare.compare(path, tmp_path / "outputs")


class TestSeedsNeeded:
    """The fewest seeds per arm that an arm's test needs."""

    def test_at_a_seed_sd_of_0_0045(self):
        """Both arms vary by 0.0045 from seed to seed; alpha is 0.025 and the minimum effect 0.01.

        Superiority: a gain of 0.01 is significant once 0.01 is at least
        t(0.9875, 2(n - 1)) * 0.0045 * sqrt(2 / n). That is 0.0128 at 3 seeds
        and 0.0094 at 4. Non-inferiority: Welch's power, when the arms do not
        differ, is 0.76 at 5 seeds and 0.87 at 6, so 6 give 80%.
        """
        assert (
            experiment_compare.seeds_needed("superiority", 0.0045, 0.0045, 0.025, 0.01)
            == 4
        )
        assert (
            experiment_compare.seeds_needed(
                "non-inferiority", 0.0045, 0.0045, 0.025, 0.01
            )
            == 6
        )

    def test_a_wide_spread_needs_many_seeds(self):
        """At a seed sd of 0.02, superiority needs 42 seeds: the count is the first that suffices.

        A gain of 0.01 is significant at n seeds once 0.01 is at least
        t(0.9875, 2(n - 1)) * 0.02 * sqrt(2 / n), so the count suffices and
        the one before it does not.
        """
        needed_count = experiment_compare.seeds_needed(
            "superiority", 0.02, 0.02, 0.025, 0.01
        )

        def margin_reached(seed_count):
            """Whether a gain of 0.01 is significant at seed_count seeds per arm."""
            return 0.01 >= stats.t.ppf(0.9875, 2 * (seed_count - 1)) * 0.02 * np.sqrt(
                2 / seed_count
            )

        assert needed_count == 42
        assert margin_reached(needed_count) and not margin_reached(needed_count - 1)


class TestCommandLine:
    """experiment compare on the command line."""

    def test_compare(self, tmp_path, capsys):
        """experiment compare compares a trained experiment's arms, and prints its recommendation."""
        path = trained_experiment(
            tmp_path,
            {"base": BASE_SCORES, "dim32": [0.821, 0.822, 0.820, 0.823, 0.819]},
        )
        flags = ["--config", str(path), "--outputs", str(tmp_path / "outputs")]
        assert main(["experiment", "compare", *flags]) == 0
        printed = capsys.readouterr().out
        assert "recommended: dim32" in printed
        folder = tmp_path / "outputs" / "experiments" / "abcd-experiment" / "comparison"
        assert (
            f"     figure: {folder / 'figures' / 'arms_and_differences.png'}\n"
            in printed
        )
        assert f"     table: {folder / 'differences.csv'}\n" in printed
