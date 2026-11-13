"""Compare an experiment's arms with its base, and recommend an arm to build on.

``experiment compare`` reads each run's score on the validation pairs, by the
decision's metric, from the run's manifest. It compares each arm with the
base by Welch's interval and t-test, at the decision's alpha split over the
arms, and gives each arm its verdict and the seeds that its test needs. It
writes ``comparison/`` in the experiment's folder whole: ``arms.csv``,
``differences.csv``, the figure in ``figures/`` and ``manifest.json``. It
prints the verdicts, the recommendation, and the ``base:`` line that builds
the next experiment on it. experiments.py reads the experiment's file, and
finds its runs.
"""

import json
import math
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import elementwise

from embedding_model import manifests
from embedding_model.compare import (
    COMPARE_LIBRARIES,
    CONFIDENCE,
    FIGURES,
    MUTED,
    PRINT_WIDTH,
    STYLE,
    SURFACE,
    TEXT,
    colors,
    mark_sides,
    print_destination,
    save_figure,
    t_interval,
)
from embedding_model.experiments import (
    BASE,
    DONE,
    EXPERIMENT_FILE,
    base_config,
    check_earlier_runs,
    checked_experiment,
    experiment_runs,
    run_state,
)
from embedding_model.folders import staged
from embedding_model.provenance import code_commit, library_versions
from embedding_model.scoring import corpus_benchmarks

# The folder of the experiment's folder that ``experiment compare`` writes
# whole, and the stem of its figure.
COMPARISON = "comparison"
COMPARISON_FIGURE = "arms_and_differences"
# The power that non-inferiority's seeds needed aim at: 80%, the usual
# convention (Cohen, 1988), which weighs a false finding as four times as
# serious as a missed one.
TARGET_POWER = 0.8
# The gray of the figure's bands at the minimum effect: lighter than the grid,
# so that the band stays behind the marks, and each role's color keeps a 3:1
# contrast on it.
BAND = "#f4f3f0"
# The figure's color for the base and for each test: three of matplotlib's
# petroff10 colors, its blue, dark orange and purple, by their places in the
# sequence. They stay apart under each color vision deficiency, and keep a 3:1
# contrast on white and on BAND. Red is left out, which a user could take
# for a failed test.
ROLE_COLOR_PLACES = {"base": 0, "non-inferiority": 6, "superiority": 4}
# Each test's verdict, by the arm's outcome: it passes its test; it does not,
# and its interval still reaches the margin, so its question stays open; or
# its interval lies wholly past the margin, which settles the question.
VERDICTS = {
    "superiority": {
        "passes": "better by at least {effect}",
        "open": "not shown better by at least {effect}",
        "settled": "not better by at least {effect}",
    },
    "non-inferiority": {
        "passes": "no more than {effect} worse",
        "open": "may be more than {effect} worse",
        "settled": "more than {effect} worse",
    },
}
# The decimals of arms.csv and differences.csv. A run records its validation
# metrics to 4 decimals, and a mean over seeds needs one more.
DECIMALS = 5


def metric_name(metric):
    """A decision metric as a user sees it, in the figure and the printout: MRR@10, accuracy@1.

    The experiment's file and the tables keep the metric's key, such as
    ``mrr@10``, as a run's manifest records it.

    Args:
        metric (str): The decision's metric.

    Returns:
        str: Its name.
    """
    return metric.replace("mrr", "MRR")


def seed_scores(experiment, runs, path):
    """Each arm's score at each seed: the decision's metric on the validation pairs.

    A run's manifest records its validation metrics, as ``train`` wrote them.
    An arm is read only when all its runs are done.

    Args:
        experiment (dict): The experiment, from read_experiment.
        runs (list[Run]): Its runs, from experiment_runs.
        path (Path): The experiment's file, for the command in a message.

    Returns:
        dict[str, list[float]]: Each arm's scores, in seed order, the base's
            first, under BASE.

    Raises:
        ValueError: If a run is not done.
    """
    unfinished_run_states = [
        f"{run.folder.name} ({state})"
        for run in runs
        if (state := run_state(run.folder)) != DONE
    ]
    if unfinished_run_states:
        raise ValueError(
            f"experiment {experiment['name']} has runs that are not done:"
            f" {', '.join(unfinished_run_states)}. Train them first: python -m"
            f" embedding_model experiment train --config {path}"
        )
    metric = experiment["decision"]["metric"]
    arm_scores = {}
    for run in runs:
        run_manifest = json.loads(
            (run.folder / "manifest.json").read_text(encoding="utf-8")
        )
        arm_scores.setdefault(run.arm, []).append(
            float(run_manifest["validation"][metric])
        )
    return arm_scores


def seeds_needed(test, base_sd, arm_sd, alpha, minimum_effect):
    """The fewest seeds per arm that the arm's test needs, at the two arms' seed sds.

    Superiority's is the fewest seeds at which a mean gain of exactly the
    minimum effect is significant: the sample size of a dual-criterion
    design (Roychoudhury et al., 2018), by Welch's t. With more seeds, a gain
    larger than the minimum effect passes more often. A gain of exactly the
    minimum effect passes at most half the time, as its mean falls on either
    side, and about half the time once the seeds suffice. Non-inferiority's
    is the fewest seeds that give TARGET_POWER to show it when the arm and
    the base do not differ: the power of Welch's test, from the noncentral
    t. Both take Welch-Satterthwaite's degrees of freedom at the two sds,
    with the same seeds in each arm.

    The count is the one at which the test's need is met exactly, rounded up.
    scipy's optimize.elementwise finds it in two steps. bracket_root finds
    two counts, one too few and one enough. find_root then finds the exact
    count between them. When neither arm's scores vary, 2 seeds suffice.

    Args:
        test (str): The arm's test, superiority or non-inferiority.
        base_sd (float): The base's seed-to-seed sd.
        arm_sd (float): The arm's seed-to-seed sd.
        alpha (float): The alpha of each arm, after Bonferroni.
        minimum_effect (float): The decision's minimum effect.

    Returns:
        int: The seeds per arm, at least 2.
    """

    # Without spread, the difference is known exactly, and more seeds would
    # not change the verdict. 2 seeds are the fewest that give an sd.
    if base_sd == 0 and arm_sd == 0:
        return 2

    def surplus(seed_count):
        """What seed_count seeds per arm give beyond the test's need: 0 or more when they suffice.

        seed_count may be fractional, and an array, as bracket_root and
        find_root pass it.
        """
        # The variance of each arm's mean score, at seed_count seeds.
        base_mean_variance = base_sd**2 / seed_count
        arm_mean_variance = arm_sd**2 / seed_count
        difference_standard_error = np.sqrt(base_mean_variance + arm_mean_variance)
        # Welch-Satterthwaite's degrees of freedom, with seed_count seeds in
        # each arm.
        degrees_of_freedom = (base_mean_variance + arm_mean_variance) ** 2 / (
            (base_mean_variance**2 + arm_mean_variance**2) / (seed_count - 1)
        )
        # The two-sided 1 - alpha interval reaches critical_t standard errors
        # on each side of the difference.
        critical_t = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
        if test == "superiority":
            # A gain is significant when the interval's low end is above 0:
            # when the gain is more than critical_t standard errors.
            smallest_significant_gain = critical_t * difference_standard_error
            return minimum_effect - smallest_significant_gain
        # Non-inferiority passes when the interval's low end is above
        # -minimum_effect: when (difference + minimum_effect) /
        # difference_standard_error is above critical_t. When the arms do not
        # differ, that ratio follows the noncentral t, with this
        # noncentrality. Its chance to be above critical_t is the power.
        noncentrality = minimum_effect / difference_standard_error
        power = stats.nct.sf(critical_t, degrees_of_freedom, noncentrality)
        return power - TARGET_POWER

    if surplus(2) >= 0:
        return 2
    # surplus is below 0 at short_count, and 0 or more at enough_count.
    # bracket_root starts from 2 and 3 seeds, and moves the upper count up
    # until surplus is 0 or more there. xmin keeps the lower count at 2 or
    # more, since Welch's degrees of freedom need 2 seeds.
    short_count, enough_count = elementwise.bracket_root(surplus, 2, xmin=2).bracket
    # The count at which surplus is exactly 0, which may be fractional.
    # surplus only grows with more seeds, so this count rounded up is the
    # fewest whole seeds that suffice.
    exact_count = elementwise.find_root(surplus, (short_count, enough_count)).x
    return math.ceil(exact_count)


def welch_difference(scores, base_scores, alpha):
    """An arm's mean minus the base's, with Welch's interval and t-test.

    scipy's ``ttest_ind``, with ``equal_var=False``, gives the interval at
    one minus alpha, the degrees of freedom and the two-sided p. When
    neither arm's scores vary, such as when every run scores 1, the t-test
    has no spread to measure. The difference is then exact: its interval is
    the difference itself, and p is 1 when the difference is 0 and 0
    otherwise, as compare's paired t-test takes the same case.

    Args:
        scores (list[float]): The arm's scores, one per seed.
        base_scores (list[float]): The base's scores, one per seed.
        alpha (float): The alpha of each arm, after Bonferroni.

    Returns:
        tuple[float, float, float, float | None, float]: The difference, its
            interval's low and high ends, the degrees of freedom (None with
            no spread) and p.
    """
    difference = float(np.mean(scores) - np.mean(base_scores))
    if min(scores) == max(scores) and min(base_scores) == max(base_scores):
        p = 1.0 if difference == 0 else 0.0
        return difference, difference, difference, None, p
    welch_test = stats.ttest_ind(scores, base_scores, equal_var=False)
    interval = welch_test.confidence_interval(1 - alpha)
    return (
        difference,
        float(interval.low),
        float(interval.high),
        float(welch_test.df),
        float(welch_test.pvalue),
    )


def arm_rows(experiment, arm_scores):
    """Each arm's row of arms.csv: its scores' mean, sd and 95% t interval, and each seed's score.

    The sd is statistics.stdev's, which is exact when every seed scores the
    same: it is then 0. numpy's can be a rounding error instead, such as
    1e-16 for five scores of 0.8123.

    Args:
        experiment (dict): The experiment, from read_experiment.
        arm_scores (dict[str, list[float]]): Each arm's scores, from
            seed_scores.

    Returns:
        list[dict]: One row per arm, the base's first.
    """
    rows = []
    for arm, scores in arm_scores.items():
        low, high = t_interval(scores)
        rows.append(
            {
                "arm": arm,
                "test": None if arm == BASE else experiment["arms"][arm]["test"],
                "metric": experiment["decision"]["metric"],
                "seeds": len(scores),
                "mean": round(float(np.mean(scores)), DECIMALS),
                "sd": round(statistics.stdev(scores), DECIMALS),
                "low": round(low, DECIMALS),
                "high": round(high, DECIMALS),
                **{
                    f"s{seed}": score
                    for seed, score in zip(experiment["seeds"], scores)
                },
            }
        )
    return rows


def difference_rows(experiment, arm_scores):
    """Each arm minus the base: Welch's interval and test, the verdict and the seeds needed.

    Each arm is tested at the decision's alpha divided by the number of arms,
    by Bonferroni, so the chance of any false finding over the experiment
    is at most alpha. Its interval is Welch's two-sided interval at one minus
    that alpha, and its p is Welch's t-test of no difference, so the two
    agree.

    - The superiority test passes when the arm is significantly better, with
      a mean gain of at least the minimum effect: a dual-criterion design
      (Roychoudhury et al., 2018).
    - The non-inferiority test passes when the interval's low end is above
      minus the minimum effect, as in the extension for non-inferiority
      trials of CONSORT, the guideline for reporting randomized trials
      (Piaggio et al., 2012).

    An arm that does not pass its test, while its interval still reaches the
    margin, leaves its question open: it says too little, either way. With
    fewer seeds than the test needs, it is also underpowered. An interval
    wholly past the margin, such as a loss of -0.04 to -0.02 against -0.01,
    settles the question, however few the seeds. VERDICTS words each
    outcome. The verdict reads the difference and the interval as
    differences.csv shows them, rounded to DECIMALS, so a gain shown as 0.01
    is a gain of 0.01.

    Args:
        experiment (dict): The experiment, from read_experiment.
        arm_scores (dict[str, list[float]]): Each arm's scores, from
            seed_scores.

    Returns:
        list[dict]: One row per arm, in the file's order.
    """
    decision = experiment["decision"]
    minimum_effect = decision["minimum_effect"]
    alpha = decision["alpha"] / len(experiment["arms"])
    base_scores = arm_scores[BASE]
    # statistics.stdev is exact when every seed scores the same, as
    # seeds_needed's check for no spread needs. numpy's sd can be a rounding
    # error instead, such as 1e-16.
    base_sd = statistics.stdev(base_scores)
    rows = []
    for arm, arm_fields in experiment["arms"].items():
        scores = arm_scores[arm]
        test = arm_fields["test"]
        difference, low, high, degrees_of_freedom, p = welch_difference(
            scores, base_scores, alpha
        )
        difference, low, high = (
            round(value, DECIMALS) for value in (difference, low, high)
        )
        # The interval reaches the margin when it holds a difference that
        # would pass: a gain of at least minimum_effect for superiority, and a
        # loss of less than minimum_effect for non-inferiority.
        if test == "superiority":
            passes = p < alpha and difference >= minimum_effect
            margin_in_reach = high >= minimum_effect
        else:
            passes = low > -minimum_effect
            margin_in_reach = high > -minimum_effect
        if passes:
            outcome = "passes"
        elif margin_in_reach:
            outcome = "open"
        else:
            outcome = "settled"
        verdict = VERDICTS[test][outcome].format(effect=f"{minimum_effect:g}")
        needed_count = seeds_needed(
            test, base_sd, statistics.stdev(scores), alpha, minimum_effect
        )
        underpowered = outcome == "open" and len(scores) < needed_count
        if underpowered:
            verdict += f"; underpowered: {needed_count} seeds needed"
        rows.append(
            {
                "arm": arm,
                "test": test,
                "metric": decision["metric"],
                "difference": difference,
                "low": low,
                "high": high,
                "confidence": round(1 - alpha, 6),
                "df": (
                    None if degrees_of_freedom is None else round(degrees_of_freedom, 1)
                ),
                "p": p,
                "alpha": round(alpha, 6),
                "minimum_effect": minimum_effect,
                "passes": passes,
                "seeds_needed": needed_count,
                "underpowered": underpowered,
                "verdict": verdict,
            }
        )
    return rows


def recommendation(experiment, differences):
    """The arm that the comparison recommends, why, and the base lines to copy.

    One arm that passes its test is recommended. With none, the base is.
    With several, none is: which is worth its cost is the user's choice,
    and each one's line is given. The base's line names this experiment's
    base, whose runs the next experiment reuses, wherever they were trained.

    Args:
        experiment (dict): The experiment, from read_experiment.
        differences (list[dict]): The rows of difference_rows.

    Returns:
        tuple[str | None, str, list[str]]: The recommended arm, BASE, or None;
            the reason; and the ``base:`` line of each arm to build on.
    """
    name = experiment["name"]
    passing_rows = [row for row in differences if row["passes"]]
    if not passing_rows:
        return BASE, "no arm passes its test", [f"base: {name}/{BASE}"]
    base_lines = [f"base: {name}/{row['arm']}" for row in passing_rows]
    if len(passing_rows) == 1:
        return passing_rows[0]["arm"], passing_rows[0]["verdict"], base_lines
    return (
        None,
        f"{len(passing_rows)} arms pass their tests: choose the one worth its cost",
        base_lines,
    )


def arms_figure(experiment, arm_scores, arms, differences):
    """Each arm's seeds, with its mean and interval, beside each arm minus the base.

    Each row is one arm, the base's first, so a row reads across both panels.
    On the left, each seed's score is a dot, in a strip just above the row,
    and the arm's mean is a bar on its 95% t interval, on the row itself, so
    that it lines up with the arm's diamond on the right; a gray band spans
    the base's mean, plus and minus the minimum effect. The raw scores beside
    their summary are the layout of an estimation plot (Ho et al., 2019).
    On the right, each arm minus the base has Welch's interval, at the
    confidence of its test. 0 is at the center, and a gray band spans
    minus to plus the minimum effect, with a dashed line at each end, as in
    the figure of CONSORT, the guideline for reporting randomized trials, for
    non-inferiority trials (Piaggio et al., 2012). The color says the arm's
    test, and on the right only, a filled diamond that it passes, a hollow
    one that it does not. The legend shows the colors as swatches, so that a
    diamond means one thing. The verdicts' words are in the printout and in
    differences.csv: under each row, they crowd the panel past a few arms.

    Args:
        experiment (dict): The experiment, from read_experiment.
        arm_scores (dict[str, list[float]]): Each arm's scores, from
            seed_scores.
        arms (list[dict]): The rows of arm_rows.
        differences (list[dict]): The rows of difference_rows.

    Returns:
        Figure: The figure.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    decision = experiment["decision"]
    minimum_effect = decision["minimum_effect"]
    palette = colors()
    role_colors = {role: palette[place] for role, place in ROLE_COLOR_PLACES.items()}
    figure = Figure(figsize=(PRINT_WIDTH, 1.3 + 0.45 * len(arms)), layout="constrained")
    figure.suptitle(
        f"Experiment {experiment['name']}: validation {metric_name(decision['metric'])},"
        f" at {len(experiment['seeds'])} seeds"
    )
    seeds_axes, differences_axes = figure.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 1.2]}
    )
    for place, row in enumerate(arms):
        role_color = role_colors[row["test"] or BASE]
        scores = arm_scores[row["arm"]]
        offsets = np.linspace(-0.34, -0.14, len(scores))
        seeds_axes.plot(
            scores,
            place + offsets,
            "o",
            color=role_color,
            markersize=3.5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
            zorder=2,
        )
        seeds_axes.errorbar(
            row["mean"],
            place,
            xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
            fmt="|",
            color=role_color,
            markersize=10,
            markeredgewidth=2.5,
            elinewidth=1.4,
            capsize=3,
            zorder=3,
        )
    base_mean = arms[0]["mean"]
    seeds_axes.axvspan(
        base_mean - minimum_effect,
        base_mean + minimum_effect,
        color=BAND,
        linewidth=0,
        zorder=0,
    )
    seeds_axes.set_yticks(range(len(arms)), [row["arm"] for row in arms])
    seeds_axes.set_ylim(len(arms) - 0.5, -0.5)
    seeds_axes.grid(axis="y", visible=False)
    seeds_axes.set_xlabel(
        f"Each seed's score, and the mean with its {CONFIDENCE:.0%} interval"
    )
    widest_end = minimum_effect
    for place, row in enumerate(differences, start=1):
        role_color = role_colors[row["test"]]
        differences_axes.errorbar(
            row["difference"],
            place,
            xerr=[
                [row["difference"] - row["low"]],
                [row["high"] - row["difference"]],
            ],
            fmt="D",
            color=role_color,
            markerfacecolor=role_color if row["passes"] else SURFACE,
            markeredgecolor=SURFACE if row["passes"] else role_color,
            markeredgewidth=0.6 if row["passes"] else 1.4,
            markersize=6,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
        widest_end = max(widest_end, abs(row["low"]), abs(row["high"]))
    differences_axes.axvspan(
        -minimum_effect, minimum_effect, color=BAND, linewidth=0, zorder=0
    )
    differences_axes.axvline(0, color=TEXT, linewidth=0.8, zorder=1)
    for margin in (-minimum_effect, minimum_effect):
        differences_axes.axvline(
            margin, color=MUTED, linewidth=0.8, linestyle="--", zorder=1
        )
    differences_axes.set_xlim(-1.15 * widest_end, 1.15 * widest_end)
    differences_axes.grid(axis="y", visible=False)
    confidence = differences[0]["confidence"]
    arm_count = len(differences)
    differences_axes.set_xlabel(
        f"Arm minus the base, with {confidence:.1%} intervals"
        + (f"\n(Bonferroni over {arm_count} arms)" if arm_count > 1 else "")
    )
    mark_sides(differences_axes, "the base")
    seeds_axes.xaxis.labelpad = differences_axes.xaxis.labelpad
    figure.align_xlabels([seeds_axes, differences_axes])
    drawn_tests = {row["test"] for row in differences}
    handles = [
        Patch(facecolor=role_colors[role], label=label)
        for role, label in (
            (BASE, "the base"),
            ("non-inferiority", "non-inferiority arm"),
            ("superiority", "superiority arm"),
        )
        if role == BASE or role in drawn_tests
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            color=MUTED,
            label="right panel: passes its test",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markerfacecolor=SURFACE,
            markeredgecolor=MUTED,
            markeredgewidth=1.4,
            label="right panel: does not pass",
        ),
        Patch(
            facecolor=BAND,
            edgecolor=MUTED,
            linestyle="--",
            linewidth=0.6,
            label=f"within \u00b1{minimum_effect:g} of the base",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    return figure


@dataclass
class Comparison:
    """An experiment's comparison: its runs and scores, its two tables and its recommendation.

    Attributes:
        experiment (dict): The experiment, from read_experiment.
        folder (Path): The experiment's folder.
        runs (list[Run]): Its runs, from experiment_runs.
        arm_scores (dict[str, list[float]]): Each arm's scores, from
            seed_scores.
        arms (list[dict]): The rows of arms.csv, from arm_rows.
        differences (list[dict]): The rows of differences.csv, from
            difference_rows.
        recommended_arm (str | None): The recommended arm, BASE, or None, from
            recommendation.
        reason (str): Why, from recommendation.
        base_lines (list[str]): The ``base:`` line of each arm to build on,
            from recommendation.
    """

    experiment: dict
    folder: Path
    runs: list
    arm_scores: dict
    arms: list
    differences: list
    recommended_arm: str | None
    reason: str
    base_lines: list


def comparison_manifest(comparison):
    """The comparison's manifest: what it read, its settings and its recommendation.

    Args:
        comparison (Comparison): The comparison.

    Returns:
        dict: The manifest's fields.
    """
    experiment, folder = comparison.experiment, comparison.folder
    differences = comparison.differences
    return {
        "experiment": experiment["name"],
        "built": date.today().isoformat(),
        "sources": [
            {
                "role": "experiment",
                "location": str(folder),
                "files": {
                    EXPERIMENT_FILE: {
                        "sha256": manifests.file_sha256(folder / EXPERIMENT_FILE)
                    }
                },
            },
            *[
                {
                    "role": "run",
                    "arm": run.arm,
                    "seed": run.seed,
                    "location": str(run.folder),
                    "files": {
                        "manifest.json": {
                            "sha256": manifests.file_sha256(
                                run.folder / "manifest.json"
                            )
                        }
                    },
                }
                for run in comparison.runs
            ],
        ],
        "settings": {
            **experiment["decision"],
            "alpha_per_arm": differences[0]["alpha"],
            "correction": "Bonferroni",
            "interval": "Student's t interval, over the seeds",
            "confidence": CONFIDENCE,
            "difference_interval": "Welch's interval",
            "difference_confidence": differences[0]["confidence"],
            "test": "Welch's t-test",
            "power": "Welch's t-test's, from the noncentral t distribution",
            "target_power": TARGET_POWER,
        },
        "seeds": experiment["seeds"],
        "recommended": comparison.recommended_arm,
        "ai_harmonization": code_commit(),
        "libraries": library_versions(COMPARE_LIBRARIES),
    }


def write_comparison(comparison, destination):
    """Write the comparison's folder whole: arms.csv, differences.csv, the figure and manifest.json.

    Args:
        comparison (Comparison): The comparison.
        destination (Path): The comparison's folder, in the experiment's.
    """
    from matplotlib import rc_context

    with staged(destination) as built, rc_context(STYLE):
        pd.DataFrame(comparison.arms).to_csv(built / "arms.csv", index=False)
        pd.DataFrame(comparison.differences).to_csv(
            built / "differences.csv", index=False
        )
        (built / FIGURES).mkdir()
        figure = arms_figure(
            comparison.experiment,
            comparison.arm_scores,
            comparison.arms,
            comparison.differences,
        )
        save_figure(figure, built / FIGURES, COMPARISON_FIGURE)
        manifests.write(built / "manifest.json", comparison_manifest(comparison))


def best_run_line(comparison):
    """The recommended arm's best run, by the decision's metric, and the command that scores it.

    The command names the benchmarks that the run's corpus was checked
    against, since score refuses any other. With none, the line says what to
    build instead.

    Args:
        comparison (Comparison): The comparison, with an arm recommended.

    Returns:
        str: The line to print.
    """
    recommended_arm = comparison.recommended_arm
    # The arm's runs and its scores come in the same order: by seed.
    recommended_runs = [run for run in comparison.runs if run.arm == recommended_arm]
    scores = comparison.arm_scores[recommended_arm]
    best_score = max(scores)
    best_run = recommended_runs[scores.index(best_score)]
    run_manifest = json.loads(
        (best_run.folder / "manifest.json").read_text(encoding="utf-8")
    )
    benchmark_names = sorted(corpus_benchmarks(run_manifest))
    if benchmark_names:
        next_step = (
            "To score the run on the benchmarks: python -m embedding_model score"
            f" --model {best_run.folder} --benchmark {' '.join(benchmark_names)}"
        )
    else:
        next_step = (
            "Its corpus was checked against no benchmark, so score refuses every"
            " one: build the benchmarks, then build the corpus again."
        )
    metric_shown = metric_name(comparison.experiment["decision"]["metric"])
    return (
        f"Its best run is {best_run.folder.name}, at validation {metric_shown}"
        f" {best_score}. The comparison uses the validation pairs only. {next_step}"
    )


def print_summary(comparison, destination):
    """Print each arm's verdict, the recommendation, the next step, and where the folder went.

    Each name is padded to the longest arm's name, so that the numbers after
    the names line up.

    Args:
        comparison (Comparison): The comparison.
        destination (Path): The comparison's folder.
    """
    experiment, differences = comparison.experiment, comparison.differences
    metric_shown = metric_name(experiment["decision"]["metric"])
    base_row = comparison.arms[0]
    name_width = max(len(arm) for arm in comparison.arm_scores) + 2
    print(
        f"experiment {experiment['name']}: validation {metric_shown}, the base and"
        f" {len(differences)} arm{'s' if len(differences) > 1 else ''} at"
        f" {base_row['seeds']} seeds"
    )
    print(
        f"  {BASE:<{name_width}}{base_row['mean']:.{DECIMALS}f},"
        f" {CONFIDENCE:.0%} interval {base_row['low']:.{DECIMALS}f} to"
        f" {base_row['high']:.{DECIMALS}f}"
    )
    for row in differences:
        print(
            f"  {row['arm']:<{name_width}}{row['difference']:+.{DECIMALS}f},"
            f" {row['confidence']:.1%} interval {row['low']:+.{DECIMALS}f} to"
            f" {row['high']:+.{DECIMALS}f}, p {row['p']:.2g}: {row['verdict']}"
        )
    print(
        f"  recommended: {comparison.recommended_arm or 'none'}, as {comparison.reason}"
    )
    print(
        "  To build on it, write in the next experiment's file:"
        if len(comparison.base_lines) == 1
        else "  To build on one, write its line in the next experiment's file:"
    )
    for line in comparison.base_lines:
        print(f"    {line}")
    if comparison.recommended_arm is not None:
        print(f"  {best_run_line(comparison)}")
    print_destination(destination)


def compare(path, outputs):
    """Compare each arm of an experiment with the base, and recommend an arm to build on.

    It writes ``comparison/`` in the experiment's folder whole: ``arms.csv``,
    ``differences.csv``, the figure in ``figures/`` and ``manifest.json``.
    It prints each arm's verdict, the recommendation, and the ``base:`` line
    that builds the next experiment on it, and the command that scores the
    recommended arm's best run on the benchmarks that its corpus was checked
    against: score refuses any other. Then it prints the folder, and the
    figure and the tables by their full paths, as compare does.

    Args:
        path (Path): The experiment's file.
        outputs (Path): The outputs folder.

    Returns:
        int: 0.

    Raises:
        ValueError: If the file or the base cannot be used, the file differs
            from the copy that its first training kept, a base run of an
            earlier experiment has started but not finished, or a run is not
            done.
    """
    experiment, folder = checked_experiment(path, outputs)
    _, earlier_arms = base_config(experiment["base"], outputs)
    runs = experiment_runs(experiment, folder, earlier_arms)
    check_earlier_runs(runs, folder)
    arm_scores = seed_scores(experiment, runs, path)
    differences = difference_rows(experiment, arm_scores)
    recommended_arm, reason, base_lines = recommendation(experiment, differences)
    comparison = Comparison(
        experiment=experiment,
        folder=folder,
        runs=runs,
        arm_scores=arm_scores,
        arms=arm_rows(experiment, arm_scores),
        differences=differences,
        recommended_arm=recommended_arm,
        reason=reason,
        base_lines=base_lines,
    )
    destination = folder / COMPARISON
    write_comparison(comparison, destination)
    print_summary(comparison, destination)
    return 0
