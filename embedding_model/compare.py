"""Compare models that are already scored: metrics with intervals, paired tests and figures.

``compare`` reads the scores that ``score`` wrote for each model, on one or
more benchmarks in one text format. It writes one folder,
``outputs/comparisons/<name>/``:

- ``metrics.csv``: each model's metrics on each benchmark, with their 95%
  confidence intervals;
- ``differences.csv``: each other model minus the reference model, with the
  interval of the difference and a paired test;
- the figures, in ``figures/``, as EPS, PDF and PNG: accuracy@1, @5 and @10
  with MRR, accuracy at every k from 1 to 50, MRR by the model's size, and,
  with a reference, each model's difference from it;
- ``manifest.json``: the models, the scores read and the settings.

Each metric's mean comes from ``score``'s ``metrics.csv``, as its evaluator
computed it. Every interval and test needs each source variable's own value,
so it comes from the variable's rank in ``per_variable.csv``, as does the
curve of accuracy at every k. nDCG@10, the median rank and each model's
number of parameters come from ``metrics.csv`` too, as plain numbers.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from embedding_model import benchmarks
from embedding_model.folders import FOLDER_NAME, FOLDER_NAME_RULE, staged
from embedding_model.manifests import file_sha256
from embedding_model.manifests import write as write_manifest
from embedding_model.provenance import code_commit, library_versions
from embedding_model.scoring import (
    ACCURACY_AT_K,
    MRR_CUTOFF,
    NDCG_CUTOFF,
    VARIABLE_KEY,
    model_id,
    model_name,
    read_scores,
    run_record,
)
from embedding_model.texts import DEFAULT_FORMAT

# Where comparisons are written, under the outputs folder.
COMPARISONS = "comparisons"
# The subfolder of a comparison that holds its figures, beside its tables.
FIGURES = "figures"
# 95%, the usual confidence level.
CONFIDENCE = 0.95
# The metrics that get an interval and a paired test: each is a mean over the
# source variables of a value that each variable's rank gives.
TESTED_METRICS = ("mrr", f"mrr@{MRR_CUTOFF}", *[f"accuracy@{k}" for k in ACCURACY_AT_K])
MCNEMAR = "McNemar's exact test"
PAIRED_T = "paired t-test"
# Each tested metric's paired test: McNemar's exact test for accuracy@k, whose
# values are 1 or 0, and the paired t-test for MRR and MRR@10.
TESTS = {
    metric: MCNEMAR if metric.startswith("accuracy@") else PAIRED_T
    for metric in TESTED_METRICS
}
WILSON = "Wilson score interval"
NEWCOMBE = "Newcombe's interval for paired shares"
T_INTERVAL = "t interval"
# Each tested metric's intervals, for a model's metric and for a difference
# from the reference. accuracy@k is a share of 1s and 0s: Wilson's interval
# for one model, and Newcombe's for a paired difference. MRR and MRR@10 take
# values between 0 and 1: the t interval of the mean for one model, and of
# the variables' differences for a difference, the paired t-test's own
# interval.
INTERVALS = {
    metric: WILSON if metric.startswith("accuracy@") else T_INTERVAL
    for metric in TESTED_METRICS
}
DIFFERENCE_INTERVALS = {
    metric: NEWCOMBE if metric.startswith("accuracy@") else T_INTERVAL
    for metric in TESTED_METRICS
}
# The stars over a bracket, by the test's p.
STARS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
# The figures: the k of the accuracy bars, as the team's evaluation notebook
# draws them, and the k of the accuracy curve.
FIGURE_K = (1, 5, 10)
# The metrics of accuracy_and_mrr and differences_from_reference, by their
# names in the figures.
FIGURE_METRICS = {
    **{f"accuracy@{k}": f"Accuracy@{k}" for k in FIGURE_K},
    "mrr": "MRR",
}
CURVE_K = np.arange(1, 51)
# The x axis of accuracy_and_mrr and accuracy_by_k: what k means to a data
# curator.
K_LABEL = "k: suggestions a data curator reviews"
# One color per model, by its place in the order that the models are given,
# so that a model keeps its color in every figure: matplotlib's
# petroff10 color sequence, made for readers with a color vision deficiency
# (Petroff, 2021, https://doi.org/10.48550/arXiv.2107.02270).
PALETTE = "petroff10"
# Past 8 models, the bars grow too thin to read, and the brackets fill half of
# each panel.
MAX_MODELS = 8
# The note under the legend of accuracy_and_mrr, on what its brackets mean.
STARS_NOTE = (
    "Brackets: the reference against each model."
    " * p < 0.05, ** p < 0.01, *** p < 0.001;"
    " no bracket: no significant difference, p \u2265 0.05"
)
# The brackets of accuracy_and_mrr, on its 0-to-1 axis: the first bracket's
# gap above its group's highest interval, a bracket's rise, the step from one
# bracket to the next, and the room above the last one for its stars.
BRACKET_GAP = 0.03
BRACKET_RISE = 0.015
BRACKET_STEP = 0.06
STARS_ROOM = 0.06
# A marker per model in accuracy_by_k, mrr_by_size and
# differences_from_reference, beside its color, so that models
# are not told apart by color alone. matplotlib names only four line styles,
# so the lines of accuracy_by_k also take one of them, as a second cue.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
LINE_STYLES = ("-", "--", ":", "-.") * 2
# The markers of accuracy_by_k: matplotlib's markevery spaces them a tenth of
# the panel's diagonal apart. Each model's start is shifted, so that two lines
# that run together do not put their markers on one spot.
MARK_EVERY = 0.1
# Models of the same size sit side by side in mrr_by_size, this factor apart on
# its log axis, so that their points and intervals do not hide each other.
DODGE = 1.12
# The libraries that a comparison's manifest records, by their modules' names.
COMPARE_LIBRARIES = ("numpy", "pandas", "scipy", "matplotlib")
TEXT = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
# The figures are drawn 7.2 inches wide, about the width of a printed page,
# with text at 7 to 9 points, so they need no scaling. EPS and PDF are vector
# files, which stay sharp at any size, and PNG is a raster. Fonts are
# embedded as TrueType, so that a drawing program can edit the text.
PRINT_WIDTH = 7.2
FORMATS = ("png", "pdf", "eps")
PNG_DPI = 300
# The look, as matplotlib's rcParams: recessive axes, with a hairline grid
# behind the marks. An axis that matplotlib scales itself ends on a labelled
# tick, so the log axis of mrr_by_size ends on whole decades.
STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.edgecolor": AXIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": TEXT,
    "ytick.labelcolor": TEXT,
    "legend.frameon": False,
    "axes.autolimit_mode": "round_numbers",
}


def colors():
    """The colors of PALETTE, one per model, from matplotlib's color sequences.

    Every figure takes its colors from here, by the model's place in
    ``--model``. So a model keeps one color in all the figures.

    Returns:
        list[tuple[float, float, float]]: The sequence's colors, as RGB, in
            its order.
    """
    from matplotlib import color_sequences

    return color_sequences[PALETTE]


def label_of(model):
    """The name that a model takes in the tables and the legends.

    A run of ``train`` takes its run's name, a model folder its folder's name,
    and a Hugging Face model its id.

    Args:
        model (str): The model, as ``--model`` gives it.

    Returns:
        str: The model's name.
    """
    run = run_record(model)
    if run is not None:
        return run[1]["run"]
    return model_name(model)


def variable_values(ranks):
    """Each source variable's value of each tested metric, from its rank.

    A metric of TESTED_METRICS is the mean of these values over the
    variables: 1 / rank for MRR, the same with every rank past 10 counted as 0
    for MRR@10, and 1 or 0 for accuracy@k, as the variable's expected target
    is among the first k or not.

    Args:
        ranks (np.ndarray): Each source variable's rank.

    Returns:
        dict[str, np.ndarray]: Each tested metric's values, by its name.
    """
    ranks = np.asarray(ranks)
    values = {
        "mrr": 1 / ranks,
        f"mrr@{MRR_CUTOFF}": np.where(ranks <= MRR_CUTOFF, 1 / ranks, 0.0),
    }
    for k in ACCURACY_AT_K:
        values[f"accuracy@{k}"] = (ranks <= k).astype(float)
    return values


def t_interval(values):
    """The 95% t interval of the mean of ``values``.

    It is the interval of scipy's one-sample t-test. On each source
    variable's difference between two models, it is the paired t-test's
    interval. When every value is the same, such as an MRR of 1 for every
    variable, the values have no spread: the interval is that value.

    Args:
        values (np.ndarray): One value for each source variable, or each
            variable's difference between two models.

    Returns:
        tuple[float, float]: The interval's low and high ends.
    """
    values = np.asarray(values, dtype=float)
    if values.min() == values.max():
        return float(values[0]), float(values[0])
    interval = stats.ttest_1samp(values, 0.0).confidence_interval(CONFIDENCE)
    return float(interval.low), float(interval.high)


def wilson_interval(values):
    """The 95% Wilson score interval of a share: the mean of values of 1 and 0.

    scipy's ``binomtest`` gives it from the count of 1s. It keeps a width when
    every value is 1, or every value is 0.

    Args:
        values (np.ndarray): One value, 1 or 0, for each source variable.

    Returns:
        tuple[float, float]: The interval's low and high ends.
    """
    values = np.asarray(values)
    interval = stats.binomtest(int(values.sum()), len(values)).proportion_ci(
        confidence_level=CONFIDENCE, method="wilson"
    )
    return float(interval.low), float(interval.high)


def newcombe_interval(model_values, reference_values):
    """Newcombe's 95% interval for a paired difference of shares: model minus reference.

    This is Newcombe's method 10 (1998). It combines the two shares' Wilson
    intervals with the correlation of the two models' values. Two models that
    agree on many variables get a narrower interval than their two Wilson
    intervals would give apart. The correlation is Newcombe's phi, from the
    2 x 2 table of the two models' values, with his correction of a positive
    numerator by n / 2. scipy has no such interval. statsmodels has added it
    as ``confint_proportions_paired``, but has no release with it yet.

    Args:
        model_values (np.ndarray): The model's value for each variable: 1
            where an expected target is among its first k, and 0 otherwise.
        reference_values (np.ndarray): The reference model's value for each
            variable, in the same order.

    Returns:
        tuple[float, float]: The interval's low and high ends.
    """
    model_values = np.asarray(model_values)
    reference_values = np.asarray(reference_values)
    count = len(model_values)
    # The 2 x 2 table of the two models' values over the source variables:
    # Newcombe's a, b, c and d.
    both = int(((model_values == 1) & (reference_values == 1)).sum())
    only_model = int(((model_values == 1) & (reference_values == 0)).sum())
    only_reference = int(((model_values == 0) & (reference_values == 1)).sum())
    neither = count - both - only_model - only_reference
    # Newcombe's phi, the correlation of the two models' values. A positive
    # numerator, ad - bc, is lowered by count / 2, his correction. phi is 0
    # when a row or a column of the table is empty.
    phi_numerator = both * neither - only_model * only_reference
    if phi_numerator > 0:
        phi_numerator = max(phi_numerator - count / 2, 0)
    totals_product = (
        (both + only_model)
        * (only_reference + neither)
        * (both + only_reference)
        * (only_model + neither)
    )
    phi = phi_numerator / np.sqrt(totals_product) if totals_product else 0.0
    # Each model's share of 1s, and its Wilson interval.
    model_share, reference_share = model_values.mean(), reference_values.mean()
    model_low, model_high = wilson_interval(model_values)
    reference_low, reference_high = wilson_interval(reference_values)
    # How far each Wilson interval reaches from its share, down and up.
    model_down, model_up = model_share - model_low, model_high - model_share
    reference_down = reference_share - reference_low
    reference_up = reference_high - reference_share
    # The lower end takes the model's reach down and the reference's reach
    # up, since both make the difference smaller. The upper end takes the
    # other two. Each pair combines as the errors of two independent
    # estimates do, the square root of the sum of their squares, less a
    # term for phi. Two models that succeed on the same variables move
    # together, so their difference moves less.
    lower_distance = np.sqrt(
        model_down**2 - 2 * phi * model_down * reference_up + reference_up**2
    )
    upper_distance = np.sqrt(
        model_up**2 - 2 * phi * model_up * reference_down + reference_down**2
    )
    difference = model_share - reference_share
    return float(difference - lower_distance), float(difference + upper_distance)


def paired_test(metric, model_values, reference_values):
    """The paired test of two models' values of a metric, on the same variables.

    For accuracy@k, each variable's value is 1 or 0. McNemar's exact test
    counts the variables where the two models differ: under no difference,
    each is a coin flip, so p is the binomial test's. For MRR and MRR@10,
    scipy's paired t-test compares the two models' values on each variable. When
    every variable differs by the same amount, the differences have no
    spread: p is 1 when that amount is 0, and 0 otherwise, the limit that
    scipy reaches with a warning.

    Args:
        metric (str): A metric of TESTED_METRICS.
        model_values (np.ndarray): A model's value for each variable.
        reference_values (np.ndarray): The reference model's value for each
            variable, in the same order.

    Returns:
        tuple[str, float]: The test's name and its two-sided p.
    """
    if TESTS[metric] == MCNEMAR:
        only_model = int(((model_values == 1) & (reference_values == 0)).sum())
        only_reference = int(((model_values == 0) & (reference_values == 1)).sum())
        if only_model + only_reference == 0:
            return MCNEMAR, 1.0
        binomial_test = stats.binomtest(only_model, only_model + only_reference, 0.5)
        return MCNEMAR, float(binomial_test.pvalue)
    differences = np.asarray(model_values, dtype=float) - np.asarray(
        reference_values, dtype=float
    )
    if differences.min() == differences.max():
        return PAIRED_T, 1.0 if differences[0] == 0 else 0.0
    return PAIRED_T, float(stats.ttest_rel(model_values, reference_values).pvalue)


def stars(p):
    """The stars that mark a p below 0.05, 0.01 or 0.001.

    Args:
        p (float): A paired test's p.

    Returns:
        str: ``***``, ``**`` or ``*``, by STARS, or "" for a p of 0.05 or
            more.
    """
    for limit, mark in STARS:
        if p < limit:
            return mark
    return ""


def metrics_row(label, benchmark, built_version, text_format, row, values):
    """One model's metrics on one benchmark, each tested one with its interval.

    Each mean is the one that ``score`` reported. Its interval comes from the
    source variables' values, by the metric's method in INTERVALS.

    Args:
        label (str): The model's name, from label_of.
        benchmark (str): The benchmark.
        built_version (str): The benchmark's version.
        text_format (str): The text format of the scores.
        row (pd.Series): The model's row of ``metrics.csv``.
        values (dict[str, np.ndarray]): Each tested metric's values, from
            variable_values.

    Returns:
        dict: The row of the comparison's ``metrics.csv``.
    """
    comparison_row = {
        "model": label,
        "benchmark": benchmark,
        "benchmark_version": built_version,
        "format": text_format,
        "n": len(values["mrr"]),
        "parameters": int(row["parameters"]),
    }
    for metric in TESTED_METRICS:
        if INTERVALS[metric] == WILSON:
            low, high = wilson_interval(values[metric])
        else:
            low, high = t_interval(values[metric])
        comparison_row[metric] = float(row[metric])
        comparison_row[f"{metric}_low"] = round(low, 4)
        comparison_row[f"{metric}_high"] = round(high, 4)
    comparison_row[f"ndcg@{NDCG_CUTOFF}"] = float(row[f"ndcg@{NDCG_CUTOFF}"])
    comparison_row["median_rank"] = float(row["median_rank"])
    return comparison_row


def difference_rows(benchmark, reference, others, values):
    """Each other model minus the reference model, on each tested metric.

    Each difference's interval comes by the metric's method in
    DIFFERENCE_INTERVALS.

    Args:
        benchmark (str): The benchmark.
        reference (str): The reference model's name.
        others (list[str]): The other models' names.
        values (dict[str, dict[str, np.ndarray]]): Each model's values of each
            tested metric, by the model's name.

    Returns:
        list[dict]: One row per other model and metric: the difference, its
            95% confidence interval, the test and its p.
    """
    rows = []
    for other in others:
        for metric in TESTED_METRICS:
            model_values = values[other][metric]
            reference_values = values[reference][metric]
            if DIFFERENCE_INTERVALS[metric] == NEWCOMBE:
                low, high = newcombe_interval(model_values, reference_values)
            else:
                low, high = t_interval(model_values - reference_values)
            test, p = paired_test(metric, model_values, reference_values)
            rows.append(
                {
                    "benchmark": benchmark,
                    "metric": metric,
                    "reference": reference,
                    "model": other,
                    "difference": round(
                        float(np.mean(model_values - reference_values)), 4
                    ),
                    "low": round(low, 4),
                    "high": round(high, 4),
                    "test": test,
                    "p": p,
                }
            )
    return rows


def bracket(axes, x_from, x_to, height, mark):
    """Draw a bracket on ``axes`` from one bar to another, with its stars above it.

    It returns nothing: the line and the stars are drawn onto ``axes``. The
    line has the gid ``bracket``, so that a test can find it.

    Args:
        axes (Axes): The panel to draw on.
        x_from (float): The middle of the first bar.
        x_to (float): The middle of the other bar.
        height (float): Where the bracket's ends start, above both bars.
        mark (str): The stars, from stars.
    """
    rise = BRACKET_RISE
    axes.plot(
        [x_from, x_from, x_to, x_to],
        [height, height + rise, height + rise, height],
        color=TEXT,
        linewidth=0.8,
        gid="bracket",
    )
    axes.text((x_from + x_to) / 2, height + rise, mark, ha="center", fontsize=7)


def bars(axes, groups, legend_labels, heights, errors, bar_colors):
    """Draw grouped bars: one group per item of ``groups``, one bar per model.

    The bars of a group sit side by side, each model at its own offset from
    the group's middle. The offsets and the interval tops come back, so that
    the caller can draw each comparison bracket over its two bars.

    Args:
        axes (Axes): Where the bars go.
        groups (list[str]): The groups' labels, such as the k of accuracy@k.
        legend_labels (list[str]): Each model's text in the legend, in the
            bars' order: its name, such as ``abcd-run_s0 (reference)`` for
            the reference.
        heights (list[list[float]]): Each model's bar heights, by group.
        errors (list[list[tuple[float, float]]]): Each bar's interval.
        bar_colors (list): Each model's color.

    Returns:
        tuple[np.ndarray, list[float]]: Each model's offset from the middle of
            a group, and the top of each group's highest interval.
    """
    count = len(heights)
    width = 0.8 / count
    offsets = (np.arange(count) - (count - 1) / 2) * width
    positions = np.arange(len(groups))
    axes.set_ylim(0, 1)
    axes.set_yticks(np.linspace(0, 1, 6))
    for index, legend_label in enumerate(legend_labels):
        low = [
            height - interval_[0]
            for height, interval_ in zip(heights[index], errors[index])
        ]
        high = [
            interval_[1] - height
            for height, interval_ in zip(heights[index], errors[index])
        ]
        axes.bar(
            positions + offsets[index],
            heights[index],
            width,
            color=bar_colors[index],
            edgecolor=SURFACE,
            linewidth=0.8,
            yerr=[low, high],
            capsize=2,
            error_kw={"linewidth": 0.8, "ecolor": TEXT},
            label=legend_label,
        )
    axes.set_xticks(positions, groups)
    axes.set_xlim(-0.5, len(groups) - 0.5)
    tops = [
        max(model_errors[group][1] for model_errors in errors)
        for group in range(len(groups))
    ]
    return offsets, tops


def benchmark_title(benchmark, benchmark_result):
    """A benchmark's title in a figure: its name, its version and its source variables.

    Args:
        benchmark (str): The benchmark's name, such as ``topmed``.
        benchmark_result (dict): The benchmark's results, with its version and
            its number of source variables.

    Returns:
        str: The title, such as ``Benchmark: TOPMed v2 (900 source variables)``.
    """
    name = benchmarks.TITLES[benchmark]
    return (
        f"Benchmark: {name} v{benchmark_result['version']}"
        f" ({benchmark_result['n']:,} source variables)"
    )


def legend(figure, axes, count, note=None):
    """Draw the figure's legend below its panels: one entry per model, three to a row.

    Args:
        figure (Figure): The figure.
        axes (Axes): A panel that holds every entry.
        count (int): The number of entries.
        note (str | None): A line under the entries, such as STARS_NOTE.
    """
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(count, 3),
        title=note,
        title_fontsize=7,
        alignment="center",
    )


def drawn(models, reference):
    """The models in the order that the figures draw them, each with its look.

    The reference comes first. Each model keeps the color, the line style and
    the marker of its place in ``models``, so it looks the same in every
    figure, whichever model is the reference.

    Args:
        models (list[str]): The models' names, in the order given.
        reference (str | None): The reference model's name.

    Returns:
        list[tuple[str, str, int]]: Each model's name, its label in the
            legends, with " (reference)" for the reference, and its place in
            ``models``.
    """
    if reference is None:
        order = models
    else:
        order = [reference, *(model for model in models if model != reference)]
    return [
        (
            model,
            f"{model} (reference)" if model == reference else model,
            models.index(model),
        )
        for model in order
    ]


def overview_figure(benchmark_results, models, reference, differences):
    """``accuracy_and_mrr``: accuracy@1, @5 and @10 and MRR as bars, per benchmark.

    Each benchmark has a row, under its title: accuracy@k on the left, MRR on
    the right, on its own axis. Each bar has its 95% confidence interval. The
    reference comes first in each group. A bracket joins the reference to
    each model that differs from it, with p below 0.05: a group's brackets
    start just above its highest interval. A note under the legend says what
    the stars mean. Draw it under STYLE. The bars carry no numbers:
    metrics.csv holds every value.

    Args:
        benchmark_results (dict): Each benchmark's results: its version, its
            number of source variables, and each model's metrics row, values
            and ranks.
        models (list[str]): The models' names, in order.
        reference (str | None): The reference model's name.
        differences (list[dict]): The rows of difference_rows.

    Returns:
        Figure: The figure.
    """
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(PRINT_WIDTH, 3.4 * len(benchmark_results)), layout="constrained"
    )
    figure.suptitle("Accuracy@k and MRR of each model")
    panels = figure.subfigures(
        len(benchmark_results), 1, hspace=0.08, squeeze=False
    ).ravel()
    tested = {
        (row["benchmark"], row["metric"], row["model"]): row["p"]
        for row in differences
        if stars(row["p"])
    }
    order = drawn(models, reference)
    legend_labels = [legend_label for _, legend_label, _ in order]
    bar_colors = [colors()[place] for _, _, place in order]
    legend_axes = None
    for panel, (benchmark, result) in zip(panels, benchmark_results.items()):
        panel.suptitle(benchmark_title(benchmark, result))
        accuracy, mrr = panel.subplots(1, 2, width_ratios=[3, 1])
        legend_axes = legend_axes or accuracy
        metrics = result["metrics"]
        for axes, panel_metrics in (
            (accuracy, [f"accuracy@{k}" for k in FIGURE_K]),
            (mrr, ["mrr"]),
        ):
            heights = [
                [metrics[model][metric] for metric in panel_metrics]
                for model, _, _ in order
            ]
            errors = [
                [
                    (metrics[model][f"{metric}_low"], metrics[model][f"{metric}_high"])
                    for metric in panel_metrics
                ]
                for model, _, _ in order
            ]
            groups = [str(k) for k in FIGURE_K] if axes is accuracy else ["MRR"]
            offsets, tops = bars(
                axes, groups, legend_labels, heights, errors, bar_colors
            )
            top = 1.0
            for group, metric in enumerate(panel_metrics):
                height = tops[group] + BRACKET_GAP
                for index, (model, _, _) in enumerate(order):
                    if (benchmark, metric, model) not in tested:
                        continue
                    bracket(
                        axes,
                        group + offsets[0],
                        group + offsets[index],
                        height,
                        stars(tested[(benchmark, metric, model)]),
                    )
                    top = max(top, height + STARS_ROOM)
                    height += BRACKET_STEP
            axes.set_ylim(0, top)
        accuracy.set_xlabel(K_LABEL)
        accuracy.set_ylabel("Accuracy@k")
        mrr.set_xticks([])
        mrr.set_ylabel("MRR")
    legend(figure, legend_axes, len(models), STARS_NOTE if reference else None)
    return figure


def curve_figure(benchmark_results, models, reference):
    """``accuracy_by_k``: accuracy at every k from 1 to 50, a line per model.

    Accuracy@k at every k is the empirical distribution function of the
    variables' ranks, which scipy's ``ecdf`` gives. Draw it under STYLE.

    Args:
        benchmark_results (dict): Each benchmark's results, as for
            overview_figure.
        models (list[str]): The models' names, in order.
        reference (str | None): The reference model's name.

    Returns:
        Figure: The figure.
    """
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(PRINT_WIDTH, 2.6 * len(benchmark_results)), layout="constrained"
    )
    figure.get_layout_engine().set(hspace=0.08)
    figure.suptitle(f"Accuracy@k for every k from 1 to {CURVE_K[-1]}")
    rows = figure.subplots(len(benchmark_results), 1, squeeze=False)
    for (axes,), (benchmark, result) in zip(rows, benchmark_results.items()):
        for model, legend_label, place in drawn(models, reference):
            axes.plot(
                CURVE_K,
                stats.ecdf(result["ranks"][model]).cdf.evaluate(CURVE_K),
                color=colors()[place],
                linestyle=LINE_STYLES[place],
                linewidth=1.4,
                marker=MARKERS[place],
                markersize=4,
                markeredgecolor=SURFACE,
                markeredgewidth=0.5,
                markevery=(MARK_EVERY * place / len(models), MARK_EVERY),
                label=legend_label,
            )
        axes.set_xlim(1, CURVE_K[-1])
        axes.set_xticks([1, 5, 10, 20, 30, 40, 50])
        axes.set_ylim(0, 1)
        axes.set_xlabel(K_LABEL)
        axes.set_ylabel("Accuracy@k")
        axes.set_title(benchmark_title(benchmark, result))
    legend(figure, rows[0][0], len(models))
    return figure


def size_figure(benchmark_results, models, reference):
    """``mrr_by_size``: each model's MRR and interval, by its number of parameters.

    Each benchmark has a panel. The number of parameters is on a log scale,
    so that a static model and a transformer fit on one axis. Draw it under
    STYLE.

    Args:
        benchmark_results (dict): Each benchmark's results, as for
            overview_figure.
        models (list[str]): The models' names, in order.
        reference (str | None): The reference model's name.

    Returns:
        Figure: The figure.
    """
    from matplotlib.figure import Figure
    from matplotlib.ticker import NullFormatter, StrMethodFormatter

    figure = Figure(
        figsize=(PRINT_WIDTH, 2.6 * len(benchmark_results)), layout="constrained"
    )
    figure.get_layout_engine().set(hspace=0.08)
    figure.suptitle("MRR by model size")
    rows = figure.subplots(len(benchmark_results), 1, squeeze=False)
    for (axes,), (benchmark, result) in zip(rows, benchmark_results.items()):
        metrics = result["metrics"]
        order = drawn(models, reference)
        millions = [metrics[model]["parameters"] / 1e6 for model, _, _ in order]
        for index, (model, legend_label, place) in enumerate(order):
            same = [
                other for other, size in enumerate(millions) if size == millions[index]
            ]
            position = millions[index] * DODGE ** (
                same.index(index) - (len(same) - 1) / 2
            )
            mrr, low, high = (
                metrics[model][key] for key in ("mrr", "mrr_low", "mrr_high")
            )
            # The marker goes over its interval, which takes the model's color.
            axes.errorbar(
                position,
                mrr,
                yerr=[[mrr - low], [high - mrr]],
                fmt="none",
                ecolor=colors()[place],
                elinewidth=1.2,
                capsize=3,
                zorder=2,
            )
            axes.plot(
                position,
                mrr,
                MARKERS[place],
                color=colors()[place],
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=0.6,
                zorder=3,
                label=legend_label,
            )
        axes.set_xscale("log")
        axes.xaxis.set_major_formatter(StrMethodFormatter("{x:,g}"))
        axes.xaxis.set_minor_formatter(NullFormatter())
        axes.set_ylim(0, 1)
        axes.set_xlabel("Parameters, in millions (log scale, base 10)")
        axes.set_ylabel("MRR")
        axes.set_title(benchmark_title(benchmark, result))
    legend(figure, rows[0][0], len(models))
    return figure


def differences_figure(benchmark_results, models, reference, differences):
    """``differences_from_reference``: each model minus the reference, per benchmark.

    Each benchmark has a panel, and each metric of accuracy_and_mrr a row. Each model
    is a point at its difference from the reference, which is the line at 0.
    Right of 0, the model does better than the reference. The panels share
    one x axis, so their 0s line up, and a difference has the same length in
    each. Draw it under STYLE.

    Args:
        benchmark_results (dict): Each benchmark's results, as for
            overview_figure.
        models (list[str]): The models' names, in order.
        reference (str): The reference model's name.
        differences (list[dict]): The rows of difference_rows.

    Returns:
        Figure: The figure.
    """
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(
        figsize=(PRINT_WIDTH, 2.7 * len(benchmark_results)), layout="constrained"
    )
    figure.get_layout_engine().set(hspace=0.1)
    figure.suptitle(f"Each model against the reference, {reference}")
    rows = figure.subplots(len(benchmark_results), 1, squeeze=False)
    others = [
        drawn_model
        for drawn_model in drawn(models, reference)
        if drawn_model[0] != reference
    ]
    differences_by_key = {
        (row["benchmark"], row["metric"], row["model"]): row for row in differences
    }
    spread = 0.6 / len(others)
    drawn_rows = [row for row in differences if row["metric"] in FIGURE_METRICS]
    lowest = min([0.0] + [row["low"] for row in drawn_rows])
    highest = max([0.0] + [row["high"] for row in drawn_rows])
    margin = 0.08 * (highest - lowest) or 0.01
    for (axes,), (benchmark, result) in zip(rows, benchmark_results.items()):
        for line, metric in enumerate(FIGURE_METRICS):
            for index, (model, legend_label, place) in enumerate(others):
                row = differences_by_key[(benchmark, metric, model)]
                height = line + (index - (len(others) - 1) / 2) * spread
                axes.errorbar(
                    row["difference"],
                    height,
                    xerr=[
                        [row["difference"] - row["low"]],
                        [row["high"] - row["difference"]],
                    ],
                    fmt="none",
                    ecolor=colors()[place],
                    elinewidth=1.2,
                    capsize=2.5,
                    zorder=2,
                )
                axes.plot(
                    row["difference"],
                    height,
                    MARKERS[place],
                    color=colors()[place],
                    markersize=6,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.6,
                    zorder=3,
                    label=legend_label if line == 0 else None,
                )
        axes.axvline(0, color=TEXT, linewidth=0.8, zorder=1)
        axes.set_xlim(lowest - margin, highest + margin)
        # matplotlib picks the ticks, by steps of 1, 2 or 5 times a power of ten.
        axes.xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
        axes.set_yticks(range(len(FIGURE_METRICS)), list(FIGURE_METRICS.values()))
        axes.set_ylim(len(FIGURE_METRICS) - 0.5, -0.5)
        axes.grid(axis="y", visible=False)
        axes.grid(axis="x", visible=True)
        axes.set_xlabel("Model minus the reference, with 95% confidence intervals")
        for side, text, align in (
            (0, "\u2190 worse than the reference", "left"),
            (1, "better than the reference \u2192", "right"),
        ):
            axes.annotate(
                text,
                (side, 0),
                xycoords="axes fraction",
                xytext=(0, -24),
                textcoords="offset points",
                ha=align,
                va="top",
                fontsize=7,
                color=MUTED,
            )
        axes.set_title(benchmark_title(benchmark, result))
    legend(figure, rows[0][0], len(others))
    return figure


def save_figure(figure, folder, stem):
    """Write a figure in each of FORMATS. Call it under STYLE, which embeds the fonts.

    Args:
        figure (Figure): The figure.
        folder (Path): Where it goes.
        stem (str): The files' name, without its suffix.
    """
    for suffix in FORMATS:
        figure.savefig(folder / f"{stem}.{suffix}", dpi=PNG_DPI)


@dataclass
class Comparison:
    """What a comparison compares, and what it found on each benchmark.

    Attributes:
        name (str): The comparison's name, and its folder's.
        text_format (str): The text format of the scores.
        models (dict[str, str]): Each model, as ``--model`` gives it, by its
            name in the tables, in the order given.
        reference (str | None): The reference model's name, or None.
        results (dict[str, dict]): Each benchmark's results, from
            benchmark_result, by the benchmark's name.
    """

    name: str
    text_format: str
    models: dict
    reference: str | None
    results: dict = field(default_factory=dict)

    @property
    def labels(self):
        """The models' names in the tables, in the order given."""
        return list(self.models)

    @property
    def differences(self):
        """The rows of differences.csv, from every benchmark."""
        return [
            row for result in self.results.values() for row in result["differences"]
        ]


def checked_models(name, models, reference):
    """Check the comparison's name and models, and name each model.

    Args:
        name (str): The comparison's name, a FOLDER_NAME.
        models (list[str]): The models, as ``score`` was given them.
        reference (str | None): One of ``models``, which may name it another
            way, such as by a relative path, or None.

    Returns:
        tuple[dict[str, str], str | None]: Each model by its name in the
            tables, in the order given, and the reference's name, or None.

    Raises:
        ValueError: If the name is not a FOLDER_NAME, there are more than
            MAX_MODELS models, a model is named twice, two models have one
            name, or the reference is not one of the models or has no other
            model to be measured against.
    """
    if not FOLDER_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a comparison name: use {FOLDER_NAME_RULE}")
    if len(models) > MAX_MODELS:
        raise ValueError(
            f"{len(models)} models: a comparison holds at most {MAX_MODELS}, so"
            " that its figures stay readable"
        )
    ids = [model_id(model) for model in models]
    for index, key in enumerate(ids):
        if key in ids[:index]:
            raise ValueError(
                f"--model names one model twice: {models[ids.index(key)]} and"
                f" {models[index]}. Name each model once"
            )
    labels = [label_of(model) for model in models]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "two models have the same name: "
            + ", ".join(sorted({label for label in labels if labels.count(label) > 1}))
        )
    if reference is None:
        return dict(zip(labels, models)), None
    if model_id(reference) not in ids:
        raise ValueError(
            f"the reference {reference} is not one of the models: give it with"
            " --model too"
        )
    if len(models) < 2:
        raise ValueError(
            f"the reference {reference} has no other model to be measured"
            " against it: give two models or more"
        )
    return dict(zip(labels, models)), labels[ids.index(model_id(reference))]


def benchmark_result(comparison, benchmark, inputs, outputs):
    """Read every model's scores on one benchmark, and compute what they show.

    Args:
        comparison (Comparison): The comparison.
        benchmark (str): The benchmark.
        inputs (Path): The inputs folder that the benchmark was built into.
        outputs (Path): The outputs folder.

    Returns:
        dict: The benchmark's ``version`` and ``n``, its source variables.
            By each model's name: its row of ``metrics``, its ``values`` and
            its ``ranks``. Then the scores read, as the manifest's
            ``sources``, and the rows of ``differences``, with a reference.

    Raises:
        ValueError: If a model's scores cannot be read, as read_scores says,
            or list other source variables than the first model's.
        FileNotFoundError: If the benchmark has not been built.
    """
    _, _, manifest = benchmarks.read_benchmark(inputs, benchmark)
    version = manifest["version"]
    result = {
        "version": version,
        "metrics": {},
        "values": {},
        "ranks": {},
        "sources": [],
    }
    variables = None
    for label, model in comparison.models.items():
        row, ranks, folder = read_scores(
            model, benchmark, comparison.text_format, outputs, manifest
        )
        if variables is None:
            variables = ranks[list(VARIABLE_KEY)]
        elif not ranks[list(VARIABLE_KEY)].equals(variables):
            raise ValueError(
                f"{model}'s scores on {benchmark} list other source variables,"
                f" or list them in another order, than {comparison.labels[0]}'s,"
                " though both were scored on this build of it. Score both again."
            )
        values = variable_values(ranks["rank"].to_numpy())
        result["metrics"][label] = metrics_row(
            label, benchmark, version, comparison.text_format, row, values
        )
        result["values"][label] = values
        result["ranks"][label] = ranks["rank"].to_numpy()
        result["sources"].append(
            {
                "role": "scores",
                "model": model,
                "name": label,
                "benchmark": benchmark,
                "location": str(folder),
                "files": {
                    file: {"sha256": file_sha256(folder / file)}
                    for file in ("metrics.csv", "per_variable.csv")
                },
            }
        )
    result["n"] = len(variables)
    result["differences"] = []
    if comparison.reference is not None:
        others = [label for label in comparison.labels if label != comparison.reference]
        result["differences"] = difference_rows(
            benchmark, comparison.reference, others, result["values"]
        )
    return result


def write_comparison(comparison, destination):
    """Write the comparison's folder whole: its tables, figures and manifest.

    ``differences.csv`` and ``differences_from_reference`` come only with a
    reference.

    Args:
        comparison (Comparison): The comparison, with every benchmark's
            results.
        destination (Path): The comparison's folder.
    """
    from matplotlib import rc_context

    results, labels = comparison.results, comparison.labels
    reference, differences = comparison.reference, comparison.differences
    with staged(destination) as built, rc_context(STYLE):
        pd.DataFrame(
            [
                result["metrics"][label]
                for result in results.values()
                for label in labels
            ]
        ).to_csv(built / "metrics.csv", index=False)
        if reference is not None:
            pd.DataFrame(differences).to_csv(built / "differences.csv", index=False)
        (built / FIGURES).mkdir()
        figures = {
            "accuracy_and_mrr": overview_figure(
                results, labels, reference, differences
            ),
            "accuracy_by_k": curve_figure(results, labels, reference),
            "mrr_by_size": size_figure(results, labels, reference),
        }
        if reference is not None:
            figures["differences_from_reference"] = differences_figure(
                results, labels, reference, differences
            )
        for stem, figure in figures.items():
            save_figure(figure, built / FIGURES, stem)
        write_manifest(
            built / "manifest.json",
            {
                "comparison": comparison.name,
                "built": date.today().isoformat(),
                "sources": [
                    source
                    for result in results.values()
                    for source in result["sources"]
                ],
                "settings": {
                    "format": comparison.text_format,
                    "confidence": CONFIDENCE,
                    "intervals": INTERVALS,
                    "difference_intervals": DIFFERENCE_INTERVALS,
                    "tests": TESTS,
                },
                "models": labels,
                "reference": reference,
                "benchmarks": {
                    benchmark: result["version"]
                    for benchmark, result in results.items()
                },
                "ai_harmonization": code_commit(),
                "libraries": library_versions(COMPARE_LIBRARIES),
            },
        )


def print_destination(destination):
    """Print where a comparison went: its folder, then each figure and table by its full path.

    One path per line, so that a terminal that links paths, such as VS
    Code's, opens each with a click. Each figure is a PNG, with its PDF and
    EPS beside it.

    Args:
        destination (Path): The comparison's folder.
    """
    print(f"  -> {destination}")
    for path in sorted((destination / FIGURES).glob("*.png")):
        print(f"     figure: {path}")
    for path in sorted(destination.glob("*.csv")):
        print(f"     table: {path}")


def print_summary(comparison, destination):
    """Print each model's MRR and accuracy@1 on each benchmark, and where the folder went.

    Args:
        comparison (Comparison): The comparison, with every benchmark's
            results.
        destination (Path): The comparison's folder.
    """
    for benchmark, result in comparison.results.items():
        print(
            f"{benchmark} version {result['version']}, {result['n']:,} source variables:"
        )
        for label in comparison.labels:
            metrics = result["metrics"][label]
            print(
                f"  {label}: MRR {metrics['mrr']} ({metrics['mrr_low']} to"
                f" {metrics['mrr_high']}), accuracy@1 {metrics['accuracy@1']}"
            )
    if comparison.reference is None and len(comparison.labels) > 1:
        print(
            "  no --reference: no paired tests and no brackets. To test the"
            " differences, name one model with --reference."
        )
    print_destination(destination)


def compare(
    name,
    models,
    benchmark_names,
    inputs,
    outputs,
    text_format=DEFAULT_FORMAT,
    reference=None,
):
    """Compare scored models and write the comparison's folder.

    A benchmark named twice is compared once, as ``score`` scores it once.

    Args:
        name (str): The comparison's name, a FOLDER_NAME.
        models (list[str]): The models, as ``score`` was given them.
        benchmark_names (list[str]): The benchmarks, such as ``["topmed"]``.
        inputs (Path): The inputs folder that the benchmarks were built into.
        outputs (Path): The outputs folder.
        text_format (str): The text format of the scores.
        reference (str | None): One of ``models``, which each other model is
            measured against by paired tests. It may name the model another
            way, such as by a relative path. None writes no differences.

    Returns:
        Path: The comparison's folder.

    Raises:
        ValueError: If the models or the name are refused, as checked_models
            says, or a model's scores cannot be read or list other variables,
            as benchmark_result says.
        FileNotFoundError: If a benchmark has not been built.
    """
    named_models, reference_name = checked_models(name, models, reference)
    comparison = Comparison(name, text_format, named_models, reference_name)
    for benchmark in dict.fromkeys(benchmark_names):
        comparison.results[benchmark] = benchmark_result(
            comparison, benchmark, inputs, outputs
        )
    destination = Path(outputs) / COMPARISONS / name
    write_comparison(comparison, destination)
    print_summary(comparison, destination)
    return destination
