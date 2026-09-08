"""
Presentation layer for the harmonization outputs: palette, similarity bands,
and the HTML the review widget and the quality summary render.

Everything visual lives here so the notebooks carry no styling and
``review.py`` carries no markup — that module is the session state machine,
this one decides how it looks. It also means the widget and the summary table
agree on what a "strong" similarity is rather than each hardcoding a palette.

Styles are inline rather than CSS classes in a ``<style>`` block on purpose.
The widget renders inside an ipywidgets ``HTML`` container, and notebook
front ends differ in whether they preserve a stylesheet there; inline
attributes always survive.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only needed for type checking
    import pandas as pd


# ── Palette ───────────────────────────────────────────────────────────────────

ACCENT = "#4a90d9"  # progress bar, similarity bars, panel rule
ACCEPTED = "#28a745"  # accepted decision
SKIPPED = "#fd7e14"  # skipped decision
PANEL_BG = "#f8f9fa"  # current-variable panel
TRACK_BG = "#e9ecef"  # unfilled progress track
HEADER_BG = "#f0f4fa"  # table header
TEXT_MUTED = "#555"
TEXT_FAINT = "#777"
TEXT_GHOST = "#aaa"
TEXT_BODY = "#444"  # variable description body text

# ── Similarity bands ──────────────────────────────────────────────────────────
#
# One definition of "strong", used by everything that colours or counts a
# similarity score: the review widget's candidate table, the quality summary's
# gradient, and the strong-match percentages in
# ``dbgap.summarize_rank1_similarity``. Bootstrap's alert colours, so the bands
# read the same way in any Jupyter theme.

STRONG_SIMILARITY = 0.75
VERY_STRONG_SIMILARITY = 0.80
MODERATE_SIMILARITY = 0.5

STRONG_CELL_CSS = "background-color:#d4edda;color:#155724;font-weight:bold"
MODERATE_CELL_CSS = "background-color:#fff3cd;color:#856404"
WEAK_CELL_CSS = "background-color:#f8d7da;color:#721c24"

# The summary gradient tops out where "strong" begins, so a study whose mean
# rank-1 similarity is strong renders fully green. Starting at MODERATE keeps
# the observed spread of study means (roughly 0.50-0.58) discriminable rather
# than squashing every study into one shade.
SUMMARY_GRADIENT_VMIN = MODERATE_SIMILARITY
SUMMARY_GRADIENT_VMAX = STRONG_SIMILARITY
SUMMARY_GRADIENT_CMAP = "RdYlGn"


def similarity_cell_css(value) -> str:
    """Return the CSS for one similarity cell, banded by strength.

    Args:
        value: A similarity score, normally 0.0-1.0.

    Returns:
        str: A CSS declaration string for pandas ``Styler.map``.
    """
    if value >= STRONG_SIMILARITY:
        return STRONG_CELL_CSS
    if value >= MODERATE_SIMILARITY:
        return MODERATE_CELL_CSS
    return WEAK_CELL_CSS


# ── Table styling ─────────────────────────────────────────────────────────────

SUMMARY_TABLE_STYLES = [
    {
        "selector": "caption",
        "props": [
            ("font-size", "14px"),
            ("font-weight", "bold"),
            ("text-align", "left"),
            ("padding-bottom", "8px"),
        ],
    },
    {
        "selector": "th",
        "props": [
            ("background-color", HEADER_BG),
            ("font-size", "12px"),
            ("text-align", "center"),
        ],
    },
    {
        "selector": "td",
        "props": [("font-size", "12px"), ("padding", "4px 10px")],
    },
]

SUMMARY_CAPTION = "Mapping Quality by Study — rank-1 similarity statistics"
_GRADIENT_COLUMNS = ["Mean sim (rank 1)", "Median sim"]


def style_mapping_quality_summary(
    summary: "pd.DataFrame", caption: str = SUMMARY_CAPTION
):
    """Style the per-study quality summary for display in a notebook.

    Args:
        summary: One row per study, as produced by grouping preliminary mappings
            by ``study_id`` and applying ``summarize_rank1_similarity``.
        caption: Table caption.

    Returns:
        pandas.io.formats.style.Styler: Ready to hand to ``IPython.display.display``.
    """
    gradient_columns = [c for c in _GRADIENT_COLUMNS if c in summary.columns]
    styler = summary.style
    if gradient_columns:
        styler = styler.background_gradient(
            subset=gradient_columns,
            cmap=SUMMARY_GRADIENT_CMAP,
            vmin=SUMMARY_GRADIENT_VMIN,
            vmax=SUMMARY_GRADIENT_VMAX,
        )
    return (
        styler.set_caption(caption)
        .set_table_styles(SUMMARY_TABLE_STYLES)
        .format({column: "{:.3f}" for column in gradient_columns})
    )


# ── Review widget: button dimensions ─────────────────────────────────────────
#
# These are ipywidgets Layout values, not CSS, so they cannot live in a
# stylesheet. Named here anyway so every dimension the widget uses is in one
# place.

BACK_BUTTON_WIDTH = "80px"
NEXT_BUTTON_WIDTH = "50px"
SKIP_BUTTON_WIDTH = "100px"
ACCEPT_FIRST_BUTTON_WIDTH = "70px"
ACCEPT_RANK_BUTTON_WIDTH = "50px"
CONTROLS_MARGIN = "4px 0"
CONTROLS_GAP = "4px"

# ipywidgets button_style keywords for each decision state.
ACCEPTED_BUTTON_STYLE = "success"
SKIPPED_BUTTON_STYLE = "danger"
SKIP_BUTTON_STYLE = "warning"


# ── Review widget: HTML ──────────────────────────────────────────────────────


def progress_html(
    position: int,
    total: int,
    accepted: int,
    skipped: int,
    auto_saving: bool = False,
) -> str:
    """Render the counters and progress bar above the current variable.

    Args:
        position: 1-based index of the variable on screen.
        total: Number of variables in the session.
        accepted: How many variables have an accepted candidate.
        skipped: How many variables are marked unmappable.
        auto_saving: Whether to note that every click is being written to disk.

    Returns:
        str: HTML for an ipywidgets ``HTML`` widget.
    """
    reviewed = accepted + skipped
    remaining = total - reviewed
    pct = reviewed / total * 100 if total else 0.0
    bar = (
        f'<div style="height:4px;background:{TRACK_BG};border-radius:2px;margin:4px 0">'
        f'<div style="height:100%;width:{pct:.1f}%;background:{ACCENT};border-radius:2px"></div></div>'
    )
    autosave_note = (
        f' &nbsp;<span style="color:{TEXT_GHOST};font-size:0.85em">· auto-saving</span>'
        if auto_saving
        else ""
    )
    return (
        f'<div style="font-family:sans-serif;font-size:0.9em;color:{TEXT_MUTED};">'
        f"Variable <b>{position}</b> of {total} &nbsp;|&nbsp; "
        f'<span style="color:{ACCEPTED}">✓ {accepted} accepted</span> &nbsp;'
        f'<span style="color:{SKIPPED}">⊘ {skipped} skipped</span> &nbsp;'
        f"{remaining} remaining{autosave_note}{bar}</div>"
    )


def accepted_status_html(target: str, similarity: float) -> str:
    """Render the green "accepted" badge shown beside a variable's name."""
    return (
        f' &nbsp;<span style="color:{ACCEPTED};font-weight:normal">'
        f"✓ {target} ({similarity:.3f})</span>"
    )


def skipped_status_html() -> str:
    """Render the orange "skipped" badge shown beside a variable's name."""
    return f' &nbsp;<span style="color:{SKIPPED}">⊘ Skipped</span>'


def variable_panel_html(
    variable: str,
    description: str,
    values: str = "",
    status_html: str = "",
) -> str:
    """Render the panel describing the source variable under review.

    Args:
        variable: The ``table.variable`` identifier.
        description: The variable's description from the study's data dictionary.
        values: Comma-joined ``code=meaning`` labels, omitted when empty.
        status_html: Output of ``accepted_status_html`` or ``skipped_status_html``.

    Returns:
        str: HTML for an ipywidgets ``HTML`` widget.
    """
    values_line = ""
    if values:
        values_line = (
            f'<div style="color:{TEXT_FAINT};font-size:0.88em;'
            f'margin-top:2px">Values: {values}</div>'
        )
    return (
        f'<div style="font-family:sans-serif;padding:10px 14px;border-left:4px solid {ACCENT};'
        f'background:{PANEL_BG};margin:8px 0;border-radius:0 4px 4px 0">'
        f'<div style="font-size:1.1em;font-weight:bold;margin-bottom:4px">{variable}{status_html}</div>'
        f'<div style="color:{TEXT_BODY}">{description}</div>{values_line}</div>'
    )


def accept_label_html() -> str:
    """Render the "Accept:" caption that precedes the rank buttons."""
    return (
        f'<span style="color:{TEXT_MUTED};font-size:0.9em;'
        f'line-height:28px">&nbsp;Accept:</span>'
    )


def candidates_table_html(
    candidates: "pd.DataFrame", similarity_column: str = "Similarity"
) -> str:
    """Render the ranked candidate table, colour-banded by similarity.

    Args:
        candidates: One row per candidate, already column-filtered and renamed.
        similarity_column: Column to band and draw bars for.

    Returns:
        str: HTML table.
    """
    return (
        candidates.style.map(similarity_cell_css, subset=[similarity_column])
        .bar(subset=[similarity_column], color=ACCENT, vmin=0, vmax=1)
        .format({similarity_column: "{:.3f}"})
        .hide(axis="index")
        .to_html()
    )
