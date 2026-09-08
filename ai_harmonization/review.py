"""
Interactive review widget for preliminary harmonization mappings.

VariableReviewSession provides a step-by-step expert review UI for
candidates produced by harmonize_studies.ipynb. One source variable
at a time — ranked candidates, similarity color-coding, accept or skip.

Usage:
    from ai_harmonization.review import VariableReviewSession

    session = VariableReviewSession.from_csv(
        mapping_file,
        sort_by='similarity',
        resume_from=state_file,
        auto_save=state_file,
    )
    session.start()
"""

import os
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

from ai_harmonization import styles
from ai_harmonization.dbgap import CSV_HEADERS


class VariableReviewSession:
    """Step-by-step expert review of harmonization candidates.

    Loading:
      from_csv(path)                         — CSV order
      from_csv(path, sort_by='similarity')   — highest similarity first
      from_csv(path, resume_from=state_file) — restore a prior session
      from_csv(path, auto_save=state_file)   — write outputs after every click

    Interaction:
      start()           — display the interactive UI
      accept(rank)      — record the rank-N candidate and advance
      skip()            — toggle unmappable; advances when skipping, stays when un-skipping
      clear_decision()  — discard the current variable's decision

    Output files (derived from the mapping file path):
      {study_id}_review_state.csv      — 3-column state for resuming
      {study_id}_curated_mappings.csv  — accepted variables
      {study_id}_skipped_variables.csv — skipped variables + manual_mapping column
    """

    CANDIDATE_COLUMNS = [
        "rank",
        "Suggested Target Node.Property",
        "Similarity",
        "Target Description",
        "Target Values",
        "prompt_variant",
    ]
    CANDIDATE_DISPLAY_NAMES = {
        "rank": "Rank",
        "prompt_variant": "Prompt",
    }
    SOURCE_COLUMNS = [
        "Original Node.Property",
        "study_id",
        "source_table_id",
        "source_variable_name",
        "Original Description",
        "Original Values",
    ]
    STATE_COLUMNS = ["Original Node.Property", "rank", "review_decision"]
    TARGET_RENAME = {
        "Suggested Target Node.Property": "Best Suggested Target",
        "Similarity": "Best Similarity",
        "Target Description": "Best Target Description",
        "Target Values": "Best Target Values",
    }

    def __init__(self, mappings_df, sort_by="index", auto_save=None):
        ordered_vars = list(dict.fromkeys(mappings_df["Original Node.Property"]))
        groups = dict(tuple(mappings_df.groupby("Original Node.Property", sort=False)))
        self._variables = [(var, groups[var].copy()) for var in ordered_vars]
        if sort_by == "similarity":
            self._variables.sort(key=lambda x: x[1]["Similarity"].max(), reverse=True)
        self._idx = 0
        self._accepted = {}
        self._skipped = set()
        self._df = mappings_df
        self._auto_save = auto_save
        # One accept button per candidate rank actually present in the data, so
        # a higher top_n_suggestions does not leave candidates unreachable.
        self._max_rank = int(mappings_df["rank"].max()) if len(mappings_df) else 0
        self._control_sets = []
        self._skip_buttons = []
        self._progress_w = None
        self._variable_w = None
        self._table_w = None

    @classmethod
    def from_csv(cls, path, sort_by="index", resume_from=None, auto_save=None):
        """Load a mapping CSV and return a ready-to-use session.

        Args:
            path (str): Path to a ``*_preliminary_mappings.csv`` file.
            sort_by (str): ``'similarity'`` to review highest-confidence variables
                first; ``'index'`` for original CSV order.
            resume_from (str | None): Path to a ``*_review_state.csv`` written by
                a previous session. Restores accepted/skipped decisions and jumps
                to the first unreviewed variable.
            auto_save (str | None): Path to write state after every accept/skip.
                Pass the same value as ``resume_from`` to enable seamless resume.

        Returns:
            VariableReviewSession: Initialised (and optionally restored) session.
        """
        try:
            df = pd.read_csv(
                path,
                dtype={"Original Values": str, "Target Values": str},
                low_memory=False,
            )
        except pd.errors.EmptyDataError:
            # A zero-byte file, e.g. from an interrupted harmonization run.
            df = pd.DataFrame(columns=CSV_HEADERS)
        session = cls(df, sort_by=sort_by, auto_save=auto_save)
        if session.n_variables == 0:
            print(
                f"No mapping rows found in {path} — nothing to review. "
                "Re-run harmonize_studies.ipynb for this study."
            )
            return session
        if resume_from:
            session._load_state(resume_from)
        return session

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path, quiet=False):
        """Write session state and derived output files to disk.

        Writes three files derived from ``path``:
        - ``path`` — 3-column state CSV (variable, rank, decision) for resuming.
        - ``*_curated_mappings.csv`` — accepted variables with full mapping columns.
        - ``*_skipped_variables.csv`` — skipped variables + empty ``manual_mapping`` column.

        Args:
            path (str): Destination for the review state CSV.
            quiet (bool): Suppress printed confirmation messages.
        """
        rows = []
        for var_name, row in self._accepted.items():
            rows.append(
                {
                    "Original Node.Property": var_name,
                    "rank": int(row["rank"]),
                    "review_decision": "accepted",
                }
            )
        for var_name in self._skipped:
            rows.append(
                {
                    "Original Node.Property": var_name,
                    "rank": None,
                    "review_decision": "skipped",
                }
            )
        # Columns are declared so a session with no decisions yet still writes a
        # header row, and rank uses the nullable integer dtype so that skipped
        # rows (rank=None) do not promote the column to float and serialise
        # accepted ranks as "2.0", which resume cannot parse.
        state_df = pd.DataFrame(rows, columns=self.STATE_COLUMNS)
        state_df["rank"] = state_df["rank"].astype("Int64")
        state_df.to_csv(path, index=False)

        curated_path = path.replace("_review_state.csv", "_curated_mappings.csv")
        if curated_path != path:
            self.curated_df.to_csv(curated_path, index=False, na_rep="N/A")
        skipped_path = path.replace("_review_state.csv", "_skipped_variables.csv")
        if skipped_path != path:
            self.skipped_df.to_csv(skipped_path, index=False, na_rep="N/A")

        if not quiet:
            print(
                f"Saved: {len(self._accepted)} accepted, {len(self._skipped)} skipped → {path}"
            )
            if curated_path != path:
                print(f"Curated mappings → {curated_path}")
            if skipped_path != path:
                print(f"Skipped variables → {skipped_path}")

    def _load_state(self, path):
        if not os.path.exists(path):
            print(f"No state file found at {path} — starting fresh.")
            return
        try:
            state_df = pd.read_csv(path, dtype=str)
        except pd.errors.EmptyDataError:
            return
        if state_df.empty:
            return
        groups = dict(self._variables)
        for _, state_row in state_df.iterrows():
            var_name = state_row["Original Node.Property"]
            decision = state_row.get("review_decision", "")
            if decision == "skipped" and var_name in groups:
                self._skipped.add(var_name)
            elif decision == "accepted" and var_name in groups:
                # float() first: state files written before the Int64 fix hold
                # ranks like "2.0", which int() rejects outright.
                try:
                    rank = int(float(state_row.get("rank", 1)))
                except (ValueError, TypeError):
                    rank = 1
                group_df = groups[var_name]
                match = group_df[group_df["rank"] == rank]
                self._accepted[var_name] = (
                    match.iloc[0] if not match.empty else group_df.iloc[0]
                )
        self._goto_first_unreviewed()
        reviewed = len(self._accepted) + len(self._skipped)
        print(
            f"Resumed: {len(self._accepted)} accepted, {len(self._skipped)} skipped"
            f" — starting at variable {self._idx + 1} of {self.n_variables}"
            f" ({self.n_variables - reviewed} remaining)"
        )

    def _auto_save_if_set(self):
        if self._auto_save:
            self.save(self._auto_save, quiet=True)

    def _goto_first_unreviewed(self):
        reviewed = set(self._accepted) | self._skipped
        for i, (var_name, _) in enumerate(self._variables):
            if var_name not in reviewed:
                self._idx = i
                return
        self._idx = self.n_variables - 1

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_variables(self):
        return len(self._variables)

    @property
    def _current(self):
        if not self._variables:
            raise IndexError("This review session has no variables to review.")
        return self._variables[self._idx]

    @property
    def curated_df(self):
        if not self._accepted:
            return pd.DataFrame(columns=self._df.columns)
        return pd.DataFrame(list(self._accepted.values()))

    @property
    def skipped_df(self):
        if not self._skipped:
            return pd.DataFrame(
                columns=self.SOURCE_COLUMNS
                + list(self.TARGET_RENAME.values())
                + ["manual_mapping"]
            )
        groups = dict(self._variables)
        rows = []
        for var_name in self._skipped:
            if var_name not in groups:
                continue
            group_df = groups[var_name]
            top = group_df[group_df["rank"] == 1]
            row_src = top.iloc[0] if not top.empty else group_df.iloc[0]
            row = {
                col: row_src.get(col, "")
                for col in self.SOURCE_COLUMNS
                if col in row_src.index
            }
            for src_col, dst_col in self.TARGET_RENAME.items():
                row[dst_col] = (
                    row_src.get(src_col, "") if src_col in row_src.index else ""
                )
            row["manual_mapping"] = ""
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Navigation ────────────────────────────────────────────────────────────

    def accept(self, rank=1):
        """Accept the rank-N candidate for the current variable and advance.

        Does nothing if the current variable has no candidate at that rank.

        Args:
            rank (int): 1-based rank of the candidate to accept.
        """
        var_name, group_df = self._current
        match = group_df[group_df["rank"] == rank]
        if match.empty:
            print(f"{var_name} has no rank-{rank} candidate — nothing accepted.")
            return
        self._accepted[var_name] = match.iloc[0]
        self._skipped.discard(var_name)
        self._advance()
        self._auto_save_if_set()

    def skip(self):
        """Toggle the current variable's skipped state.

        Marks it as unmappable and advances. Calling this on an
        already-skipped variable clears the decision and stays put, so a
        mistaken skip can be undone by navigating back and clicking again.
        """
        var_name, _ = self._current
        if var_name in self._skipped:
            self._skipped.discard(var_name)
            self._refresh()
        else:
            self._skipped.add(var_name)
            self._accepted.pop(var_name, None)
            self._advance()
        self._auto_save_if_set()

    def clear_decision(self):
        """Discard the current variable's accept/skip decision and stay put."""
        var_name, _ = self._current
        self._accepted.pop(var_name, None)
        self._skipped.discard(var_name)
        self._refresh()
        self._auto_save_if_set()

    def _advance(self):
        if self._idx < self.n_variables - 1:
            self._idx += 1
        self._refresh()

    def prev(self):
        """Navigate to the previous variable without changing its decision."""
        if self._idx > 0:
            self._idx -= 1
        self._refresh()

    def next(self):
        """Navigate to the next variable without changing its decision."""
        if self._idx < self.n_variables - 1:
            self._idx += 1
        self._refresh()

    def goto(self, idx):
        """Jump to a specific variable by 0-based index.

        Args:
            idx (int): Target index, clamped to [0, n_variables - 1].
        """
        self._idx = max(0, min(idx, self.n_variables - 1))
        self._refresh()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _progress_html(self):
        return styles.progress_html(
            position=self._idx + 1,
            total=self.n_variables,
            accepted=len(self._accepted),
            skipped=len(self._skipped),
            auto_saving=bool(self._auto_save),
        )

    def _variable_html(self):
        var_name, group_df = self._current
        row0 = group_df.iloc[0]

        status_html = ""
        if var_name in self._accepted:
            accepted = self._accepted[var_name]
            status_html = styles.accepted_status_html(
                target=accepted["Suggested Target Node.Property"],
                similarity=float(accepted["Similarity"]),
            )
        elif var_name in self._skipped:
            status_html = styles.skipped_status_html()

        values = str(row0.get("Original Values", "") or "")
        return styles.variable_panel_html(
            variable=var_name,
            description=str(row0.get("Original Description", "") or ""),
            values="" if values == "nan" else values,
            status_html=status_html,
        )

    def _candidates_html(self):
        _, group_df = self._current
        cols = [c for c in self.CANDIDATE_COLUMNS if c in group_df.columns]
        candidates = (
            group_df[cols]
            .reset_index(drop=True)
            .rename(columns=self.CANDIDATE_DISPLAY_NAMES)
        )
        return styles.candidates_table_html(candidates)

    def _update_decision_buttons(self):
        """Highlight the buttons matching the current variable's decision.

        The skip button doubles as its own state indicator, so a variable that
        was already skipped is visibly distinguishable from an undecided one.
        """
        var_name, group_df = self._current
        accepted_rank = None
        if var_name in self._accepted:
            try:
                accepted_rank = int(self._accepted[var_name]["rank"])
            except (ValueError, TypeError, KeyError):
                pass
        available_ranks = set(group_df["rank"])

        for control_set in self._control_sets:
            for rank, btn in control_set.items():
                btn.button_style = (
                    styles.ACCEPTED_BUTTON_STYLE if rank == accepted_rank else ""
                )
                btn.disabled = rank not in available_ranks

        is_skipped = var_name in self._skipped
        for btn in self._skip_buttons:
            btn.description = "⊘ Skipped" if is_skipped else "Skip →"
            btn.button_style = (
                styles.SKIPPED_BUTTON_STYLE if is_skipped else styles.SKIP_BUTTON_STYLE
            )
            btn.tooltip = (
                "Click to un-skip this variable"
                if is_skipped
                else "Mark as having no good mapping and advance"
            )

    def _refresh(self):
        if self._progress_w is None or not self._variables:
            return
        self._progress_w.value = self._progress_html()
        self._variable_w.value = self._variable_html()
        self._table_w.value = self._candidates_html()
        self._update_decision_buttons()

    # ── Widget UI ─────────────────────────────────────────────────────────────

    def _make_controls(self, handlers):
        W = widgets.Layout
        btn_prev = widgets.Button(
            description="← Back", layout=W(width=styles.BACK_BUTTON_WIDTH)
        )
        btn_next = widgets.Button(
            description="→", layout=W(width=styles.NEXT_BUTTON_WIDTH)
        )
        btn_skip = widgets.Button(
            description="Skip →",
            button_style=styles.SKIP_BUTTON_STYLE,
            layout=W(width=styles.SKIP_BUTTON_WIDTH),
        )
        btn_accept1 = widgets.Button(
            description="✓ #1", layout=W(width=styles.ACCEPT_FIRST_BUTTON_WIDTH)
        )
        rank_btns = [
            widgets.Button(
                description=f"#{rank}", layout=W(width=styles.ACCEPT_RANK_BUTTON_WIDTH)
            )
            for rank in range(2, self._max_rank + 1)
        ]
        btn_prev.on_click(handlers["prev"])
        btn_next.on_click(handlers["next"])
        btn_skip.on_click(handlers["skip"])
        btn_accept1.on_click(handlers["accept1"])
        for rank, btn in enumerate(rank_btns, start=2):
            btn.on_click(handlers[f"accept{rank}"])

        self._skip_buttons.append(btn_skip)
        self._control_sets.append(
            {
                1: btn_accept1,
                **{rank: btn for rank, btn in enumerate(rank_btns, start=2)},
            }
        )
        label = widgets.HTML(styles.accept_label_html())
        return widgets.HBox(
            [btn_prev, btn_next, btn_skip, label, btn_accept1] + rank_btns,
            layout=widgets.Layout(
                margin=styles.CONTROLS_MARGIN, gap=styles.CONTROLS_GAP
            ),
        )

    def start(self):
        """Render the interactive review UI in the current Jupyter cell.

        Displays navigation buttons (repeated at top and bottom), a progress bar,
        the current variable panel, and the ranked candidates table. One accept
        button is rendered per candidate rank present in the mapping file. Call
        once per session; subsequent accept/skip/prev/next calls refresh the UI
        in place.
        """
        if not self._variables:
            print("Nothing to review — this session has no variables.")
            return

        self._progress_w = widgets.HTML()
        self._variable_w = widgets.HTML()
        self._table_w = widgets.HTML()
        self._control_sets = []
        self._skip_buttons = []

        handlers = {
            "prev": lambda _: self.prev(),
            "next": lambda _: self.next(),
            "skip": lambda _: self.skip(),
            **{
                f"accept{rank}": (lambda r: lambda _: self.accept(rank=r))(rank)
                for rank in range(1, self._max_rank + 1)
            },
        }
        top_controls = self._make_controls(handlers)
        bot_controls = self._make_controls(handlers)
        display(
            widgets.VBox(
                [
                    top_controls,
                    self._progress_w,
                    self._variable_w,
                    self._table_w,
                    bot_controls,
                ]
            )
        )
        self._refresh()
