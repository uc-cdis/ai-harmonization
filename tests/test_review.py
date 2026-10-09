"""Tests for ai_harmonization.review.VariableReviewSession.

Everything here runs headlessly: decisions are driven through accept/skip/prev
rather than through the ipywidgets buttons, which only wrap those same calls.
"""

# The tests read the session's internal state directly, which is what they
# check, so protected access is expected here.
# pylint: disable=protected-access

import logging

import pandas as pd
import pytest

from ai_harmonization.dbgap import CSV_HEADERS
from ai_harmonization.review import VariableReviewSession


CANDIDATES = [
    # variable,      rank, target,             similarity, variant
    ("pht001.AGE", 1, "TargetClass.field_a", 0.91, "B"),
    ("pht001.AGE", 2, "TargetClass.field_b", 0.66, "A"),
    ("pht001.AGE", 3, "OtherClass.field_e", 0.51, "C"),
    ("pht001.SEX", 1, "TargetClass.field_c", 0.88, "A"),
    ("pht001.SEX", 2, "OtherClass.field_g", 0.72, "D"),
    ("pht001.SEX", 3, "TargetClass.field_d", 0.40, "B"),
    ("pht001.WEIRDVAR", 1, "OtherClass.field_e", 0.31, "C"),
]

VARIABLE_IDS = {
    "pht001.AGE": "phv99999901.v1",
    "pht001.SEX": "phv99999902.v1",
    "pht001.WEIRDVAR": "phv99999903.v1",
}


def make_mappings_df(rows=CANDIDATES):
    """Build a mapping DataFrame shaped exactly like harmonize_studies.ipynb writes.

    Passing ``rows=[]`` yields a header-only frame, which is what that notebook
    produces for a study where no variable got a suggestion.
    """
    return pd.DataFrame(
        [
            {
                "Original Node.Property": variable,
                "Suggested Target Node.Property": target,
                "Similarity": similarity,
                "Target Description": f"Description of {target}",
                "Target Values": "",
                "Original Description": f"Description of {variable}",
                "Original Values": "",
                "study_id": "phs999999.v1.p1.c1",
                "source_table_id": variable.split(".")[0],
                "source_variable_name": variable.split(".")[1],
                "source_variable_id": VARIABLE_IDS[variable],
                "prompt_variant": variant,
                "rank": rank,
            }
            for variable, rank, target, similarity, variant in rows
        ],
        columns=CSV_HEADERS,
    )


@pytest.fixture
def mapping_csv(tmp_path):
    """Write the sample mappings to a preliminary-mappings CSV and return its path."""
    path = tmp_path / "phs999999.v1.p1.c1_preliminary_mappings.csv"
    make_mappings_df().to_csv(path, index=False)
    return str(path)


@pytest.fixture
def session():
    """Build a review session over the sample mappings."""
    return VariableReviewSession(make_mappings_df())


class TestLoading:
    def test_groups_rows_into_variables(self, session):
        """The seven candidate rows group into one entry per source variable."""
        assert session.n_variables == 3

    def test_max_rank_derived_from_data(self, session):
        """The UI renders one accept button per rank, so this drives the layout."""
        assert session._max_rank == 3

    def test_csv_order_preserved_by_default(self, session):
        """Without sort_by, variables stay in the order the mapping rows list them."""
        assert [name for name, _ in session._variables] == [
            "pht001.AGE",
            "pht001.SEX",
            "pht001.WEIRDVAR",
        ]

    def test_sort_by_similarity_orders_by_best_candidate(self):
        """sort_by="similarity" orders variables by best candidate, highest first."""
        session = VariableReviewSession(make_mappings_df(), sort_by="similarity")
        assert [name for name, _ in session._variables] == [
            "pht001.AGE",
            "pht001.SEX",
            "pht001.WEIRDVAR",
        ]

    def test_from_csv_loads_rows(self, mapping_csv):
        """from_csv reads a mapping file from disk and finds all three variables."""
        session = VariableReviewSession.from_csv(mapping_csv)
        assert session.n_variables == 3


class TestEmptyMappingFile:
    """A study where no variable got a suggestion still produces a mapping
    file, so loading one must report the situation rather than crash."""

    @pytest.fixture
    def empty_csv(self, tmp_path):
        """Write a header-only mapping CSV and return its path."""
        path = tmp_path / "phs999997.v1.p1.c1_preliminary_mappings.csv"
        make_mappings_df(rows=[]).to_csv(path, index=False)
        return str(path)

    def test_from_csv_does_not_raise(self, empty_csv):
        """A header-only mapping file loads as a session with zero variables."""
        session = VariableReviewSession.from_csv(empty_csv)
        assert session.n_variables == 0

    def test_from_csv_reports_the_problem(self, empty_csv, caplog):
        """Loading a header-only file logs a warning that there is nothing to review."""
        with caplog.at_level(logging.WARNING):
            VariableReviewSession.from_csv(empty_csv)
        assert "nothing to review" in caplog.text

    def test_progress_does_not_divide_by_zero(self, empty_csv):
        """With zero variables the progress HTML still renders "0 remaining"."""
        session = VariableReviewSession.from_csv(empty_csv)
        assert "0 remaining" in session._progress_html()

    def test_accept_raises_a_clear_error(self, empty_csv):
        """accept on an empty session raises IndexError saying it has no variables."""
        session = VariableReviewSession.from_csv(empty_csv)
        with pytest.raises(IndexError, match="no variables"):
            session.accept()

    def test_start_reports_instead_of_rendering(self, empty_csv, caplog):
        """start on an empty session logs a warning that there is nothing to review."""
        session = VariableReviewSession.from_csv(empty_csv)
        with caplog.at_level(logging.WARNING):
            session.start()
        assert "Nothing to review" in caplog.text

    def test_resume_from_is_skipped(self, empty_csv, tmp_path):
        """An empty mapping file loads with zero variables even with resume_from set."""
        state = tmp_path / "state.csv"
        make_mappings_df(rows=[]).to_csv(state, index=False)
        session = VariableReviewSession.from_csv(empty_csv, resume_from=str(state))
        assert session.n_variables == 0

    def test_zero_byte_file_does_not_raise(self, tmp_path, caplog):
        """An interrupted harmonization run can leave a truly empty file."""
        path = tmp_path / "phs999997.v1.p1.c1_preliminary_mappings.csv"
        path.write_text("")
        with caplog.at_level(logging.WARNING):
            session = VariableReviewSession.from_csv(str(path))
        assert session.n_variables == 0
        assert "nothing to review" in caplog.text


class TestAccept:
    def test_records_the_chosen_rank(self, session):
        """accept(rank=2) records the rank-2 candidate, not the top one."""
        session.accept(rank=2)
        accepted = session._accepted["pht001.AGE"]
        assert accepted["Suggested Target Node.Property"] == "TargetClass.field_b"

    def test_advances_to_the_next_variable(self, session):
        """Accepting moves the session on to the next variable."""
        session.accept(rank=1)
        assert session._current[0] == "pht001.SEX"

    def test_out_of_range_rank_records_nothing(self, session):
        """Buttons are global but candidate counts are per-variable."""
        session.goto(2)
        session.accept(rank=3)
        assert "pht001.WEIRDVAR" not in session._accepted

    def test_out_of_range_rank_does_not_advance(self, session):
        """Accepting a rank the variable lacks leaves the session on that variable."""
        session.goto(2)
        session.accept(rank=3)
        assert session._current[0] == "pht001.WEIRDVAR"

    def test_out_of_range_rank_explains_itself(self, session, caplog):
        """Accepting a rank the variable lacks logs a warning naming that rank."""
        session.goto(2)
        with caplog.at_level(logging.WARNING):
            session.accept(rank=3)
        assert "no rank-3 candidate" in caplog.text

    def test_accepting_clears_a_previous_skip(self, session):
        """Accepting a skipped variable drops the skip and records the accept."""
        session.skip()
        session.goto(0)
        session.accept(rank=1)
        assert "pht001.AGE" not in session._skipped
        assert "pht001.AGE" in session._accepted

    def test_last_variable_stays_put(self, session):
        """Accepting on the last variable leaves the session on it."""
        session.goto(2)
        session.accept(rank=1)
        assert session._current[0] == "pht001.WEIRDVAR"


class TestSkipToggle:
    def test_skip_marks_and_advances(self, session):
        """Skipping marks the variable as skipped and moves on to the next one."""
        session.skip()
        assert "pht001.AGE" in session._skipped
        assert session._current[0] == "pht001.SEX"

    def test_skipping_again_un_skips(self, session):
        """Skipping is a toggle, so a mistaken skip can be undone by
        navigating back and clicking again."""
        session.skip()
        session.prev()
        session.skip()
        assert "pht001.AGE" not in session._skipped

    def test_un_skipping_stays_on_the_variable(self, session):
        """Un-skipping leaves the session on that variable instead of advancing."""
        session.skip()
        session.prev()
        session.skip()
        assert session._current[0] == "pht001.AGE"

    def test_skip_clears_a_previous_accept(self, session):
        """Skipping an accepted variable drops the accept and records the skip."""
        session.accept(rank=1)
        session.prev()
        session.skip()
        assert "pht001.AGE" not in session._accepted
        assert "pht001.AGE" in session._skipped

    def test_un_skip_is_persisted(self, session, tmp_path):
        """With auto-save on, un-skipping rewrites the state file without the skip."""
        state = str(tmp_path / "phs999999_review_state.csv")
        session._auto_save = state
        session.skip()
        session.prev()
        session.skip()
        assert pd.read_csv(state).empty

    def test_clear_decision_removes_accept(self, session):
        """clear_decision drops the accept recorded for the current variable."""
        session.accept(rank=1)
        session.goto(0)
        session.clear_decision()
        assert "pht001.AGE" not in session._accepted

    def test_clear_decision_removes_skip(self, session):
        """clear_decision drops the skip recorded for the current variable."""
        session.skip()
        session.goto(0)
        session.clear_decision()
        assert "pht001.AGE" not in session._skipped


class TestNavigation:
    def test_prev_does_not_change_decisions(self, session):
        """Navigating back to an accepted variable leaves its accept in place."""
        session.accept(rank=1)
        session.prev()
        assert "pht001.AGE" in session._accepted

    def test_prev_stops_at_the_first_variable(self, session):
        """prev on the first variable leaves the session on it."""
        session.prev()
        assert session._current[0] == "pht001.AGE"

    def test_next_stops_at_the_last_variable(self, session):
        """Repeated next calls stop at the last variable instead of running past it."""
        for _ in range(10):
            session.next()
        assert session._current[0] == "pht001.WEIRDVAR"

    def test_goto_clamps_out_of_range_index(self, session):
        """goto clamps an index beyond either end to the first or last variable."""
        session.goto(99)
        assert session._current[0] == "pht001.WEIRDVAR"
        session.goto(-5)
        assert session._current[0] == "pht001.AGE"


class TestOutputFiles:
    @pytest.fixture
    def state_path(self, tmp_path):
        """Return a state file path under tmp_path with the conventional suffix."""
        return str(tmp_path / "phs999999.v1.p1.c1_review_state.csv")

    def test_writes_all_three_files(self, session, state_path, tmp_path):
        """save writes exactly the state, curated and skipped files, sharing a stem."""
        session.accept(rank=1)
        session.skip()
        session.save(state_path, quiet=True)

        names = {p.name for p in tmp_path.iterdir()}
        assert names == {
            "phs999999.v1.p1.c1_review_state.csv",
            "phs999999.v1.p1.c1_curated_mappings.csv",
            "phs999999.v1.p1.c1_skipped_variables.csv",
        }

    def test_writes_all_three_files_for_an_unconventional_name(self, session, tmp_path):
        """The sibling paths come from the stem, so a state file without the
        conventional suffix still yields three distinct paths."""
        session.accept(rank=1)
        session.skip()
        session.save(str(tmp_path / "mystate.csv"), quiet=True)

        names = {f.name for f in tmp_path.iterdir()}
        assert names == {
            "mystate.csv",
            "mystate_curated_mappings.csv",
            "mystate_skipped_variables.csv",
        }

    def test_output_paths_never_collide_with_the_state_path(self):
        """Both output paths differ from the state path and from each other."""
        for name in ("s_review_state.csv", "mystate.csv", "no_extension"):
            curated, skipped = VariableReviewSession.output_paths(name)
            assert curated != name and skipped != name
            assert curated != skipped

    def test_output_paths_does_not_substitute_inside_a_directory_name(self):
        """Only a trailing suffix may be stripped: a directory name that
        happens to contain it must survive intact."""
        curated, _ = VariableReviewSession.output_paths(
            "out/_review_state.csv_runs/phs1_review_state.csv"
        )
        assert curated == "out/_review_state.csv_runs/phs1_curated_mappings.csv"

    def test_saving_twice_is_byte_identical(self, session, tmp_path):
        """Row order is sorted, so an unchanged session re-saves identically."""
        session.accept(rank=1)
        session.skip()
        first = tmp_path / "a_review_state.csv"
        second = tmp_path / "b_review_state.csv"
        session.save(str(first), quiet=True)
        session.save(str(second), quiet=True)
        assert first.read_text() == second.read_text()
        assert (tmp_path / "a_curated_mappings.csv").read_text() == (
            tmp_path / "b_curated_mappings.csv"
        ).read_text()

    def test_state_records_both_decision_types(self, session, state_path):
        """The state file records an accept with its rank and a skip as skipped."""
        session.accept(rank=2)
        session.skip()
        session.save(state_path, quiet=True)

        state = pd.read_csv(state_path).set_index("Original Node.Property")
        assert state.loc["pht001.AGE", "review_decision"] == "accepted"
        assert state.loc["pht001.AGE", "rank"] == 2
        assert state.loc["pht001.SEX", "review_decision"] == "skipped"

    def test_curated_file_holds_one_row_per_accepted_variable(
        self, session, state_path
    ):
        """The curated file holds one row per accept, each with the chosen target."""
        session.accept(rank=1)
        session.accept(rank=2)
        session.save(state_path, quiet=True)

        curated = pd.read_csv(state_path.replace("_review_state", "_curated_mappings"))
        assert len(curated) == 2
        assert set(curated["Suggested Target Node.Property"]) == {
            "TargetClass.field_a",
            "OtherClass.field_g",
        }

    def test_skipped_file_has_a_manual_mapping_column(self, session, state_path):
        """The skipped file has one row per skip and a manual_mapping column."""
        session.skip()
        session.save(state_path, quiet=True)

        skipped = pd.read_csv(state_path.replace("_review_state", "_skipped_variables"))
        assert "manual_mapping" in skipped.columns
        assert len(skipped) == 1

    def test_skipped_file_carries_the_best_automated_guess(self, session, state_path):
        """A skipped variable's row gives its rank-1 target as Best Suggested Target."""
        session.skip()
        session.save(state_path, quiet=True)

        skipped = pd.read_csv(state_path.replace("_review_state", "_skipped_variables"))
        assert skipped.loc[0, "Best Suggested Target"] == "TargetClass.field_a"

    def test_skipped_file_carries_the_variable_id(self, session, state_path):
        """A skipped variable keeps its variable_id."""
        session.skip()
        session.save(state_path, quiet=True)

        skipped = pd.read_csv(state_path.replace("_review_state", "_skipped_variables"))
        assert skipped.loc[0, "source_variable_id"] == "phv99999901.v1"

    def test_curated_file_carries_the_variable_id(self, session, state_path):
        """An accepted variable's row in the curated file keeps its variable_id."""
        session.accept(rank=1)
        session.save(state_path, quiet=True)

        curated = pd.read_csv(state_path.replace("_review_state", "_curated_mappings"))
        assert curated.loc[0, "source_variable_id"] == "phv99999901.v1"

    def test_mapping_without_variable_ids_still_writes_skipped(self, state_path):
        """A mapping file written before the id column existed still reviews."""
        session = VariableReviewSession(
            make_mappings_df().drop(columns=["source_variable_id"])
        )
        session.skip()
        session.save(state_path, quiet=True)

        skipped = pd.read_csv(state_path.replace("_review_state", "_skipped_variables"))
        assert len(skipped) == 1
        assert "source_variable_id" not in skipped.columns

    def test_no_decisions_writes_empty_but_valid_files(self, session, state_path):
        """With no decisions, save writes a state file that parses as an empty table."""
        session.save(state_path, quiet=True)
        assert pd.read_csv(state_path).empty

    def test_auto_save_writes_after_every_decision(self, session, state_path):
        """With auto-save on, the state file is rewritten after each accept and skip."""
        session._auto_save = state_path
        session.accept(rank=1)
        assert len(pd.read_csv(state_path)) == 1
        session.skip()
        assert len(pd.read_csv(state_path)) == 2


class TestResume:
    @pytest.fixture
    def state_path(self, tmp_path):
        """Return a state file path under tmp_path with the conventional suffix."""
        return str(tmp_path / "phs999999.v1.p1.c1_review_state.csv")

    def test_round_trip_restores_decisions(self, session, state_path, mapping_csv):
        """A saved session resumes with the same accepted rank and skipped variable."""
        session.accept(rank=2)
        session.skip()
        session.save(state_path, quiet=True)

        resumed = VariableReviewSession.from_csv(mapping_csv, resume_from=state_path)
        assert resumed._accepted["pht001.AGE"]["rank"] == 2
        assert "pht001.SEX" in resumed._skipped

    def test_accepted_rank_survives_a_skip_in_the_same_session(
        self, session, state_path, mapping_csv
    ):
        """Skipped rows carry no rank, so the column is a nullable integer:
        accepted ranks stay whole numbers in the file rather than "3.0"."""
        session.accept(rank=3)
        session.skip()
        session.save(state_path, quiet=True)

        assert ",3," in open(state_path).read()
        resumed = VariableReviewSession.from_csv(mapping_csv, resume_from=state_path)
        assert resumed._accepted["pht001.AGE"]["rank"] == 3

    def test_float_formatted_rank_is_still_readable(self, mapping_csv, state_path):
        """The state file is plain CSV a curator may open in a spreadsheet, and
        Excel rewrites an integer column as "2.0". Resume must still parse it."""
        with open(state_path, "w") as f:
            f.write(
                "Original Node.Property,rank,review_decision\npht001.AGE,2.0,accepted\n"
            )

        resumed = VariableReviewSession.from_csv(mapping_csv, resume_from=state_path)
        assert resumed._accepted["pht001.AGE"]["rank"] == 2

    def test_resume_jumps_to_first_unreviewed(self, session, state_path, mapping_csv):
        """A resumed session opens on the first variable with no decision yet."""
        session.accept(rank=1)
        session.skip()
        session.save(state_path, quiet=True)

        resumed = VariableReviewSession.from_csv(mapping_csv, resume_from=state_path)
        assert resumed._current[0] == "pht001.WEIRDVAR"

    def test_missing_state_file_starts_fresh(self, mapping_csv, tmp_path, caplog):
        """A missing state file logs "starting fresh" and nothing is accepted."""
        with caplog.at_level(logging.INFO):
            resumed = VariableReviewSession.from_csv(
                mapping_csv, resume_from=str(tmp_path / "nope.csv")
            )
        assert resumed._accepted == {}
        assert "starting fresh" in caplog.text

    def test_unknown_variable_in_state_is_ignored(self, mapping_csv, state_path):
        """A skip for a variable absent from the mapping file is dropped on resume."""
        pd.DataFrame(
            [
                {
                    "Original Node.Property": "pht999.GONE",
                    "rank": None,
                    "review_decision": "skipped",
                },
            ]
        ).to_csv(state_path, index=False)

        resumed = VariableReviewSession.from_csv(mapping_csv, resume_from=state_path)
        assert resumed._skipped == set()

    def test_stale_rank_is_left_unreviewed_not_guessed(
        self, mapping_csv, state_path, caplog
    ):
        """A rank with no matching candidate is dropped rather than rewritten."""
        pd.DataFrame(
            [
                {
                    "Original Node.Property": "pht001.WEIRDVAR",
                    "rank": 9,
                    "review_decision": "accepted",
                },
            ]
        ).to_csv(state_path, index=False)

        with caplog.at_level(logging.WARNING):
            resumed = VariableReviewSession.from_csv(
                mapping_csv, resume_from=state_path
            )
        assert "pht001.WEIRDVAR" not in resumed._accepted
        assert "pht001.WEIRDVAR" not in resumed._skipped
        assert "No rank-9 candidate" in caplog.text

    def test_unreadable_rank_is_left_unreviewed(self, mapping_csv, state_path, caplog):
        """A rank that will not parse at all is dropped the same way."""
        pd.DataFrame(
            [
                {
                    "Original Node.Property": "pht001.AGE",
                    "rank": "not-a-number",
                    "review_decision": "accepted",
                },
            ]
        ).to_csv(state_path, index=False)

        with caplog.at_level(logging.WARNING):
            resumed = VariableReviewSession.from_csv(
                mapping_csv, resume_from=state_path
            )
        assert "pht001.AGE" not in resumed._accepted
        assert "Unreadable rank" in caplog.text
