"""Tests for ai_harmonization.styles — shared palette, bands and table styling."""

import pandas as pd
import pytest

from ai_harmonization import styles
from ai_harmonization.formatters import VALUE_SEPARATOR


class TestSimilarityCellCss:
    @pytest.mark.parametrize("value", [0.75, 0.78, 0.88, 1.0])
    def test_strong_band(self, value):
        assert styles.similarity_cell_css(value) == styles.STRONG_CELL_CSS

    @pytest.mark.parametrize("value", [0.5, 0.66, 0.74])
    def test_moderate_band(self, value):
        assert styles.similarity_cell_css(value) == styles.MODERATE_CELL_CSS

    @pytest.mark.parametrize("value", [0.0, 0.31, 0.49])
    def test_weak_band(self, value):
        assert styles.similarity_cell_css(value) == styles.WEAK_CELL_CSS

    def test_boundaries_are_inclusive_at_the_lower_edge(self):
        """The notebook documents "green >= 0.75, yellow 0.5-0.75, red < 0.5"."""
        assert (
            styles.similarity_cell_css(styles.STRONG_SIMILARITY)
            == styles.STRONG_CELL_CSS
        )
        assert (
            styles.similarity_cell_css(styles.MODERATE_SIMILARITY)
            == styles.MODERATE_CELL_CSS
        )

    def test_one_definition_of_strong_across_the_package(self):
        """The widget's green band, the summary gradient's top, and the
        strong-match column in summarize_rank1_similarity must not drift."""
        import pandas as pd

        from ai_harmonization.dbgap import summarize_rank1_similarity

        assert styles.SUMMARY_GRADIENT_VMAX == styles.STRONG_SIMILARITY
        assert styles.SUMMARY_GRADIENT_VMIN == styles.MODERATE_SIMILARITY

        # A score exactly on the threshold counts as strong in the stats, and
        # is green in the widget.
        stats = summarize_rank1_similarity(
            pd.DataFrame(
                {
                    "Similarity": [styles.STRONG_SIMILARITY],
                    "Suggested Target Node.Property": ["T.slot"],
                }
            )
        )
        assert stats[f"≥{styles.STRONG_SIMILARITY:.2f} (strong)"] == "100%"
        assert (
            styles.similarity_cell_css(styles.STRONG_SIMILARITY)
            == styles.STRONG_CELL_CSS
        )

    def test_bands_are_distinct(self):
        assert (
            len(
                {styles.STRONG_CELL_CSS, styles.MODERATE_CELL_CSS, styles.WEAK_CELL_CSS}
            )
            == 3
        )


class TestStyleMappingQualitySummary:
    @pytest.fixture
    def summary(self):
        return pd.DataFrame(
            {
                "study_id": ["phs999998.v1.p1.c1", "phs999999.v1.p1.c1"],
                "Variables": [400, 100],
                "Mean sim (rank 1)": [0.5001, 0.6789],
                "Median sim": [0.4952, 0.6713],
                "Top bdchm target": ["TargetClass.field_c", "TargetClass.field_d"],
            }
        )

    def test_caption_is_set(self, summary):
        assert (
            styles.SUMMARY_CAPTION
            in styles.style_mapping_quality_summary(summary).to_html()
        )

    def test_caption_is_overridable(self, summary):
        html = styles.style_mapping_quality_summary(summary, caption="Custom").to_html()
        assert "Custom" in html

    def test_header_background_applied(self, summary):
        assert (
            styles.HEADER_BG in styles.style_mapping_quality_summary(summary).to_html()
        )

    def test_similarity_columns_formatted_to_three_decimals(self, summary):
        html = styles.style_mapping_quality_summary(summary).to_html()
        assert "0.500" in html and "0.679" in html
        assert "0.5001" not in html

    def test_non_similarity_columns_left_alone(self, summary):
        html = styles.style_mapping_quality_summary(summary).to_html()
        assert "400" in html
        assert "TargetClass.field_c" in html

    def test_gradient_colours_the_similarity_columns(self, summary):
        html = styles.style_mapping_quality_summary(summary).to_html()
        assert html.count("background-color") > len(styles.SUMMARY_TABLE_STYLES)

    def test_summary_without_similarity_columns_does_not_raise(self):
        """A degenerate summary should still render rather than KeyError."""
        minimal = pd.DataFrame({"study_id": ["x"], "Variables": [1]})
        assert "<table" in styles.style_mapping_quality_summary(minimal).to_html()

    def test_empty_summary_does_not_raise(self):
        empty = pd.DataFrame(columns=["study_id", "Mean sim (rank 1)", "Median sim"])
        assert "<table" in styles.style_mapping_quality_summary(empty).to_html()


class TestReviewWidgetUsesSharedPalette:
    """The widget and the summary table must not drift apart on colour."""

    def test_candidate_table_uses_the_shared_bands(self):
        from ai_harmonization.dbgap import CSV_HEADERS
        from ai_harmonization.review import VariableReviewSession

        rows = [("pht1.AGE", 1, 0.91), ("pht1.AGE", 2, 0.66), ("pht1.AGE", 3, 0.42)]
        df = pd.DataFrame(
            [
                {
                    "Original Node.Property": v,
                    "Suggested Target Node.Property": f"T.slot{r}",
                    "Similarity": s,
                    "Target Description": "d",
                    "Target Values": "",
                    "Original Description": "src",
                    "Original Values": "",
                    "study_id": "phs1",
                    "source_table_id": "pht1",
                    "source_variable_name": "AGE",
                    "prompt_variant": "A",
                    "rank": r,
                }
                for v, r, s in rows
            ],
            columns=CSV_HEADERS,
        )
        html = VariableReviewSession(df)._candidates_html()
        # One candidate in each band, so all three band colours must appear.
        for css in (
            styles.STRONG_CELL_CSS,
            styles.MODERATE_CELL_CSS,
            styles.WEAK_CELL_CSS,
        ):
            colour = css.split("background-color:")[1].split(";")[0]
            assert colour in html, f"missing band colour {colour}"


class TestReviewWidgetHtml:
    def test_progress_reports_counts_and_remaining(self):
        html = styles.progress_html(position=3, total=10, accepted=2, skipped=1)
        assert "Variable <b>3</b> of 10" in html
        assert "2 accepted" in html and "1 skipped" in html
        assert "7 remaining" in html

    def test_progress_bar_width_tracks_completion(self):
        assert "width:30.0%" in styles.progress_html(1, 10, 2, 1)
        assert "width:100.0%" in styles.progress_html(10, 10, 6, 4)

    def test_progress_survives_an_empty_session(self):
        """An empty mapping file must not divide by zero."""
        assert "width:0.0%" in styles.progress_html(1, 0, 0, 0)

    def test_autosave_note_only_when_enabled(self):
        assert "auto-saving" in styles.progress_html(1, 5, 0, 0, auto_saving=True)
        assert "auto-saving" not in styles.progress_html(1, 5, 0, 0, auto_saving=False)

    def test_variable_panel_shows_name_and_description(self):
        html = styles.variable_panel_html("pht1.AGE", "Age at enrollment")
        assert "pht1.AGE" in html and "Age at enrollment" in html

    def test_values_line_omitted_when_empty(self):
        assert "Values:" not in styles.variable_panel_html("v", "d", values="")
        assert "Values: 1=One" in styles.variable_panel_html("v", "d", values="1=One")

    def test_status_badges_use_the_decision_colours(self):
        assert styles.ACCEPTED in styles.accepted_status_html("T.slot", 0.912)
        assert "0.912" in styles.accepted_status_html("T.slot", 0.912)
        assert styles.SKIPPED in styles.skipped_status_html()

    def test_status_badge_embeds_into_the_panel(self):
        badge = styles.skipped_status_html()
        assert "⊘ Skipped" in styles.variable_panel_html("v", "d", status_html=badge)

    def test_candidates_table_bands_each_similarity(self):
        candidates = pd.DataFrame({"Rank": [1, 2, 3], "Similarity": [0.91, 0.66, 0.42]})
        html = styles.candidates_table_html(candidates)
        for css in (
            styles.STRONG_CELL_CSS,
            styles.MODERATE_CELL_CSS,
            styles.WEAK_CELL_CSS,
        ):
            assert css.split("background-color:")[1].split(";")[0] in html
        assert "0.910" in html  # formatted to three decimals


class TestStudyTextIsEscaped:
    """Study metadata is third-party text rendered into notebook HTML."""

    HOSTILE = '<script>alert("xss")</script>'

    def test_variable_panel_escapes_variable_description_and_values(self):
        out = styles.variable_panel_html(
            variable=self.HOSTILE,
            description=f"desc {self.HOSTILE}",
            values=f"1={self.HOSTILE}",
            status_html="",
        )
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_variable_panel_keeps_our_own_status_markup(self):
        """status_html is built by this module, so it must not be escaped."""
        out = styles.variable_panel_html(
            variable="v",
            description="d",
            values="",
            status_html='<span style="color:green">accepted</span>',
        )
        assert '<span style="color:green">accepted</span>' in out

    def test_candidates_table_escapes_cell_text(self):
        html = styles.candidates_table_html(
            pd.DataFrame([{"Target": self.HOSTILE, "Similarity": 0.9}])
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_candidates_table_still_formats_similarity(self):
        """Guards the call order: escaping and formatting must both survive,
        which they only do when set in a single format() call."""
        html = styles.candidates_table_html(
            pd.DataFrame([{"Target": self.HOSTILE, "Similarity": 0.9}])
        )
        assert "0.900" in html


class TestAbbreviateValueList:
    def test_short_list_is_returned_unchanged(self):
        short = VALUE_SEPARATOR.join(["1=Yes", "2=No"])
        assert styles.abbreviate_value_list(short) == short

    def test_non_strings_pass_through(self):
        """Cells can hold NaN when a variable has no values."""
        assert styles.abbreviate_value_list(None) is None
        assert styles.abbreviate_value_list(3.5) == 3.5

    def test_long_list_is_cut_and_counted(self):
        values = [f"{i}=meaning number {i}" for i in range(1, 120)]
        out = styles.abbreviate_value_list(VALUE_SEPARATOR.join(values))
        assert out.endswith("… (119 values)")
        assert len(out) < 200

    def test_cut_falls_between_values_not_inside_one(self):
        """A cell must not end on half an identifier."""
        values = [f"OBA:{i:07d}" for i in range(60)]
        out = styles.abbreviate_value_list(VALUE_SEPARATOR.join(values))
        shown = out.split(" … ")[0]
        assert all(part in values for part in shown.split(VALUE_SEPARATOR))

    def test_count_is_exact_when_values_contain_commas(self):
        """Some dbGaP value meanings contain ", " themselves. Splitting such a
        list on a comma would report more values than there are, and could end
        a cell on the back half of one."""
        values = [f"{i}=exercise class, level {i}" for i in range(1, 40)]
        out = styles.abbreviate_value_list(VALUE_SEPARATOR.join(values))
        assert out.endswith("… (39 values)")
        shown = out.split(" … ")[0]
        assert all(part in values for part in shown.split(VALUE_SEPARATOR))

    def test_singular_noun_for_one_value(self):
        out = styles.abbreviate_value_list("x" * 500)
        assert out.endswith("(1 value)")

    def test_single_oversized_value_is_still_bounded(self):
        """No separator to cut on, so the value itself has to be cut."""
        out = styles.abbreviate_value_list("x" * 5000, max_chars=50)
        assert len(out) < 100

    def test_respects_an_explicit_limit(self):
        values = VALUE_SEPARATOR.join(f"v{i}" for i in range(100))
        assert len(styles.abbreviate_value_list(values, max_chars=20)) < 60


class TestAbbreviateValueColumns:
    def _frame(self):
        long_values = VALUE_SEPARATOR.join(f"{i}=meaning {i}" for i in range(1, 100))
        return pd.DataFrame(
            [
                {
                    "Rank": 1,
                    "Target Description": "a description",
                    "Target Values": long_values,
                    "Best Target Values": long_values,
                    "Original Values": long_values,
                    "Similarity": 0.9,
                }
            ]
        )

    def test_abbreviates_every_column_ending_in_values(self):
        out = styles.abbreviate_value_columns(self._frame())
        for column in ("Target Values", "Best Target Values", "Original Values"):
            assert out[column][0].endswith("(99 values)")

    def test_leaves_other_columns_alone(self):
        frame = self._frame()
        out = styles.abbreviate_value_columns(frame)
        assert out["Target Description"][0] == "a description"
        assert out["Similarity"][0] == 0.9

    def test_does_not_mutate_the_input_frame(self):
        """The CSVs are written from the same frames, so they must keep the
        full value lists."""
        frame = self._frame()
        before = frame["Target Values"][0]
        styles.abbreviate_value_columns(frame)
        assert frame["Target Values"][0] == before

    def test_frame_without_value_columns_is_returned_as_is(self):
        frame = pd.DataFrame([{"Rank": 1, "Similarity": 0.5}])
        assert styles.abbreviate_value_columns(frame) is frame


class TestCandidatesTableAbbreviates:
    def test_wide_value_cell_does_not_dominate_the_table(self):
        long_values = VALUE_SEPARATOR.join(f"OBA:{i:07d}" for i in range(255))
        frame = pd.DataFrame(
            [
                {
                    "Target Description": "the type of measurement observed",
                    "Target Values": long_values,
                    "Similarity": 0.9 - rank / 100,
                }
                for rank in range(1, 11)
            ]
        )
        html = styles.candidates_table_html(frame)
        assert long_values not in html
        assert "(255 values)" in html
        # Ten candidates of a 3.5k-character cell would run past 35k.
        assert len(html) < 15000
