"""Tests for ai_harmonization.styles — shared palette, bands and table styling."""

import pandas as pd
import pytest

from ai_harmonization import styles


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
