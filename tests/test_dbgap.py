"""Tests for ai_harmonization.dbgap — XML parsing, CSV row building, study stats."""

import textwrap

import pandas as pd
import pytest

from ai_harmonization.harmonization_approaches.base import SingleHarmonizationSuggestion
from ai_harmonization.dbgap import (
    CSV_HEADERS,
    build_mapping_rows,
    find_study_metadata_files,
    parse_dbgap_table,
    summarize_rank1_similarity,
)
from ai_harmonization.formatters import VALUE_SEPARATOR


DATA_DICT_XML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <data_table id="pht999999.v1.p1" study_id="phs999999" date_created="2024-01-01">
      <variable id="phv99999901.v1">
        <name>SUBJID</name>
        <description>Subject identifier</description>
      </variable>
      <variable id="phv99999902.v1">
        <name>SEX</name>
        <description>Biological sex</description>
        <value code="1">Male</value>
        <value code="2">Female</value>
      </variable>
    </data_table>
"""
)

VAR_REPORT_XML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <data_table>
      <variable>
        <name>SUBJID</name>
        <type>string</type>
      </variable>
      <variable>
        <name>SEX</name>
        <type>encoded value</type>
      </variable>
    </data_table>
"""
)


@pytest.fixture
def dict_path(tmp_path):
    """Write the two-variable data dictionary to a temp file and return its path."""
    p = tmp_path / "pht999999.v1_data_dict.xml"
    p.write_text(DATA_DICT_XML)
    return str(p)


@pytest.fixture
def report_path(tmp_path):
    """Write the var report for both variables to a temp file and return its path."""
    p = tmp_path / "pht999999.v1_var_report.xml"
    p.write_text(VAR_REPORT_XML)
    return str(p)


class TestParseDbgapTable:
    def test_returns_table_id_and_model(self, dict_path):
        """The table id comes from the data_table element and the model has one node."""
        table_id, model = parse_dbgap_table(dict_path)
        assert table_id == "pht999999.v1.p1"
        assert len(model.nodes) == 1

    def test_property_names(self, dict_path):
        """Each variable becomes a property named after it, in document order."""
        _, model = parse_dbgap_table(dict_path)
        names = [p.name for p in model.nodes[0].properties]
        assert names == ["SUBJID", "SEX"]

    def test_enum_values_populated(self, dict_path):
        """Values hold the value meanings and value_labels hold code=meaning pairs."""
        _, model = parse_dbgap_table(dict_path)
        sex = next(p for p in model.nodes[0].properties if p.name == "SEX")
        assert sex.values == ["Male", "Female"]
        assert "1=Male" in sex.additional_metadata["value_labels"]

    def test_variable_id_is_captured(self, dict_path):
        """Each variable carries its variable_id verbatim, version suffix included."""
        _, model = parse_dbgap_table(dict_path)
        variable_ids = {
            p.name: p.additional_metadata["variable_id"]
            for p in model.nodes[0].properties
        }
        assert variable_ids == {"SUBJID": "phv99999901.v1", "SEX": "phv99999902.v1"}

    def test_variable_id_present_on_variables_without_values(self, dict_path):
        """The metadata dict is built even when there are no value labels."""
        _, model = parse_dbgap_table(dict_path)
        subjid = next(p for p in model.nodes[0].properties if p.name == "SUBJID")
        assert subjid.values is None
        assert subjid.additional_metadata == {"variable_id": "phv99999901.v1"}

    def test_variable_id_and_value_labels_share_the_metadata(self, dict_path):
        """Adding the variable_id leaves the value labels in place beside it."""
        _, model = parse_dbgap_table(dict_path)
        sex = next(p for p in model.nodes[0].properties if p.name == "SEX")
        assert sex.additional_metadata == {
            "value_labels": ["1=Male", "2=Female"],
            "variable_id": "phv99999902.v1",
        }

    def test_variable_without_an_id_has_no_metadata_variable_id(self, tmp_path):
        """Without an id there is no variable_id key.

        A variable with values keeps only its value labels; one with neither an
        id nor values has nothing to record, so its additional_metadata is None.
        """
        path = tmp_path / "pht999998.v1_data_dict.xml"
        path.write_text(
            '<data_table id="pht999998.v1" study_id="phs999999">'
            "<variable><name>NO_VALUES</name><description>No id, no values</description></variable>"
            "<variable><name>WITH_VALUES</name><description>No id</description>"
            '<value code="1">Yes</value><value code="0">No</value></variable>'
            "</data_table>"
        )
        _, model = parse_dbgap_table(str(path))
        without_values, with_values = model.nodes[0].properties
        assert without_values.additional_metadata is None
        assert with_values.additional_metadata == {"value_labels": ["1=Yes", "0=No"]}

    def test_same_name_in_two_tables_is_two_variables(self, tmp_path):
        """A name is unique only within its table; the variable_id tells two apart."""
        variable_names, variable_ids = [], []
        for table, variable_id in (
            ("pht999998.v1", "phv99999801.v1"),
            ("pht999997.v1", "phv99999701.v1"),
        ):
            path = tmp_path / f"{table}_data_dict.xml"
            path.write_text(
                f'<data_table id="{table}" study_id="phs999999">'
                f'<variable id="{variable_id}"><name>SUBJID</name>'
                "<description>Subject identifier</description></variable>"
                "</data_table>"
            )
            _, model = parse_dbgap_table(str(path))
            variable = model.nodes[0].properties[0]
            variable_names.append(variable.name)
            variable_ids.append(variable.additional_metadata["variable_id"])
        assert variable_names == ["SUBJID", "SUBJID"]
        assert variable_ids == ["phv99999801.v1", "phv99999701.v1"]

    def test_var_report_sets_type(self, dict_path, report_path):
        """The var report's "encoded value" type maps to string/encoded."""
        _, model = parse_dbgap_table(dict_path, report_path)
        sex = next(p for p in model.nodes[0].properties if p.name == "SEX")
        assert sex.type == "string/encoded"

    def test_missing_report_defaults_to_string(self, dict_path):
        """Without a var report every property gets the string type."""
        _, model = parse_dbgap_table(dict_path, report_path=None)
        for prop in model.nodes[0].properties:
            assert prop.type == "string"


class TestFindStudyMetadataFiles:
    def test_raises_when_missing(self, tmp_path):
        """A study without a metadata directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            find_study_metadata_files(str(tmp_path), "phs999000.v1.p1.c1")

    def test_finds_dict_and_report(self, tmp_path):
        """The data dict is listed and the var report is keyed by its pht accession."""
        study_dir = tmp_path / "phs999999.v1.p1.c1" / "metadata"
        study_dir.mkdir(parents=True)
        (study_dir / "pht999999_data_dict.xml").write_text("<x/>")
        (study_dir / "pht999999_var_report.xml").write_text("<x/>")

        path, dicts, reports = find_study_metadata_files(
            str(tmp_path), "phs999999.v1.p1.c1"
        )
        assert len(dicts) == 1
        assert "pht999999" in reports

    def test_returns_correct_path(self, tmp_path):
        """The returned path is the study's metadata subdirectory."""
        study_dir = tmp_path / "phs999999.v1.p1.c1" / "metadata"
        study_dir.mkdir(parents=True)
        (study_dir / "pht001_data_dict.xml").write_text("<x/>")

        path, _, _ = find_study_metadata_files(str(tmp_path), "phs999999.v1.p1.c1")
        assert path == str(study_dir)


def make_suggestion(
    slot_key,
    similarity,
    target_description,
    prompt_variant,
    value_labels=None,
    variable_id=None,
):
    """Build one suggestion as MultiPromptSimilaritySearch would emit it."""
    target_node, target_property = slot_key.rsplit(".", 1)
    return SingleHarmonizationSuggestion(
        source_node="subject",
        source_property="age",
        source_description="Age at enrollment",
        source_additional_metadata={
            "type": "integer",
            "value_labels": value_labels or [],
            "variable_id": variable_id,
        },
        target_node=target_node,
        target_property=target_property,
        target_description=target_description,
        target_additional_metadata={"prompt_variant": prompt_variant},
        similarity=similarity,
    )


class TestBuildMappingRows:
    @pytest.fixture
    def suggestions(self):
        """Return one variable's two suggestions, best first, from variants A and B."""
        return [
            make_suggestion("target.slot_a", 0.9, "desc a", "A"),
            make_suggestion("target.slot_b", 0.7, "desc b", "B"),
        ]

    def test_row_count_matches_suggestions(self, suggestions):
        """Each suggestion becomes one row."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert len(rows) == 2

    def test_ranks_are_sequential(self, suggestions):
        """Ranks run from 1 in the order the suggestions are given."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert [r["rank"] for r in rows] == [1, 2]

    def test_no_suggestions_yields_no_rows(self):
        """An empty suggestion list yields no rows."""
        assert build_mapping_rows([], {}, "phs999999") == []

    def test_slot_values_lookup_used(self):
        """Target Values is looked up by the suggested target's node.property key."""
        suggestions = [make_suggestion("target.slot_a", 0.9, "d", "A")]
        rows = build_mapping_rows(
            suggestions, {"target.slot_a": "Yes, No"}, "phs999999"
        )
        assert rows[0]["Target Values"] == "Yes, No"

    def test_study_id_in_rows(self, suggestions):
        """Every row carries the study id passed in."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert all(r["study_id"] == "phs999999" for r in rows)

    def test_original_node_property_format(self, suggestions):
        """Original Node.Property is the source node and property joined by a dot."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert rows[0]["Original Node.Property"] == "subject.age"

    def test_suggested_target_reassembles_slot_key(self, suggestions):
        """The suggested target rejoins target node and property into the slot key."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert rows[0]["Suggested Target Node.Property"] == "target.slot_a"

    def test_prompt_variant_recorded(self, suggestions):
        """Each row records the prompt variant of its suggestion."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert [r["prompt_variant"] for r in rows] == ["A", "B"]

    def test_value_labels_become_original_values(self):
        """Source value labels are joined with VALUE_SEPARATOR into Original Values."""
        suggestions = [
            make_suggestion(
                "target.slot_a", 0.9, "d", "A", value_labels=["1=Male", "2=Female"]
            )
        ]
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert rows[0]["Original Values"] == VALUE_SEPARATOR.join(
            ["1=Male", "2=Female"]
        )

    def test_variable_id_carried_into_rows(self):
        """Each row names exactly one source variable, by its variable_id."""
        suggestions = [
            make_suggestion(
                "target.slot_a", 0.9, "d", "A", variable_id="phv99999903.v1"
            ),
            make_suggestion(
                "target.slot_b", 0.7, "d", "B", variable_id="phv99999903.v1"
            ),
        ]
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert [r["source_variable_id"] for r in rows] == ["phv99999903.v1"] * 2

    def test_missing_variable_id_is_an_empty_cell(self, suggestions):
        """A source with no recorded variable_id still writes a complete row."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert all(r["source_variable_id"] == "" for r in rows)

    def test_rows_match_the_csv_headers(self, suggestions):
        """Every row has exactly the CSV's columns, so DictWriter accepts it."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert all(list(r) == CSV_HEADERS for r in rows)


class TestSummarizeRank1Similarity:
    @pytest.fixture
    def rank1_df(self):
        """Return rank-1 rows for four variables, two of them at or above 0.75."""
        return pd.DataFrame(
            {
                "Similarity": [0.85, 0.72, 0.90, 0.60],
                "Suggested Target Node.Property": ["a.x", "b.y", "a.x", "c.z"],
            }
        )

    def test_variable_count(self, rank1_df):
        """Variables counts the rank-1 rows."""
        result = summarize_rank1_similarity(rank1_df)
        assert result["Variables"] == 4

    def test_top_target_is_highest_similarity(self, rank1_df):
        """The top target is the target of the highest-similarity row."""
        result = summarize_rank1_similarity(rank1_df)
        assert result["Top bdchm target"] == "a.x"

    def test_strong_match_percentage(self, rank1_df):
        """The strong column is the percentage of rows at or above 0.75."""
        result = summarize_rank1_similarity(rank1_df)
        assert result["≥0.75 (strong)"] == "50%"

    def test_mean_and_median_rounded(self, rank1_df):
        """Mean and median similarity are returned as floats."""
        result = summarize_rank1_similarity(rank1_df)
        assert isinstance(result["Mean sim (rank 1)"], float)
        assert isinstance(result["Median sim"], float)
