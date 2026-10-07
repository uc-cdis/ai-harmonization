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
    p = tmp_path / "pht999999.v1_data_dict.xml"
    p.write_text(DATA_DICT_XML)
    return str(p)


@pytest.fixture
def report_path(tmp_path):
    p = tmp_path / "pht999999.v1_var_report.xml"
    p.write_text(VAR_REPORT_XML)
    return str(p)


class TestParseDbgapTable:
    def test_returns_table_id_and_model(self, dict_path):
        table_id, model = parse_dbgap_table(dict_path)
        assert table_id == "pht999999.v1.p1"
        assert len(model.nodes) == 1

    def test_property_names(self, dict_path):
        _, model = parse_dbgap_table(dict_path)
        names = [p.name for p in model.nodes[0].properties]
        assert names == ["SUBJID", "SEX"]

    def test_enum_values_populated(self, dict_path):
        _, model = parse_dbgap_table(dict_path)
        sex = next(p for p in model.nodes[0].properties if p.name == "SEX")
        assert sex.values == ["Male", "Female"]
        assert "1=Male" in sex.additional_metadata["value_labels"]

    def test_variable_id_is_captured(self, dict_path):
        """Each variable carries its accession verbatim, version suffix included."""
        _, model = parse_dbgap_table(dict_path)
        accessions = {
            p.name: p.additional_metadata["variable_id"]
            for p in model.nodes[0].properties
        }
        assert accessions == {"SUBJID": "phv99999901.v1", "SEX": "phv99999902.v1"}

    def test_accession_present_on_variables_without_values(self, dict_path):
        """The metadata dict is built even when there are no value labels."""
        _, model = parse_dbgap_table(dict_path)
        subjid = next(p for p in model.nodes[0].properties if p.name == "SUBJID")
        assert subjid.values is None
        assert subjid.additional_metadata == {"variable_id": "phv99999901.v1"}

    def test_accession_and_value_labels_share_the_metadata(self, dict_path):
        """Adding the accession leaves the value labels in place beside it."""
        _, model = parse_dbgap_table(dict_path)
        sex = next(p for p in model.nodes[0].properties if p.name == "SEX")
        assert sex.additional_metadata == {
            "value_labels": ["1=Male", "2=Female"],
            "variable_id": "phv99999902.v1",
        }

    def test_variable_without_an_id_has_no_accession(self, tmp_path):
        """Without an id there is no accession key.

        CODED keeps only its value labels; BARE, with neither an id nor
        values, has nothing to record, so its additional_metadata is None.
        """
        p = tmp_path / "pht999998.v1_data_dict.xml"
        p.write_text(
            '<data_table id="pht999998.v1" study_id="phs999999">'
            "<variable><name>BARE</name><description>No id, no values</description></variable>"
            "<variable><name>CODED</name><description>No id</description>"
            '<value code="1">Yes</value><value code="0">No</value></variable>'
            "</data_table>"
        )
        _, model = parse_dbgap_table(str(p))
        bare, coded = model.nodes[0].properties
        assert bare.additional_metadata is None
        assert coded.additional_metadata == {"value_labels": ["1=Yes", "0=No"]}

    def test_same_name_in_two_tables_is_two_variables(self, tmp_path):
        """A name is unique only within its table; the accession tells two apart."""
        found = []
        for table, accession in (
            ("pht999998.v1", "phv99999801.v1"),
            ("pht999997.v1", "phv99999701.v1"),
        ):
            p = tmp_path / f"{table}_data_dict.xml"
            p.write_text(
                f'<data_table id="{table}" study_id="phs999999">'
                f'<variable id="{accession}"><name>SUBJID</name>'
                "<description>Subject identifier</description></variable>"
                "</data_table>"
            )
            _, model = parse_dbgap_table(str(p))
            (prop,) = model.nodes[0].properties
            found.append((prop.name, prop.additional_metadata["variable_id"]))
        (name_a, acc_a), (name_b, acc_b) = found
        assert name_a == name_b == "SUBJID"
        assert acc_a != acc_b

    def test_var_report_sets_type(self, dict_path, report_path):
        _, model = parse_dbgap_table(dict_path, report_path)
        sex = next(p for p in model.nodes[0].properties if p.name == "SEX")
        assert sex.type == "string/encoded"

    def test_missing_report_defaults_to_string(self, dict_path):
        _, model = parse_dbgap_table(dict_path, report_path=None)
        for prop in model.nodes[0].properties:
            assert prop.type == "string"


class TestFindStudyMetadataFiles:
    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_study_metadata_files(str(tmp_path), "phs999000.v1.p1.c1")

    def test_finds_dict_and_report(self, tmp_path):
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
        return [
            make_suggestion("target.slot_a", 0.9, "desc a", "A"),
            make_suggestion("target.slot_b", 0.7, "desc b", "B"),
        ]

    def test_row_count_matches_suggestions(self, suggestions):
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert len(rows) == 2

    def test_ranks_are_sequential(self, suggestions):
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert [r["rank"] for r in rows] == [1, 2]

    def test_no_suggestions_yields_no_rows(self):
        assert build_mapping_rows([], {}, "phs999999") == []

    def test_slot_values_lookup_used(self):
        suggestions = [make_suggestion("target.slot_a", 0.9, "d", "A")]
        rows = build_mapping_rows(
            suggestions, {"target.slot_a": "Yes, No"}, "phs999999"
        )
        assert rows[0]["Target Values"] == "Yes, No"

    def test_study_id_in_rows(self, suggestions):
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert all(r["study_id"] == "phs999999" for r in rows)

    def test_original_node_property_format(self, suggestions):
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert rows[0]["Original Node.Property"] == "subject.age"

    def test_suggested_target_reassembles_slot_key(self, suggestions):
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert rows[0]["Suggested Target Node.Property"] == "target.slot_a"

    def test_prompt_variant_recorded(self, suggestions):
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert [r["prompt_variant"] for r in rows] == ["A", "B"]

    def test_value_labels_become_original_values(self):
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
        """Each row names exactly one source variable, by its accession."""
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

    def test_missing_accession_is_an_empty_cell(self, suggestions):
        """A source with no recorded accession still writes a complete row."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert all(r["source_variable_id"] == "" for r in rows)

    def test_rows_match_the_csv_headers(self, suggestions):
        """Every row has exactly the CSV's columns, so DictWriter accepts it."""
        rows = build_mapping_rows(suggestions, {}, "phs999999")
        assert all(list(r) == CSV_HEADERS for r in rows)


class TestSummarizeRank1Similarity:
    @pytest.fixture
    def rank1_df(self):
        return pd.DataFrame(
            {
                "Similarity": [0.85, 0.72, 0.90, 0.60],
                "Suggested Target Node.Property": ["a.x", "b.y", "a.x", "c.z"],
            }
        )

    def test_variable_count(self, rank1_df):
        result = summarize_rank1_similarity(rank1_df)
        assert result["Variables"] == 4

    def test_top_target_is_highest_similarity(self, rank1_df):
        result = summarize_rank1_similarity(rank1_df)
        assert result["Top bdchm target"] == "a.x"

    def test_strong_match_percentage(self, rank1_df):
        result = summarize_rank1_similarity(rank1_df)
        assert result["≥0.75 (strong)"] == "50%"

    def test_mean_and_median_rounded(self, rank1_df):
        result = summarize_rank1_similarity(rank1_df)
        assert isinstance(result["Mean sim (rank 1)"], float)
        assert isinstance(result["Median sim"], float)
