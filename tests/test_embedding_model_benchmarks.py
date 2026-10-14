"""Tests for embedding_model.benchmarks -- building TOPMed and GDC, and manifests."""

import csv
import io
import json
import shutil
import subprocess
import tarfile
import textwrap
import zipfile

import pandas as pd
import pytest

from ai_harmonization.formatters import MAX_VALUES_IN_PROMPT, VALUE_SEPARATOR
from embedding_model import benchmarks, download
from embedding_model.__main__ import main
from embedding_model.benchmarks import gdc, topmed

COMMIT_FOLDER = "topmed-dcc-harmonized-phenotypes-abcd1234"
AGE_REFERENCE = "phs999999.v1.pht999999.v1.phv99999901.v1"
SEX_REFERENCE = "phs999999.v1.pht999999.v1.phv99999902.v1"

DATA_DICT_XML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <data_table id="pht999999.v1.p1" study_id="phs999999">
      <variable id="phv99999901.v1">
        <name>AGE01</name>
        <description>Age at exam</description>
      </variable>
      <variable id="phv99999902.v1">
        <name>SEX</name>
        <description>Sex of participant</description>
        <value code="1">Male</value>
        <value code="2">Female</value>
      </variable>
    </data_table>
    """
)
VAR_REPORT_XML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <data_table id="pht999999.v1.p1" study_id="phs999999">
      <variable id="phv99999901.v1" var_name="AGE01" calculated_type="integer"/>
      <variable id="phv99999902.v1" var_name="SEX" calculated_type="enum_integer"/>
    </data_table>
    """
)


def documentation_archive(documents):
    """A .tar.gz shaped like the TOPMed repository, holding the given files."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path, content in documents.items():
            data = (
                content if isinstance(content, bytes) else json.dumps(content).encode()
            )
            info = tarfile.TarInfo(f"{COMMIT_FOLDER}/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def harmonized(name, has_age=True, components=(), values=()):
    """A TOPMed documentation entry for one harmonized variable."""
    return {
        "name": name,
        "data_type": "encoded" if values else "decimal",
        "description": f"Harmonized {name}. ",
        "has_age_variable": has_age,
        "encoded_values": [{"code": str(i), "value": v} for i, v in enumerate(values)],
        "harmonization_units": [
            {"name": "StudyA", "component_study_variables": list(components)}
        ],
    }


def record(name, description, phs="phs999999"):
    """A resolved dbGaP variable, keyed like the mapping columns."""
    return {
        "source_study_id": phs,
        "source_table_id": "pht999999.v1.p1",
        "source_variable_id": "phv99999901.v1",
        "source_variable_name": name,
        "source_type": "integer",
        "source_description": description,
        "source_values": "",
    }


def ground_truth(rows):
    """A GDC-SM ground-truth table from (source, target) pairs."""
    return pd.DataFrame(
        rows, columns=["original_paper_variable_names", "GDC_format_variable_names"]
    )


class TestReadDocumentation:
    def test_variables_with_their_values_and_components(self):
        """Each harmonized variable is read with its coded values and cited components."""
        archive = documentation_archive(
            {
                "harmonized-variable-documentation/demographic/sex_1.json": harmonized(
                    "sex_1",
                    has_age=False,
                    components=[SEX_REFERENCE],
                    values=["Female", "Male"],
                ),
                "harmonized-variable-documentation/atherosclerosis/cac_1.json": (
                    harmonized("cac_1", components=[AGE_REFERENCE])
                ),
                "README.md": b"not documentation",
            }
        )
        variables = topmed.read_documentation(archive)
        assert [(v["dataset"], v["name"]) for v in variables] == [
            ("atherosclerosis", "cac_1"),
            ("demographic", "sex_1"),
        ]
        sex = variables[1]
        assert sex["values"] == ["Female", "Male"]
        assert sex["has_age"] is False
        assert sex["components"] == [
            ("phs999999", "1", "pht999999", "1", "phv99999902")
        ]

    def test_only_files_one_dataset_deep(self):
        """A file beside the dataset folders, or in a look-alike folder, is not read."""
        archive = documentation_archive(
            {
                "harmonized-variable-documentation/demographic/sex_1.json": harmonized(
                    "sex_1", has_age=False, components=[SEX_REFERENCE]
                ),
                "harmonized-variable-documentation/index.json": b"{}",
                "harmonized-variable-documentation-old/demographic/sex_1.json": b"{}",
            }
        )
        variables = topmed.read_documentation(archive)
        assert [(v["dataset"], v["name"]) for v in variables] == [
            ("demographic", "sex_1")
        ]


class TestReadDocumentationErrors:
    def test_a_malformed_reference_names_its_file(self):
        """A component that is not a phs.v.pht.v.phv accession fails with its file named."""
        archive = documentation_archive(
            {
                "harmonized-variable-documentation/demographic/sex_1.json": harmonized(
                    "sex_1", components=["phs999999.v1.phv99999902"]
                )
            }
        )
        with pytest.raises(ValueError, match="sex_1.json"):
            topmed.read_documentation(archive)


class TestBuilders:
    def test_topmed_reads_its_pin_from_the_manifest(self, monkeypatch, tmp_path):
        """The documentation is fetched at the manifest's commit and kept under it."""
        archive = documentation_archive(
            {
                "harmonized-variable-documentation/demographic/sex_1.json": harmonized(
                    "sex_1", has_age=False, components=[SEX_REFERENCE]
                )
            }
        )
        fetched = []

        def fake_cached(url, path):
            """Serve the archive and record where it was asked for."""
            fetched.append((url, path))
            return archive

        monkeypatch.setattr(topmed, "cached", fake_cached)
        monkeypatch.setattr(
            topmed,
            "resolve_components",
            lambda references, downloads: (
                {("phs999999", "1", "phv99999902"): record("SEX", "Sex")},
                [],
            ),
        )
        pin = {"name": topmed.DOCUMENTATION, "role": "mappings and targets",
               "location": "https://github.com/abcd/docs", "version": "abcd1234"}  # fmt: skip
        targets, mappings, sources = topmed.build(tmp_path, {"sources": [pin]})
        assert fetched == [
            (
                "https://codeload.github.com/abcd/docs/tar.gz/abcd1234",
                tmp_path / "abcd1234.tar.gz",
            )
        ]
        assert [row["expected_target"] for row in mappings] == ["demographic.sex_1"]
        assert sources == [pin]

    @pytest.mark.parametrize("builder", [topmed, gdc])
    def test_a_missing_pin_is_named(self, builder, tmp_path):
        """A manifest without the builder's pinned source fails with the source named."""
        with pytest.raises(KeyError, match="has no source named"):
            builder.build(tmp_path, {"sources": []})


class TestTargetRows:
    def test_phenotype_and_age_targets(self):
        """A variable with an age column gives a phenotype target and an age_at_ target."""
        variables = [
            {
                "dataset": "atherosclerosis",
                "name": "cac_1",
                "type": "decimal",
                "description": "CAC score",
                "values": [],
                "has_age": True,
            },
            {
                "dataset": "demographic",
                "name": "sex_1",
                "type": "encoded",
                "description": "Sex",
                "values": ["Female", "Male"],
                "has_age": False,
            },
        ]
        rows = topmed.target_rows(variables)
        assert [row["target_property"] for row in rows] == [
            "atherosclerosis.age_at_cac_1",
            "atherosclerosis.cac_1",
            "demographic.sex_1",
        ]
        assert rows[0]["target_description"] == "age at measurement of cac_1 (years)"
        assert rows[2]["target_values"] == f"Female{VALUE_SEPARATOR}Male"


class TestIsAge:
    @pytest.mark.parametrize(
        "description",
        [
            "Age when CAC score was obtained",
            "AGE AT EXAM 1 (VERIFIED AT EXAM 2)",
            "Age @ Time of Enrollment",
        ],
    )
    def test_ages(self, description):
        """'age' as a word in the description makes an age."""
        assert topmed.is_age(description)

    @pytest.mark.parametrize(
        "description",
        [
            "Tumor stage",
            "Number of cigarettes smoked per day, averaged over all years",
            "Dosage of the medication",
            "CT scan date",
        ],
    )
    def test_not_ages(self, description):
        """'age' inside another word, or a date, is not an age."""
        assert not topmed.is_age(description)


class TestCitedVariables:
    def test_wanted_variables_only(self, tmp_path):
        """Only the wanted variables are returned, with their dbGaP text."""
        dictionary = tmp_path / "phs999999.v1.pht999999.v1.Exam.data_dict.xml"
        dictionary.write_text(DATA_DICT_XML)
        report = tmp_path / "phs999999.v1.pht999999.v1.p1.Exam.var_report.xml"
        report.write_text(VAR_REPORT_XML)
        found = topmed.cited_variables(dictionary, report, "phs999999", {"phv99999902"})
        assert list(found) == ["phv99999902"]
        assert found["phv99999902"]["source_variable_name"] == "SEX"
        assert found["phv99999902"]["source_values"] == f"Male{VALUE_SEPARATOR}Female"
        assert found["phv99999902"]["source_variable_id"] == "phv99999902.v1"


class TestResolveComponents:
    @pytest.fixture
    def served(self, monkeypatch):
        """Serve a fictional dbGaP study with two releases and record what is fetched."""
        base = f"{topmed.DBGAP}/phs999999/"
        summaries = f"{base}phs999999.v1.p1/pheno_variable_summaries/"
        pages = {
            base: b'<a href="phs999999.v1.p1/"><a href="phs999999.v2.p1/">',
            summaries: (
                b'<a href="phs999999.v1.pht999999.v1.Exam.data_dict.xml">'
                b'<a href="phs999999.v1.pht999999.v1.p1.Exam.var_report_2011_02_07.xml">'
                b'<a href="phs999999.v1.pht999998.v1.Other.data_dict.xml">'
            ),
            summaries
            + "phs999999.v1.pht999999.v1.Exam.data_dict.xml": (DATA_DICT_XML.encode()),
            summaries
            + "phs999999.v1.pht999999.v1.p1.Exam.var_report_2011_02_07.xml": (
                VAR_REPORT_XML.encode()
            ),
        }
        fetched = []

        def fake_downloaded(url, path):
            """Serve a page from `pages` once, keeping it where the builder keeps it."""
            if not path.exists():
                fetched.append(url)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pages[url])
            return path

        monkeypatch.setattr(topmed, "downloaded", fake_downloaded)
        monkeypatch.setattr(
            topmed, "cached", lambda url, path: fake_downloaded(url, path).read_bytes()
        )
        return fetched

    def test_the_cited_release_and_only_the_cited_tables(self, served, tmp_path):
        """The cited release is used, only its cited table is fetched, dated names too."""
        found, sources = topmed.resolve_components(
            {("phs999999", "1", "pht999999", "1", "phv99999901")}, tmp_path
        )
        assert set(found) == {("phs999999", "1", "phv99999901")}
        assert [s["version"] for s in sources] == ["phs999999.v1.p1"]
        assert not any("pht999998" in url for url in served)
        assert any("var_report_2011_02_07" in url for url in served)

    def test_a_kept_study_listing_without_the_release_is_fetched_again(
        self, served, tmp_path
    ):
        """A study listing kept from before a release appeared is refreshed once."""
        stale = tmp_path / "phs999999" / "listing.html"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b'<a href="phs999999.v2.p1/">')
        found, _ = topmed.resolve_components(
            {("phs999999", "1", "pht999999", "1", "phv99999901")}, tmp_path
        )
        assert set(found) == {("phs999999", "1", "phv99999901")}
        assert served.count(f"{topmed.DBGAP}/phs999999/") == 1

    def test_a_missing_table_stops_the_build(self, served, tmp_path):
        """A cited table that the release does not publish is an error, not a gap."""
        with pytest.raises(LookupError, match="pht999997"):
            topmed.resolve_components(
                {("phs999999", "1", "pht999997", "1", "phv99999901")}, tmp_path
            )


class TestTopmedMappingRows:
    def variables(self):
        """Two harmonized variables citing the same age and their own values."""
        cac = ("phs999999", "1", "pht999999", "1", "phv99999903")
        height = ("phs999999", "1", "pht999999", "1", "phv99999904")
        age = ("phs999999", "1", "pht999999", "1", "phv99999901")
        return [
            {"dataset": "atherosclerosis", "name": "cac_1", "has_age": True,
             "components": [cac, age]},
            {"dataset": "anthropometry", "name": "height_1", "has_age": True,
             "components": [height, age]},
        ]  # fmt: skip

    def resolved(self):
        """The dbGaP text of the three variables."""
        return {
            ("phs999999", "1", "phv99999901"): record("AGE01", "Age at exam 1"),
            ("phs999999", "1", "phv99999903"): record("CAC", "Calcium score"),
            ("phs999999", "1", "phv99999904"): record("HT", "Standing height"),
        }

    def test_phenotypes_and_ages(self):
        """A phenotype maps to its variable, an age to the variable's age_at_ target."""
        rows = topmed.mapping_rows(self.variables(), self.resolved())
        by_name = {row["source_variable_name"]: row["expected_target"] for row in rows}
        assert by_name["CAC"] == "atherosclerosis.cac_1"
        assert by_name["HT"] == "anthropometry.height_1"

    def test_a_shared_age_is_one_row_with_all_its_targets(self):
        """An age cited by two units is one row carrying both age_at_ targets."""
        rows = topmed.mapping_rows(self.variables(), self.resolved())
        ages = [row for row in rows if row["source_variable_name"] == "AGE01"]
        assert len(ages) == 1
        assert ages[0]["expected_target"] == (
            "anthropometry.age_at_height_1;atherosclerosis.age_at_cac_1"
        )

    def test_no_age_column_means_phenotype(self):
        """Without an age column, even an age variable maps to the phenotype."""
        variables = [
            {"dataset": "demographic", "name": "subcohort_1", "has_age": False,
             "components": [("phs999999", "1", "pht999999", "1", "phv99999901")]},
        ]  # fmt: skip
        (row,) = topmed.mapping_rows(variables, self.resolved())
        assert row["expected_target"] == "demographic.subcohort_1"

    def test_the_latest_cited_release_gives_the_text(self):
        """A variable cited from two releases takes its text from the later one."""
        variables = [
            {"dataset": "atherosclerosis", "name": "cac_1", "has_age": True,
             "components": [("phs999999", "1", "pht999999", "1", "phv99999903"),
                            ("phs999999", "2", "pht999999", "1", "phv99999903")]},
        ]  # fmt: skip
        resolved = {
            ("phs999999", "1", "phv99999903"): record("CAC", "Calcium score, v1"),
            ("phs999999", "2", "phv99999903"): record("CAC", "Calcium score, v2"),
        }
        (row,) = topmed.mapping_rows(variables, resolved)
        assert row["source_description"] == "Calcium score, v2"


class TestDictionaryTargets:
    DICTIONARY = {
        "_definitions": {"properties": {"ignored": {"type": "string"}}},
        "demographic": {
            "description": "Demographic data.",
            "systemProperties": ["id", "state"],
            "properties": {
                "id": {"type": "string", "description": "Record identifier."},
                "state": {"common": {"description": "The record's state."}},
                "type": {"type": "string"},
                "race": {
                    "type": ["string", "null"],
                    "description": "Self-reported race.",
                    "enum": ["asian", "white"],
                },
                "age_at_index": {
                    "description": "Age at index.",
                    "oneOf": [{"type": "integer"}, {"type": "null"}],
                },
                "vital_status": {
                    "description": "Alive or dead.",
                    "oneOf": [{"enum": ["Alive", "Dead"]}, {"type": "null"}],
                },
                "ethnicity": {"common": {"description": "Self-reported ethnicity."}},
                "sites": {
                    "type": "array",
                    "description": "Sites involved.",
                    "items": {"enum": ["Lung", "Liver"]},
                },
                "cases": {"anyOf": [{"items": {}}]},
            },
        },
        "_terms": ["not", "a", "node"],
        "metaschema": {
            "description": "",
            "properties": {"category": {"type": "string"}},
        },
        "root": {
            "description": "",
            "properties": {"schema_version": {"type": "string"}},
        },
    }

    def rows(self):
        """The targets of the fictional dictionary, by key."""
        return {
            row["target_property"]: row
            for row in gdc.dictionary_targets(self.DICTIONARY)
        }

    def test_only_data_properties_are_targets(self):
        """Links, system properties, the node's type and the meta-entries are not targets."""
        assert sorted(self.rows()) == [
            "demographic.age_at_index",
            "demographic.ethnicity",
            "demographic.race",
            "demographic.sites",
            "demographic.vital_status",
        ]

    def test_text_as_the_shared_parser_reads_it(self):
        """Types, values and descriptions come from from_gdc_model, nested ones included."""
        rows = self.rows()
        assert rows["demographic.race"]["target_type"] == "string"
        assert rows["demographic.age_at_index"]["target_type"] == "integer"
        assert rows["demographic.vital_status"]["target_type"] == "enum"
        assert rows["demographic.vital_status"]["target_values"] == (
            f"Alive{VALUE_SEPARATOR}Dead"
        )
        assert rows["demographic.ethnicity"]["target_description"] == (
            "Self-reported ethnicity."
        )
        assert (
            rows["demographic.sites"]["target_values"] == f"Lung{VALUE_SEPARATOR}Liver"
        )


class TestReadGroundTruth:
    def test_one_table_per_paper(self):
        """Ground-truth CSVs are read by paper; other files in the archive are not."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipped:
            zipped.writestr(
                "gdc-sm-data/ground-truth/Abcd.csv",
                "original_paper_variable_names,GDC_format_variable_names\nAge,age\n",
            )
            zipped.writestr("gdc-sm-data/target-tables/flat.csv", "age\n")
        truth = gdc.read_ground_truth(buffer.getvalue())
        assert list(truth) == ["Abcd"]
        assert truth["Abcd"].iloc[0].tolist() == ["Age", "age"]


def spreadsheet(columns, sheet="Clinical"):
    """An .xlsx file holding one sheet built from the given columns."""
    buffer = io.BytesIO()
    pd.DataFrame(columns).to_excel(buffer, sheet_name=sheet, index=False)
    return buffer.getvalue()


class TestReadSourceTable:
    def test_mcdermott_column_loses_its_tab(self):
        """McDermott's column name ending in a tab is renamed; other names are kept as published."""
        tabbed = "Age in Months at Time of Tissue Procurement \t"
        data = spreadsheet({tabbed: [820], "Site ": ["Ovary"]})
        assert list(gdc.read_source_table(data, "Clinical", "McDermott").columns) == [
            "Age in Months at Time of Tissue Procurement",
            "Site ",
        ]
        assert tabbed in gdc.read_source_table(data, "Clinical", "Abcd").columns

    def test_column_types_are_those_of_its_csv(self):
        """Numbers stored as text in the sheet read as numbers, as GDC-SM's CSV tables do."""
        data = spreadsheet({"Age": ["61", "70"], "Site": ["Ovary", "Lung"]})
        table = gdc.read_source_table(data, "Clinical", "Abcd")
        assert table["Age"].tolist() == [61, 70]
        assert table["Site"].tolist() == ["Ovary", "Lung"]

    def test_vasaikar_loses_its_label_and_count_rows(self):
        """As GDC-SM's script does, Vasaikar's first (type labels) and last (counts) rows go."""
        data = spreadsheet(
            {"Gender": ["BIN", "Male", "Female", 2], "Age": ["CON", 61, 70, 2]}
        )
        table = gdc.read_source_table(data, "Clinical", "Vasaikar")
        assert table["Gender"].tolist() == ["Male", "Female"]
        assert pd.api.types.is_numeric_dtype(table["Age"])

    def test_clark_and_wang_values_repaired(self):
        """Clark's 'Whte' typo and Wang's secondhand-smoke answers are fixed as GDC-SM fixes them."""
        clark = spreadsheet({"Ethnicity_Self_Identify": ["Whte", "Asian"]})
        assert gdc.read_source_table(clark, "Clinical", "Clark")[
            "Ethnicity_Self_Identify"
        ].tolist() == ["White", "Asian"]
        wang = spreadsheet(
            {"secondhand_smoke_exposure": [
                "No or minimal exposure to secondhand smoke",
                "Exposed in current household",
            ]}
        )  # fmt: skip
        assert gdc.read_source_table(wang, "Clinical", "Wang")[
            "secondhand_smoke_exposure"
        ].tolist() == ["No", "Yes"]


class TestGdcMappingRows:
    TARGETS = [
        {"target_property": key}
        for key in (
            "case.primary_site",
            "clinical.race",
            "demographic.race",
            "project.primary_site",
        )
    ]

    def test_property_in_several_nodes_is_accepted_in_each(self):
        """A property two nodes carry is correct in both; the description is empty."""
        truth = {"Abcd": ground_truth([("Race", "race")])}
        tables = {"Abcd": pd.DataFrame({"Race": ["Asian", "White", None]})}
        (row,) = gdc.mapping_rows(truth, tables, self.TARGETS)
        assert row["expected_target"] == "clinical.race;demographic.race"
        assert row["source_type"] == "string"
        assert row["source_description"] == ""
        assert row["source_values"] == f"Asian{VALUE_SEPARATOR}White"

    def test_not_accepted_nodes_are_left_out(self):
        """A node that gives the property another meaning is not an answer."""
        truth = {"Abcd": ground_truth([("Tumor site", "primary_site")])}
        tables = {"Abcd": pd.DataFrame({"Tumor site": ["Ovary"]})}
        (row,) = gdc.mapping_rows(truth, tables, self.TARGETS)
        assert row["expected_target"] == "case.primary_site"

    def test_several_published_targets_merge_into_one_row(self):
        """Two ground-truth pairs for one variable give one row with both targets."""
        truth = {"Abcd": ground_truth([("Site", "primary_site"), ("Site", "race")])}
        tables = {"Abcd": pd.DataFrame({"Site": ["Ovary"]})}
        (row,) = gdc.mapping_rows(truth, tables, self.TARGETS)
        assert row["expected_target"] == (
            "case.primary_site;clinical.race;demographic.race"
        )

    def test_integers_with_blanks_read_as_integers(self):
        """An integer column with a blank keeps integer values, 64 rather than 64.0."""
        truth = {"Abcd": ground_truth([("Age", "race")])}
        tables = {"Abcd": pd.DataFrame({"Age": [64, None, 58]})}
        (row,) = gdc.mapping_rows(truth, tables, self.TARGETS)
        assert row["source_values"] == f"64{VALUE_SEPARATOR}58"

    def test_numeric_column_keeps_every_value(self):
        """A numeric column is a number, and every distinct value is kept."""
        truth = {"Abcd": ground_truth([("Age", "race")])}
        tables = {"Abcd": pd.DataFrame({"Age": range(MAX_VALUES_IN_PROMPT + 5)})}
        (row,) = gdc.mapping_rows(truth, tables, self.TARGETS)
        assert row["source_type"] == "number"
        assert (
            len(row["source_values"].split(VALUE_SEPARATOR)) == MAX_VALUES_IN_PROMPT + 5
        )

    def test_target_not_in_the_dictionary_drops_the_pair(self, capsys):
        """A published target the dictionary does not carry gives no answer, and is named."""
        truth = {"Abcd": ground_truth([("Stage", "retired_property")])}
        tables = {"Abcd": pd.DataFrame({"Stage": ["I"]})}
        assert gdc.mapping_rows(truth, tables, self.TARGETS) == []
        assert "Abcd: Stage -> retired_property" in capsys.readouterr().out

    def test_variable_missing_from_its_table_raises(self):
        """A mapped variable that is not a column of its table fails the build."""
        truth = {"Abcd": ground_truth([("Missing", "race")])}
        tables = {"Abcd": pd.DataFrame({"Race": ["Asian"]})}
        with pytest.raises(ValueError, match="Abcd: Missing"):
            gdc.mapping_rows(truth, tables, self.TARGETS)


def mapping(name, target):
    """A mapping row with every column, for the manifest tests."""
    row = dict.fromkeys(benchmarks.MAPPING_COLUMNS, "")
    return {**row, "source_variable_name": name, "expected_target": target}


TARGET = dict.fromkeys(benchmarks.TARGET_COLUMNS, "")
PIN = {"role": "mappings", "name": "Fictional source", "version": "1"}


class TestWriteBenchmark:
    def test_files_and_manifest(self, tmp_path):
        """Both files are written with their columns, and the manifest describes them."""
        manifest = benchmarks.write_benchmark(
            tmp_path, "abcd", "1", [TARGET], [mapping("AGE", "a.age")], [PIN]
        )
        with open(tmp_path / "abcd_mappings.csv", newline="") as handle:
            assert next(csv.reader(handle)) == benchmarks.MAPPING_COLUMNS
        assert manifest["files"]["abcd_mappings.csv"]["rows"] == 1
        assert manifest["files"]["abcd_targets.csv"]["sha256"] == download.sha256(
            (tmp_path / "abcd_targets.csv").read_bytes()
        )
        assert json.loads((tmp_path / "abcd_manifest.json").read_text()) == manifest


class TestIdentities:
    def test_files_then_sources_by_sha256_or_version(self):
        """Files come first; a source is its SHA-256, else its version."""
        manifest = {
            "files": {"abcd_mappings.csv": {"rows": 1, "sha256": "aa"}},
            "sources": [
                {**PIN, "sha256": "cc", "version": "2"},
                {"role": "source text", "name": "Abcd release", "version": "v1"},
            ],
        }
        assert benchmarks.identities(manifest) == {
            "abcd_mappings.csv": {"rows": 1, "sha256": "aa"},
            "mappings 'Fictional source'": "cc",
            "source text 'Abcd release'": "v1",
        }


class TestDifferences:
    MANIFEST = {
        "version": "1",
        "built": "2026-01-01",
        "files": {"abcd_mappings.csv": {"rows": 1, "sha256": "aa"}},
        "sources": [{**PIN, "sha256": "cc", "fetched": "2026-01-01"}],
    }

    def test_dates_and_release_labels_do_not_count(self):
        """A rebuild on another day, from a source with the same SHA-256, matches."""
        later = {
            **self.MANIFEST,
            "built": "2026-02-01",
            "sources": [
                {**PIN, "sha256": "cc", "version": "2", "fetched": "2026-02-01"}
            ],
        }
        assert benchmarks.differences(self.MANIFEST, later) == []

    def test_changed_file_and_source_are_named(self):
        """A file and a source with another SHA-256 are each named."""
        changed = {
            **self.MANIFEST,
            "files": {"abcd_mappings.csv": {"rows": 1, "sha256": "bb"}},
            "sources": [{**PIN, "sha256": "dd"}],
        }
        found = benchmarks.differences(self.MANIFEST, changed)
        assert len(found) == 2
        assert found[0].startswith("abcd_mappings.csv")
        assert "Fictional source" in found[1]


class TestBuild:
    @pytest.fixture
    def setup(self, monkeypatch, tmp_path):
        """A fictional builder for 'topmed', with its pins-only manifest in a temp folder."""
        built = {"mappings": [mapping("AGE", "a.age")]}
        monkeypatch.setattr(
            topmed,
            "build",
            lambda downloads, manifest: (
                [TARGET],
                built["mappings"],
                manifest["sources"],
            ),
        )
        manifests = tmp_path / "manifests"
        manifests.mkdir()
        (manifests / "topmed.json").write_text(
            json.dumps({"version": "1", "files": {}, "sources": [PIN]})
        )
        monkeypatch.setattr(benchmarks, "MANIFESTS", manifests)
        monkeypatch.setattr(benchmarks, "committed_version", lambda name: None)
        return built, manifests / "topmed.json", tmp_path / "inputs" / "benchmarks"

    def test_a_first_build_needs_accepting(self, setup, tmp_path):
        """Against a pins-only manifest, a build differs until accepted."""
        _, manifest, working = setup
        assert benchmarks.build(["topmed"], tmp_path / "inputs") == 1
        assert not (working / "topmed_mappings.csv").exists()
        assert (
            benchmarks.build(["topmed"], tmp_path / "inputs", update_manifest=True) == 0
        )
        assert json.loads(manifest.read_text())["files"]
        assert benchmarks.build(["topmed"], tmp_path / "inputs") == 0

    def test_a_differing_build_leaves_the_working_files(self, setup, tmp_path):
        """A build that differs exits 1 and leaves the files and manifest as they were."""
        built, manifest, working = setup
        benchmarks.build(["topmed"], tmp_path / "inputs", update_manifest=True)
        before = (working / "topmed_mappings.csv").read_bytes(), manifest.read_text()
        built["mappings"] = [mapping("AGE", "a.age_at_exam")]
        assert benchmarks.build(["topmed"], tmp_path / "inputs") == 1
        assert (working / "topmed_mappings.csv").read_bytes() == before[0]
        assert manifest.read_text() == before[1]

    def test_a_committed_version_is_not_redefined(self, setup, monkeypatch, tmp_path):
        """A differing build is refused under a committed version, accepted under a new one."""
        built, manifest, working = setup
        benchmarks.build(["topmed"], tmp_path / "inputs", update_manifest=True)
        monkeypatch.setattr(benchmarks, "committed_version", lambda name: "1")
        before = (working / "topmed_mappings.csv").read_bytes(), manifest.read_text()
        built["mappings"] = [mapping("AGE", "a.age_at_exam")]
        assert (
            benchmarks.build(["topmed"], tmp_path / "inputs", update_manifest=True) == 1
        )
        assert (working / "topmed_mappings.csv").read_bytes() == before[0]
        assert manifest.read_text() == before[1]
        manifest.write_text(json.dumps({**json.loads(before[1]), "version": "2"}))
        assert (
            benchmarks.build(["topmed"], tmp_path / "inputs", update_manifest=True) == 0
        )
        assert json.loads(manifest.read_text())["version"] == "2"

    def test_no_answer_from_git_refuses(self, setup, monkeypatch, tmp_path):
        """When git cannot say whether the version is committed, a differing build is refused."""
        built, manifest, working = setup
        benchmarks.build(["topmed"], tmp_path / "inputs", update_manifest=True)

        def no_answer(name):
            """Fail as committed_version does outside a git checkout."""
            raise RuntimeError("not a git repository")

        monkeypatch.setattr(benchmarks, "committed_version", no_answer)
        before = manifest.read_text()
        built["mappings"] = [mapping("AGE", "a.age_at_exam")]
        assert (
            benchmarks.build(["topmed"], tmp_path / "inputs", update_manifest=True) == 1
        )
        assert manifest.read_text() == before


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
class TestCommittedVersion:
    def git(self, folder, *args):
        """Run git in a folder, as a fictional committer."""
        subprocess.run(
            ["git", "-c", "user.name=Abcd", "-c", "user.email=abcd@example.org",
             "-C", str(folder), *args],
            check=True, capture_output=True,
        )  # fmt: skip

    def test_the_version_at_head(self, monkeypatch, tmp_path):
        """The committed version is read from HEAD, not from the file on disk."""
        manifests = tmp_path / "manifests"
        manifests.mkdir()
        (manifests / "abcd.json").write_text(json.dumps({"version": "1"}))
        self.git(tmp_path, "init", "-q")
        self.git(tmp_path, "add", ".")
        self.git(tmp_path, "commit", "-q", "-m", "abcd")
        (manifests / "abcd.json").write_text(json.dumps({"version": "2"}))
        monkeypatch.setattr(benchmarks, "MANIFESTS", manifests)
        assert benchmarks.committed_version("abcd") == "1"
        assert benchmarks.committed_version("efgh") is None

    def test_outside_a_checkout_is_an_error(self, monkeypatch, tmp_path):
        """Without a git checkout to ask, the answer is an error, not 'not committed'."""
        monkeypatch.setattr(benchmarks, "MANIFESTS", tmp_path)
        with pytest.raises(RuntimeError):
            benchmarks.committed_version("abcd")

    def test_git_missing_is_an_error(self, monkeypatch, tmp_path):
        """When git cannot run at all, the answer is an error."""

        def missing(*args, **kwargs):
            """Fail as subprocess.run does when git is not installed."""
            raise FileNotFoundError("git")

        monkeypatch.setattr(benchmarks.subprocess, "run", missing)
        with pytest.raises(RuntimeError, match="git could not run"):
            benchmarks.committed_version("abcd")


class TestCached:
    def test_downloads_once_and_leaves_no_partial_file(self, monkeypatch, tmp_path):
        """The first call fetches and keeps the file; later calls read it back."""
        calls = []
        monkeypatch.setattr(download, "fetch", lambda url: calls.append(url) or b"data")
        path = tmp_path / "sub" / "file.bin"
        assert download.cached("https://example.org/f", path) == b"data"
        assert download.cached("https://example.org/f", path) == b"data"
        assert calls == ["https://example.org/f"]
        assert list(path.parent.iterdir()) == [path]

    def test_an_html_answer_is_not_kept(self, monkeypatch, tmp_path):
        """An HTML page served for a spreadsheet raises, and nothing is kept."""
        monkeypatch.setattr(
            download, "fetch", lambda url: b"<!DOCTYPE html><html>Checking...</html>"
        )
        path = tmp_path / "table.xlsx"
        with pytest.raises(ValueError, match="HTML page"):
            download.cached("https://example.org/table.xlsx", path)
        assert list(tmp_path.iterdir()) == []

    def test_a_listing_page_is_kept(self, monkeypatch, tmp_path):
        """A folder listing is HTML by nature, so it is kept."""
        monkeypatch.setattr(download, "fetch", lambda url: b"<html><a href='x'></a>")
        path = tmp_path / "listing.html"
        assert download.downloaded("https://example.org/", path) == path
        assert path.exists()


class TestCommandLine:
    def test_build_needs_a_benchmark(self):
        """`benchmarks build` with no benchmark named is a usage error."""
        with pytest.raises(SystemExit) as raised:
            main(["benchmarks", "build"])
        assert raised.value.code == 2
