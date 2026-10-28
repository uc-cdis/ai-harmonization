"""Tests for embedding_model.sources: each reader on small fictional files."""

import hashlib
import io
import json
import zipfile
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
from defusedxml import DefusedXmlException

from embedding_model import download, sources
from embedding_model.__main__ import main
from embedding_model.sources import (
    cadsr,
    cde,
    cdisc,
    common,
    loinc,
    ncit,
    nhanes,
    obo,
    owl,
    phenx,
)

NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
OBO = "http://purl.obolibrary.org/obo/"


def rdf(body, header=""):
    """An RDF/XML document with the namespaces the readers use."""
    return f"""<?xml version="1.0"?>
<rdf:RDF xmlns="{NCIT}"
     xmlns:obo="{OBO}"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#"
     xmlns:dcterms="http://purl.org/dc/terms/">
    <owl:Ontology rdf:about="{OBO}abcd.owl">{header}</owl:Ontology>
{body}
</rdf:RDF>""".encode()


def served(pages):
    """A stand-in for ``downloaded`` that writes each URL's page, or raises its error."""

    def downloaded(url, path, check=None, **request):
        """Keep the page served for a URL, unless kept already, as a real download does."""
        if path.exists():
            return path
        page = pages[url]
        if isinstance(page, Exception):
            raise page
        if check is not None:
            check(page)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(page)
        return path

    return downloaded


def http_error(status):
    """The error ``requests`` raises for an answer with an error status."""
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


def axiom(source, prop, target, annotations):
    """An owl:Axiom annotating one statement."""
    return f"""    <owl:Axiom>
        <owl:annotatedSource rdf:resource="{source}"/>
        <owl:annotatedProperty rdf:resource="{prop}"/>
        <owl:annotatedTarget>{target}</owl:annotatedTarget>
        {annotations}
    </owl:Axiom>"""


class TestOwl:
    """Reading statements and axioms from an OWL file."""

    def test_statements_keep_exact_text_and_defined_parents(self):
        """Text is kept as written, and an intersection's named members are parents."""
        document = rdf(
            f"""    <owl:Class rdf:about="{NCIT}C900002">
        <P90>Abcd </P90>
        <rdfs:subClassOf rdf:resource="{NCIT}C900001"/>
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NCIT}C900003"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NCIT}R1"/>
                        <owl:someValuesFrom rdf:resource="{NCIT}C900004"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
    </owl:Class>"""
        )
        found = owl.read_owl(io.BytesIO(document))
        values = owl.grouped(found["classes"][NCIT + "C900002"])
        assert values["{" + NCIT + "}P90"] == ["Abcd "]
        assert values[owl.DEFINED_PARENT] == [NCIT + "C900003"]
        assert values[owl.RDFS + "subClassOf"] == [NCIT + "C900001"]

    def test_an_entity_declaration_is_refused(self):
        """A file that declares an entity is refused, as an external reference could be."""
        document = rdf("").replace(
            b'<?xml version="1.0"?>',
            b'<?xml version="1.0"?>\n<!DOCTYPE rdf:RDF [<!ENTITY abcd SYSTEM "file:///abcd">]>',
        )
        with pytest.raises(DefusedXmlException):
            owl.read_owl(io.BytesIO(document))

    def test_axioms_keep_every_annotation(self):
        """Two axioms on one statement, one with two values, are all kept."""
        source, prop = NCIT + "C900001", NCIT + "P90"
        document = rdf(
            "\n".join(
                [
                    f'    <owl:Class rdf:about="{source}"><P90>Abcd</P90></owl:Class>',
                    axiom(source, prop, "Abcd", "<P383>PT</P383><P384>NCI</P384>"),
                    axiom(source, prop, "Abcd", "<P383>SY</P383><P383>AB</P383>"),
                ]
            )
        )
        found = owl.read_owl(io.BytesIO(document))
        axioms = found["axioms"][(source, prop, "Abcd")]
        assert [a["{" + NCIT + "}P383"] for a in axioms] == [["PT"], ["SY", "AB"]]


def ncit_document():
    """A fictional NCIt release: properties, a value list and four concepts."""
    ns = NCIT

    def prop(code, name):
        """An annotation property of NCIt, with its preferred name."""
        return f"""    <owl:AnnotationProperty rdf:about="{NCIT}{code}">
        <P108>{name}</P108>
    </owl:AnnotationProperty>"""

    return rdf(
        "\n".join(
            [
                prop("A90", "Has_ABCD_Value"),
                prop("A91", "Is_Value_For_ABCD"),
                f"""    <owl:Class rdf:about="{ns}C900001">
        <P108>Abcd Root</P108>
    </owl:Class>
    <owl:Class rdf:about="{ns}C900002">
        <rdfs:subClassOf rdf:resource="{ns}C900001"/>
        <P108>Abcd Measurement</P108>
        <P107>Abcd Display</P107>
        <P107>Abcd Display Two</P107>
        <P90>Abcd</P90>
        <P90>Abcd </P90>
        <P97>An abcd measure.</P97>
        <A8 rdf:resource="{ns}C900005"/>
        <A90 rdf:resource="{ns}C900003"/>
    </owl:Class>
    <owl:Class rdf:about="{ns}C900003">
        <P108>Efgh Value</P108>
        <A91 rdf:resource="{ns}C900002"/>
    </owl:Class>
    <owl:Class rdf:about="{ns}C900004">
        <P108>Old Abcd</P108>
        <P310>Retired_Concept</P310>
    </owl:Class>
    <owl:Class rdf:about="{ns}C900005">
        <P108>Abcd Subset</P108>
    </owl:Class>""",
                axiom(
                    ns + "C900002",
                    ns + "P90",
                    "Abcd",
                    "<P383>DN</P383><P384>ABCD</P384>",
                ),
                axiom(
                    ns + "C900002",
                    ns + "P90",
                    "Abcd ",
                    "<P383>PT</P383><P384>NCI</P384>",
                ),
                axiom(
                    ns + "C900002", ns + "P97", "An abcd measure.", "<P378>NCI</P378>"
                ),
            ]
        )
    )


class TestNcit:
    """NCIt's tables, and the replacements service's answers."""

    def test_concepts_synonyms_definitions_and_values(self):
        """Each FULL_SYN keeps its own axiom, and value lists keep their association."""
        tables = ncit.read_ncit(io.BytesIO(ncit_document()))
        concept = tables["concepts"].set_index("code").loc["C900002"]
        assert concept.display_names == "Abcd Display|Abcd Display Two"
        assert concept.parents == "C900001"
        assert concept.subsets == "Abcd Subset"
        synonyms = tables["synonyms"]
        assert sorted(zip(synonyms.name, synonyms.term_type, synonyms.source)) == [
            ("Abcd", "DN", "ABCD"),
            ("Abcd", "PT", "NCI"),
        ]
        assert tables["definitions"].to_dict("records") == [
            {
                "code": "C900002",
                "definition": "An abcd measure.",
                "kind": "DEFINITION",
                "source": "NCI",
            }
        ]
        assert tables["values"].to_dict("records") == [
            {
                "code": "C900002",
                "association": "Has_ABCD_Value",
                "value_code": "C900003",
                "value": "Efgh Value",
            }
        ]
        assert (
            tables["concepts"].set_index("code").loc["C900004"].status
            == "Retired_Concept"
        )

    def test_a_kept_zip_without_its_owl_file_names_the_fix(self, tmp_path):
        """A kept release zip that holds no .owl file is named, with what to do."""
        with zipfile.ZipFile(tmp_path / "Thesaurus_99.01d.OWL.zip", "w") as bundle:
            bundle.writestr("readme.txt", "abcd")
        manifest = {"sources": [{"name": ncit.RELEASE, "version": "99.01d"}]}
        with pytest.raises(ValueError, match="holds no .owl file; delete it"):
            ncit.fetch_source(tmp_path, manifest)

    def test_replacements_are_asked_once_and_kept(self, monkeypatch, tmp_path):
        """The replacements service is asked in batches, and its answers are read back later."""
        entry = {
            "code": "C900004",
            "action": "merge",
            "date": "2026-01-01",
            "replacementCode": "C900002",
            "replacementName": "Abcd Measurement",
        }
        service = MagicMock(return_value=json.dumps([entry]).encode())
        monkeypatch.setattr(ncit, "fetch", service)
        monkeypatch.setattr(ncit, "HISTORY_BATCH", 1)
        path = tmp_path / "replacements.json"
        frame = ncit.replacements(["C900004", "C900006"], path)
        assert service.call_count == 2
        assert frame.replacement_code.tolist() == ["C900002", "C900002"]
        ncit.replacements(["C900004"], path)
        assert service.call_count == 2


def obo_document():
    """A fictional ontology's base file, with a term of another ontology."""
    abcd, efgh = OBO + "ABCD_0000001", OBO + "ABCD_0000002"
    synonym = "{http://www.geneontology.org/formats/oboInOwl#}hasExactSynonym"[
        1:
    ].replace("}", "")
    return rdf(
        "\n".join(
            [
                f"""    <owl:Class rdf:about="{abcd}">
        <rdfs:label>abcd level</rdfs:label>
        <obo:IAO_0000115>The level of abcd.</obo:IAO_0000115>
        <oboInOwl:hasExactSynonym>ABCD</oboInOwl:hasExactSynonym>
        <oboInOwl:hasRelatedSynonym>abcd amount</oboInOwl:hasRelatedSynonym>
    </owl:Class>
    <owl:Class rdf:about="{efgh}">
        <rdfs:label>obsolete efgh level</rdfs:label>
        <owl:deprecated rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:deprecated>
        <obo:IAO_0100001 rdf:resource="{abcd}"/>
    </owl:Class>
    <owl:Class rdf:about="{OBO}EFGH_0000001">
        <rdfs:label>another ontology's term</rdfs:label>
    </owl:Class>
    <owl:Class rdf:about="{OBO}ABCD_0000003"/>
    <owl:Class rdf:about="{OBO}ABCD_0000005">
        <rdfs:label>obsolete Obsoleted abcd use</rdfs:label>
        <owl:deprecated rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:deprecated>
    </owl:Class>
    <owl:Class rdf:about="{OBO}ABCD_0000004">
        <owl:deprecated rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:deprecated>
        <obo:IAO_0100001 rdf:resource="{abcd}"/>
    </owl:Class>""",
                axiom(
                    abcd,
                    synonym,
                    "ABCD",
                    f'<oboInOwl:hasSynonymType rdf:resource="{OBO}OMO_0003000"/>',
                ),
            ]
        ),
        header=(
            '<dcterms:license rdf:resource="http://creativecommons.org/licenses/by/4.0/"/>'
            f'<owl:versionIRI rdf:resource="{OBO}abcd/releases/2026-01-01/abcd.owl"/>'
        ),
    )


class TestObo:
    """The OBO Foundry ontologies' terms, and their releases and licenses."""

    def test_own_terms_with_synonyms_types_and_replacements(self, monkeypatch):
        """Only the ontology's own terms; obsolete ones keep their replacement, label or not."""
        monkeypatch.setitem(obo.ONTOLOGIES, "abcd", ("ABCD", None, None))
        tables, stated = obo.read_ontology(io.BytesIO(obo_document()), "abcd")
        terms = tables["terms"].set_index("id")
        assert sorted(terms.index) == [
            "ABCD:0000001",
            "ABCD:0000002",
            "ABCD:0000004",
            "ABCD:0000005",
        ]
        assert terms.loc["ABCD:0000005", "label"] == "abcd use"
        assert terms.loc["ABCD:0000004", "label"] == ""
        assert terms.loc["ABCD:0000004", "replaced_by"] == "ABCD:0000001"
        assert terms.loc["ABCD:0000002", "label"] == "efgh level"
        assert terms.loc["ABCD:0000002", "obsolete"]
        assert terms.loc["ABCD:0000002", "replaced_by"] == "ABCD:0000001"
        synonyms = tables["synonyms"].set_index("name")
        assert synonyms.loc["ABCD", "synonym_types"] == "OMO:0003000"
        assert synonyms.loc["abcd amount", "scope"] == "RELATED"
        assert tables["definitions"].definition.tolist() == ["The level of abcd."]
        assert stated == {
            "license": "http://creativecommons.org/licenses/by/4.0/",
            "release": "2026-01-01",
        }

    def test_short_ids(self):
        """IRIs become OBO short ids; a local synonym type keeps only its name."""
        assert obo.short_id(OBO + "HP_0000001") == "HP:0000001"
        assert obo.short_id(OBO + "hp#layperson") == "layperson"

    def test_fetch_records_each_release_and_license(self, monkeypatch, tmp_path):
        """A dated release keeps its pin; an undated one takes its file's date. Licenses are the files'."""
        monkeypatch.setattr(
            obo,
            "ONTOLOGIES",
            {
                "abcd": ("ABCD", "{purl}/abcd/releases/{date}/abcd-base.owl", None),
                "efgh": ("EFGH", None, "{purl}/efgh.owl"),
            },
        )
        cc0 = "http://creativecommons.org/publicdomain/zero/1.0/"
        efgh = rdf(
            f"""<owl:Class rdf:about="{OBO}EFGH_0000001"><rdfs:label>efgh</rdfs:label></owl:Class>
    <owl:Class rdf:about="{OBO}EFGH_0000002">
        <rdfs:label>obsolete efgh use</rdfs:label>
        <rdfs:subClassOf rdf:resource="{obo.OBSOLETE_CLASS}"/>
    </owl:Class>""",
            header=(
                f'<owl:versionIRI rdf:resource="{OBO}efgh/releases/2026-02-02/efgh.owl"/>'
                f'<obo:terms_license rdf:resource="{cc0}"/>'
            ),
        )
        dated = f"{obo.PURL}/abcd/releases/2026-01-01/abcd-base.owl"
        monkeypatch.setattr(
            obo,
            "downloaded",
            served({dated: obo_document(), f"{obo.PURL}/efgh.owl": efgh}),
        )
        manifest = {
            "sources": [
                {"name": "ABCD release", "version": "2026-01-01"},
                {"name": "EFGH release"},
            ]
        }
        tables, found, counts = obo.fetch_source(tmp_path, manifest)
        assert counts == {"terms": {"abcd": 4, "efgh": 2}}
        assert tables["terms"].id.tolist() == [
            "ABCD:0000001",
            "ABCD:0000002",
            "ABCD:0000004",
            "ABCD:0000005",
            "EFGH:0000001",
            "EFGH:0000002",
        ]
        efgh_use = tables["terms"].set_index("id").loc["EFGH:0000002"]
        assert efgh_use.obsolete and efgh_use.label == "efgh use"
        assert [(s["version"], s["license"]) for s in found] == [
            ("2026-01-01", "http://creativecommons.org/licenses/by/4.0/"),
            ("2026-02-02", cc0),
        ]
        assert "fetched" in found[1] and "fetched" not in found[0]
        assert [s["role"] for s in found] == ["release", "service"]


def csv_bytes(rows):
    """A CSV file's bytes from a list of dicts."""
    return pd.DataFrame(rows).to_csv(index=False).encode()


FORM_ROW = {
    "ParentLoinc": "99999-3",
    "SEQUENCE": "1",
    "DisplayNameForForm": "Abcd?",
    "EXTERNAL_COPYRIGHT_NOTICE": "",
    "AdditionalCopyright": "",
}


def loinc_zip():
    """A fictional LOINC release with the five tables read, and a SNOMED column."""
    code = dict.fromkeys(loinc.TABLES["codes"][1], "")
    rows = {
        "LoincTable/Loinc.csv": [
            {**code, "LOINC_NUM": "99999-1", "STATUS": "ACTIVE", "LONG_COMMON_NAME": "Abcd [Mass]"},
            {**code, "LOINC_NUM": "99999-2", "STATUS": "ACTIVE", "EXTERNAL_COPYRIGHT_NOTICE": "Abcd Inc."},
            {**code, "LOINC_NUM": "99999-5", "STATUS": "ACTIVE", "LONG_COMMON_NAME": "Efgh, in a closed panel only"},
        ],
        "LoincTable/MapTo.csv": [{"LOINC": "99999-2", "MAP_TO": "99999-1", "COMMENT": ""}],
        "AccessoryFiles/PanelsAndForms/PanelsAndForms.csv": [
            {**FORM_ROW, "ID": "1", "ParentId": "1", "Loinc": "99999-3"},
            {**FORM_ROW, "ID": "2", "ParentId": "1", "Loinc": "99999-1"},
            {**FORM_ROW, "ID": "3", "ParentId": "1", "Loinc": "99999-2"},
            {**FORM_ROW, "ID": "4", "ParentId": "1", "Loinc": "99999-4",
             "AdditionalCopyright": "Copyright Abcd Inc."},
            {**FORM_ROW, "ID": "5", "ParentId": "4", "Loinc": "99999-5"},
            {**FORM_ROW, "ID": "6", "ParentId": "5", "Loinc": "99999-1"},
        ],
        "AccessoryFiles/AnswerFile/AnswerList.csv": [
            {"AnswerListId": "LL9999-1", "AnswerListName": "Abcd", "AnswerStringId": "LA9999-1",
             "LocalAnswerCode": "1", "SequenceNumber": "1", "DisplayText": "Yes",
             "Description": "", "ExtCodeDisplayName": "a SNOMED CT name"}
        ],
        "AccessoryFiles/AnswerFile/LoincAnswerListLink.csv": [
            {"LoincNumber": "99999-1", "AnswerListId": "LL9999-1",
             "AnswerListLinkType": "EXAMPLE", "ApplicableContext": ""}
        ],
    }  # fmt: skip
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, table in rows.items():
            bundle.writestr(f"Loinc_9.99/{name}", csv_bytes(table))
    return buffer.getvalue()


class TestLoinc:
    """LOINC's tables, its restriction fields, and how its zip is obtained."""

    def test_named_columns_only_and_notices_close(self):
        """SNOMED CT columns are never read; a notice closes a code, a form row and a panel's rows."""
        tables = loinc.read_release(io.BytesIO(loinc_zip()))
        assert "ExtCodeDisplayName" not in tables["answers"].columns
        assert tables["codes"].set_index("LOINC_NUM").open.to_dict() == {
            "99999-1": True,
            "99999-2": False,
            "99999-5": False,
        }
        assert tables["map_to"].MAP_TO.tolist() == ["99999-1"]
        assert tables["forms"].set_index("ID").open.to_dict() == {
            "1": True,
            "2": True,
            "3": False,
            "4": False,
            "5": False,
            "6": False,
        }

    def test_without_credentials_or_zip_the_instructions_are_given(
        self, monkeypatch, tmp_path
    ):
        """With no zip and no credentials, the error says how to get the release."""
        monkeypatch.delenv("LOINC_USERNAME", raising=False)
        manifest = {"sources": [{"name": loinc.RELEASE, "version": "9.99"}]}
        with pytest.raises(FileNotFoundError, match="LOINC_USERNAME"):
            loinc.fetch_source(tmp_path, manifest)

    def test_a_zip_with_another_md5_is_refused(self, tmp_path):
        """A zip placed by hand must have the MD5 that the manifest pins."""
        (tmp_path / "Loinc_9.99.zip").write_bytes(loinc_zip())
        manifest = {
            "sources": [{"name": loinc.RELEASE, "version": "9.99", "md5": "abcd"}]
        }
        with pytest.raises(ValueError, match="pins abcd. Delete it to download"):
            loinc.fetch_source(tmp_path, manifest)

    def test_a_download_with_another_md5_is_not_kept(self, monkeypatch, tmp_path):
        """A download without LOINC's published MD5 is refused, and not kept."""
        monkeypatch.setenv("LOINC_USERNAME", "abcd")
        monkeypatch.setenv("LOINC_PASSWORD", "efgh")
        monkeypatch.setattr(
            loinc,
            "fetch",
            lambda url, **request: json.dumps({"downloadMD5Hash": "abcd"}).encode(),
        )
        monkeypatch.setattr(
            loinc,
            "downloaded",
            served({loinc.DOWNLOAD.format(version="9.99"): loinc_zip()}),
        )
        manifest = {"sources": [{"name": loinc.RELEASE, "version": "9.99"}]}
        with pytest.raises(ValueError, match="MD5"):
            loinc.fetch_source(tmp_path, manifest)
        assert not (tmp_path / "Loinc_9.99.zip").exists()

    def test_a_download_with_another_md5_than_pinned_is_refused(
        self, monkeypatch, tmp_path
    ):
        """A download with LOINC's MD5 is refused, and not kept, when the manifest pins another."""
        data = loinc_zip()
        monkeypatch.setenv("LOINC_USERNAME", "abcd")
        monkeypatch.setenv("LOINC_PASSWORD", "efgh")
        published = {
            "downloadMD5Hash": hashlib.md5(data, usedforsecurity=False).hexdigest()
        }
        monkeypatch.setattr(
            loinc, "fetch", lambda url, **request: json.dumps(published).encode()
        )
        monkeypatch.setattr(
            loinc, "downloaded", served({loinc.DOWNLOAD.format(version="9.99"): data})
        )
        manifest = {
            "sources": [{"name": loinc.RELEASE, "version": "9.99", "md5": "abcd"}]
        }
        with pytest.raises(ValueError, match="the manifest's is abcd"):
            loinc.fetch_source(tmp_path, manifest)
        assert not (tmp_path / "Loinc_9.99.zip").exists()

    def test_a_zip_with_the_pinned_md5_is_read(self, tmp_path):
        """A zip placed by hand with the pinned MD5 is read, and its MD5 recorded."""
        data = loinc_zip()
        (tmp_path / "Loinc_9.99.zip").write_bytes(data)
        md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
        manifest = {"sources": [{"name": loinc.RELEASE, "version": "9.99", "md5": md5}]}
        tables, found, counts = loinc.fetch_source(tmp_path, manifest)
        assert len(tables["codes"]) == 3
        assert counts == {
            "codes": 3,
            "not_open": int((~tables["codes"].open).sum()),
        }
        assert found[0]["md5"] == md5


class TestCdisc:
    """CDISC's terminology files."""

    def test_na_is_a_value_not_missing(self):
        """A submission value of NA stays the text NA."""
        data = (
            "\t".join(cdisc.COLUMNS)
            + "\n"
            + "\t".join(
                [
                    "C99999",
                    "C99998",
                    "",
                    "Abcd",
                    "NA",
                    "",
                    "Not applicable",
                    "Not Applicable",
                ]
            )
        )
        frame = cdisc.read_package(data.encode(), "sdtm")
        assert frame["CDISC Submission Value"].tolist() == ["NA"]
        assert frame.package.tolist() == ["sdtm"]

    def test_quotes_are_text_and_a_lost_column_is_an_error(self):
        """A cell starting with a quote is read as written; a missing column raises."""
        row = ["C99999", "C99998", "", "Abcd", '"Abcd', "", "Abcd", "Abcd"]
        data = "\t".join(cdisc.COLUMNS) + "\n" + "\t".join(row)
        frame = cdisc.read_package(data.encode(), "sdtm")
        assert frame["CDISC Submission Value"].tolist() == ['"Abcd']
        renamed = data.replace("CDISC Synonym(s)", "CDISC Synonyms")
        with pytest.raises(ValueError):
            cdisc.read_package(renamed.encode(), "sdtm")


def cde_element(
    tiny_id, copyright_status="Public domain, free to use", no_render_allowed=False
):
    """A fictional record of the NIH CDE export."""
    return {
        "tinyId": tiny_id,
        "stewardOrg": {"name": "Abcd"},
        "registrationState": {
            "registrationStatus": "Qualified",
            "administrativeStatus": "Published",
        },
        "copyrightStatus": copyright_status,
        "noRenderAllowed": no_render_allowed,
        "designations": [
            {"designation": "Abcd score", "tags": []},
            {"designation": "Abcd?", "tags": ["Preferred Question Text"]},
        ],
        "definitions": [{"definition": "The abcd score.", "tags": []}],
        "ids": [{"source": "BRICS Variable Name", "id": "AbcdScore"}],
        "valueDomain": {
            "datatype": "Value List",
            "permissibleValues": [{"permissibleValue": "1", "valueMeaningName": "Low"}],
        },
        "properties": [{"key": "Abcd", "value": "Efgh"}],
    }


class TestCde:
    """The NIH CDE export's tables, restriction fields and count check."""

    def test_tables_and_the_allow_list(self, capsys):
        """Only a public-domain, renderable element is open; a new status is reported."""
        tables = cde.read_export(
            [
                cde_element("abcd1"),
                cde_element("abcd2", no_render_allowed=True),
                cde_element("abcd3", copyright_status="Abcd copyright"),
            ]
        )
        assert tables["elements"].set_index("tiny_id").open.to_dict() == {
            "abcd1": True,
            "abcd2": False,
            "abcd3": False,
        }
        assert "Abcd copyright" in capsys.readouterr().out
        designations = tables["designations"]
        assert designations[designations.tiny_id == "abcd1"].tags.tolist() == [
            "",
            "Preferred Question Text",
        ]
        assert tables["values"].value_meaning_name.unique().tolist() == ["Low"]
        assert tables["ids"].id.unique().tolist() == ["AbcdScore"]

    def test_retired_elements_are_asked_for(self, monkeypatch, tmp_path):
        """The export and the search total both include retired elements."""
        (tmp_path / "SearchExport.json").write_text("[]")
        export = MagicMock(return_value=[cde_element("abcd1")])
        monkeypatch.setattr(cde, "export", export)
        cde.fetch_source(tmp_path, {"sources": []})
        path, url, query = export.call_args.args
        assert query["includeRetired"] is True
        search = MagicMock(return_value=b'{"totalItems": 1}')
        monkeypatch.setattr(cde, "fetch", search)
        assert cde.search_total() == 1
        assert search.call_args.kwargs["json"]["includeRetired"] is True

    def test_an_export_short_of_the_total_is_not_kept(self, monkeypatch, tmp_path):
        """An export whose count is not the search's total raises, and nothing is kept."""
        monkeypatch.setattr(cde, "search_total", lambda: 2)
        monkeypatch.setattr(
            download,
            "fetch",
            lambda url, **request: json.dumps([cde_element("abcd1")]).encode(),
        )
        with pytest.raises(ValueError, match="search reports"):
            cde.export(tmp_path / "SearchExport.json", "https://example.org/export", {})
        assert list(tmp_path.iterdir()) == []


def cadsr_element(public_id, version, documents=(), long_name="Abcd Score"):
    """A fictional caDSR DataElement record."""
    docs = "".join(
        f"<REFERENCEDOCUMENTSLIST_ITEM><Name>{kind}</Name><DocumentType>{kind}</DocumentType>"
        f"<DocumentText>{text}</DocumentText></REFERENCEDOCUMENTSLIST_ITEM>"
        for kind, text in documents
    )
    return f"""<DataElement num="1">
  <PUBLICID>{public_id}</PUBLICID><VERSION>{version}</VERSION>
  <LONGNAME>{long_name}</LONGNAME><PREFERREDNAME>ABCD_SCR</PREFERREDNAME>
  <PREFERREDDEFINITION>The abcd score.</PREFERREDDEFINITION>
  <WORKFLOWSTATUS>RELEASED</WORKFLOWSTATUS><REGISTRATIONSTATUS>Qualified</REGISTRATIONSTATUS>
  <CONTEXTNAME>ABCD</CONTEXTNAME>
  <VALUEDOMAIN><Datatype>CHARACTER</Datatype><ValueDomainType>Enumerated</ValueDomainType>
    <PermissibleValues><PermissibleValues_ITEM><VALIDVALUE>1</VALIDVALUE><VALUEMEANING>Low</VALUEMEANING><PVBEGINDATE>2002-02-11</PVBEGINDATE><PVENDDATE>2020-01-01</PVENDDATE></PermissibleValues_ITEM></PermissibleValues>
  </VALUEDOMAIN>
  <ALTERNATENAMELIST><ALTERNATENAMELIST_ITEM><ContextName>ABCD</ContextName><AlternateName>abcd</AlternateName><AlternateNameType>ABCD Alt Name</AlternateNameType><Language>ENGLISH</Language></ALTERNATENAMELIST_ITEM></ALTERNATENAMELIST>
  <REFERENCEDOCUMENTSLIST>{docs}</REFERENCEDOCUMENTSLIST>
</DataElement>"""


class TestCadsr:
    """caDSR's snapshot: latest versions, and its restriction field."""

    def test_latest_version_once_and_the_use_field(self, capsys):
        """Each element is kept at its latest version, once; a proprietary one is closed."""
        xml = (
            "<DataElementsList>"
            + "".join(
                [
                    cadsr_element("9999901", "1", long_name="Old Abcd Score"),
                    cadsr_element(
                        "9999901", "2", [("Preferred Question Text", "Abcd?")]
                    ),
                    cadsr_element(
                        "9999901", "2", [("Preferred Question Text", "Abcd?")]
                    ),
                    cadsr_element("9999902", "1", [(cadsr.PROPRIETARY, "Ask Abcd.")]),
                    cadsr_element("9999903", "1", [("Abcd Note", "efgh")]),
                ]
            )
            + "</DataElementsList>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("cde_xml_1.xml", xml)
        tables = cadsr.read_snapshot(buffer)
        elements = tables["elements"].set_index("public_id")
        assert elements.loc["9999901", "long_name"] == "Abcd Score"
        assert len(elements) == 3
        assert elements.open.to_dict() == {
            "9999901": True,
            "9999902": False,
            "9999903": True,
        }
        assert "Abcd Note" in capsys.readouterr().out
        documents = tables["documents"]
        assert documents[documents.public_id == "9999901"].text.tolist() == ["Abcd?"]
        assert tables["values"].meaning.unique().tolist() == ["Low"]
        assert tables["values"].end_date.unique().tolist() == ["2020-01-01"]

    def test_versions_compare_as_decimal_numbers(self):
        """Version 2.4 is later than 2.31, as caDSR stores versions as numbers."""
        xml = (
            "<DataElementsList>"
            + cadsr_element("9999901", "2.31", [("Abcd Note", "a")], "Abcd Older")
            + cadsr_element("9999901", "2.4", [("Abcd Note", "b")], "Abcd Newer")
            + "</DataElementsList>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("cde_xml_1.xml", xml)
        elements = cadsr.read_snapshot(buffer)["elements"]
        assert elements.long_name.tolist() == ["Abcd Newer"]

    def test_a_version_not_a_number_names_its_element(self):
        """An element whose version is not a number is an error that names it."""
        with pytest.raises(ValueError, match="9999901"):
            cadsr.version({"public_id": "9999901", "version": "abcd"})


VARIABLE_LIST = b"""<html><body><table>
<tr><th>Variable Name</th><th>Variable Description</th><th>Data File Name</th>
<th>Data File Description</th><th>Begin Year</th><th>EndYear</th><th>Component</th>
<th>Use Constraints</th></tr>
<tr><td>ABCDXYZ</td><td>Abcd score</td><td>ABCD_J</td><td>Abcd</td><td>2017</td>
<td>2018</td><td>Questionnaire</td><td>None</td></tr>
</table></body></html>"""

CODEBOOK = b"""<html><body>
<div class="pagebreak"><h3 class="vartitle" id="ABCDXYZ">ABCDXYZ - Abcd score</h3>
<dl><dt>Variable Name: </dt><dd>ABCDXYZ</dd><dt>SAS Label: </dt><dd>Abcd score</dd>
<dt>English Text: </dt><dd>How abcd are you?</dd><dt>Target: </dt><dd> Both males and
 females 18 YEARS - 150 YEARS</dd><dt>Target: </dt><dd>Females only 16 YEARS - 49 YEARS</dd>
<dt>English Instructions: </dt><dd>Read the abcd card.</dd></dl>
<table class="values"><thead><tr><th>Code or Value</th><th>Value Description</th></tr></thead>
<tbody><tr><td>1</td><td>Very</td></tr><tr><td>2 to 9</td><td>Range of Values</td></tr>
<tr><td>.</td><td>Missing</td></tr></tbody></table></div>
</body></html>"""


class TestNhanes:
    """NHANES' variable lists and codebook pages."""

    def test_variable_list(self):
        """The variable list's columns are read under their short names."""
        frame = nhanes.read_variable_list(VARIABLE_LIST)
        assert frame.to_dict("records") == [
            {
                "name": "ABCDXYZ",
                "description": "Abcd score",
                "data_file": "ABCD_J",
                "data_file_description": "Abcd",
                "begin_year": "2017",
                "end_year": "2018",
                "component": "Questionnaire",
                "use_constraints": "None",
            }
        ]

    def test_a_list_without_a_column_is_an_error(self):
        """A variable list that lost a column is refused, not read with it empty."""
        with pytest.raises(ValueError, match="Begin Year"):
            nhanes.read_variable_list(VARIABLE_LIST.replace(b"Begin Year", b"Start"))

    def test_codebook_keeps_every_value_row(self):
        """The codebook's texts are read, every target kept, and every row of the value table."""
        variables, values = nhanes.read_codebook(CODEBOOK)
        assert variables == [
            {
                "name": "ABCDXYZ",
                "sas_label": "Abcd score",
                "english_text": "How abcd are you?",
                "english_instructions": "Read the abcd card.",
                "target": "Both males and females 18 YEARS - 150 YEARS|"
                "Females only 16 YEARS - 49 YEARS",
                "hard_edits": "",
            }
        ]
        assert [(v["code"], v["label"]) for v in values] == [
            ("1", "Very"),
            ("2 to 9", "Range of Values"),
            (".", "Missing"),
        ]

    def test_a_codebook_row_takes_open_from_its_data_file(self):
        """A codebook row that the lists do not name takes its data file's open."""
        variables = pd.DataFrame(
            {
                "cycle": ["2017", "2017"],
                "data_file": ["ABCD_J", "EFGH_J"],
                "open": [True, False],
            }
        )
        codebook = pd.DataFrame(
            {
                "cycle": ["2017", "2017"],
                "data_file": ["ABCD_J", "EFGH_J"],
                "name": ["ABCDBOX", "EFGHXYZ"],
            }
        )
        assert nhanes.add_open(codebook, variables).open.tolist() == [True, False]

    def test_cycles_start_on_odd_years(self):
        """A file beginning in 2000 is in the 1999 cycle."""
        assert nhanes.cycle("2000") == 1999
        assert nhanes.cycle("2017") == 2017

    @pytest.fixture
    def pages(self, monkeypatch):
        """A variable list of two data files, one with a new Use Constraints value."""
        monkeypatch.setattr(nhanes, "COMPONENTS", ["Questionnaire"])
        second = (
            b"<tr><td>EFGHXYZ</td><td>Efgh score</td><td>EFGH_J</td><td>Efgh</td>"
            b"<td>2017</td><td>2018</td><td>Questionnaire</td><td>Abcd Only</td></tr>"
        )
        listing = VARIABLE_LIST.replace(b"</table>", second + b"</table>")
        return {
            nhanes.VARIABLE_LIST.format(component="Questionnaire"): listing,
            nhanes.CODEBOOK.format(cycle=2017, data_file="ABCD_J"): CODEBOOK,
        }

    def test_a_codebook_not_found_is_left_out(
        self, pages, monkeypatch, capsys, tmp_path
    ):
        """A data file whose codebook page is not found is counted, and the rest is read."""
        pages[nhanes.CODEBOOK.format(cycle=2017, data_file="EFGH_J")] = http_error(404)
        monkeypatch.setattr(nhanes, "downloaded", served(pages))
        tables, found, counts = nhanes.fetch_source(tmp_path, {"sources": []})
        assert counts["data_files_without_codebook"] == 1
        printed = capsys.readouterr().out
        assert "new value of Use Constraints: 'Abcd Only' on 1 items" in printed
        assert "1 data files without a public codebook" in printed
        assert tables["variables"].set_index("name").open.to_dict() == {
            "ABCDXYZ": True,
            "EFGHXYZ": False,
        }
        assert tables["codebook"].data_file.tolist() == ["ABCD_J"]
        assert tables["variables"].cycle.unique().tolist() == ["2017"]
        assert tables["codebook"].open.tolist() == [True]
        assert len(tables["values"]) == 3
        assert [source["name"] for source in found] == [
            "NHANES variable list, Questionnaire",
            "NHANES codebooks",
        ]

    def test_an_odd_data_file_name_stops_the_fetch(self, pages, monkeypatch, tmp_path):
        """A data file name with characters other than letters, digits and _ is refused."""
        listing = pages[nhanes.VARIABLE_LIST.format(component="Questionnaire")]
        pages[nhanes.VARIABLE_LIST.format(component="Questionnaire")] = listing.replace(
            b"EFGH_J", b"../EFGH_J"
        )
        monkeypatch.setattr(nhanes, "downloaded", served(pages))
        with pytest.raises(ValueError, match="odd data files"):
            nhanes.fetch_source(tmp_path, {"sources": []})
        assert not (tmp_path / "EFGH_J.htm").exists()

    def test_a_page_not_found_is_not_asked_for_again(
        self, pages, monkeypatch, capsys, tmp_path
    ):
        """A later fetch reads the list of pages not found, and needs no network."""
        pages[nhanes.CODEBOOK.format(cycle=2017, data_file="EFGH_J")] = http_error(404)
        monkeypatch.setattr(nhanes, "downloaded", served(pages))
        nhanes.fetch_source(tmp_path, {"sources": []})
        assert (
            tmp_path / "codebooks" / nhanes.NOT_FOUND
        ).read_text() == "2017/EFGH_J\n"
        monkeypatch.setattr(nhanes, "downloaded", served({}))
        tables, _, _ = nhanes.fetch_source(tmp_path, {"sources": []})
        assert "1 data files without a public codebook" in capsys.readouterr().out
        assert tables["codebook"].data_file.tolist() == ["ABCD_J"]

    def test_another_page_than_a_codebook_is_not_kept(
        self, pages, monkeypatch, tmp_path
    ):
        """A page served with success but without a codebook stops the fetch, and is not kept."""
        failed = nhanes.CODEBOOK.format(cycle=2017, data_file="EFGH_J")
        pages[failed] = b"<html><body>Checking your browser...</body></html>"
        monkeypatch.setattr(nhanes, "downloaded", served(pages))
        with pytest.raises(RuntimeError, match="another page"):
            nhanes.fetch_source(tmp_path, {"sources": []})
        assert not (tmp_path / "codebooks" / "2017" / "EFGH_J.htm").exists()

    def test_another_download_failure_stops_the_fetch(
        self, pages, monkeypatch, tmp_path
    ):
        """A codebook page that fails for another reason stops the fetch, naming it."""
        failed = nhanes.CODEBOOK.format(cycle=2017, data_file="EFGH_J")
        pages[failed] = requests.ConnectionError("connection reset")
        monkeypatch.setattr(nhanes, "downloaded", served(pages))
        with pytest.raises(RuntimeError, match="EFGH_J"):
            nhanes.fetch_source(tmp_path, {"sources": []})


DUMP = """CREATE TABLE `protocols` (
  `id` int NOT NULL,
  `code` varchar(9),
  `name` varchar(9),
  `description` text,
  `visible` tinyint,
  `release_date` datetime,
  `version` int,
  `status` varchar(9),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB;
INSERT INTO `protocols` VALUES (1,'abcd','Abcd','It''s abcd',1,NULL,1,'new'),(2,'efgh','Efgh','Efgh\\'s',0,NULL,1,'new');
CREATE TABLE `protocol_availability_types` (
  `id` int NOT NULL,
  `name` varchar(30),
  `definition` text
);
INSERT INTO `protocol_availability_types` VALUES (1,'Available','Free'),(2,'Limited Availability','Ask');
CREATE TABLE `protocols_protocol_availability_types` (
  `id` int NOT NULL,
  `protocol_id` int,
  `protocol_availability_type_id` int
);
INSERT INTO `protocols_protocol_availability_types` VALUES (1,1,1),(2,2,2);
CREATE TABLE `variables` (
  `id` int NOT NULL,
  `name` varchar(30),
  `code` varchar(30),
  `description` text,
  `protocol_id` int,
  `type` varchar(9),
  `units` varchar(9),
  `min` varchar(9),
  `max` varchar(9),
  `comment1` text,
  `comment2` text,
  `variable_values` text,
  `version_num` int,
  `created_dt` datetime
);
INSERT INTO `variables` VALUES (2,'PX1_Abcd','PX1','Abcd?',1,'string','','','','','','Yes|No',0,NULL),(10,'PX2_Efgh','PX2','Efgh?',2,'string','','','','','','',0,NULL),(3,'PX9_Covid','PX9','Covid?',99,'string','','','','','','',0,NULL);
CREATE TABLE `variable_gap_mappings` (
  `id` int NOT NULL,
  `variable_id` int,
  `gap_study_name` varchar(30),
  `gap_study_accession` varchar(9),
  `gap_phv` varchar(12),
  `gap_variable_name` varchar(30),
  `gap_variable_desc` text,
  `created_dt` datetime,
  `map_level` varchar(12),
  `variable_name` varchar(30)
);
INSERT INTO `variable_gap_mappings` VALUES (1,2,'Abcd Study','phs999999','phv99999901','ABCD','abcd answer',NULL,'comparable','PX1_Abcd');
CREATE TABLE `protocols_old_protocols_new` (
  `id` int NOT NULL,
  `protocol_old_id` int,
  `protocol_new_id` int
);
INSERT INTO `protocols_old_protocols_new` VALUES (1,2,1);
CREATE TABLE `releases` (
  `id` int NOT NULL,
  `name` varchar(20),
  `release_date` datetime
);
INSERT INTO `releases` VALUES (1,'Ver 9.0','2026-01-01 00:00:00'),(2,'Ver 9.1','2026-02-01 00:00:00');
"""


def phenx_zip():
    """A fictional PhenX export: a zip holding phenx.sql."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("phenx.sql", DUMP)
    return buffer.getvalue()


class TestPhenx:
    """PhenX's database export: its dump, tables and release."""

    def test_insert_rows_with_quotes_escapes_and_null(self):
        """Doubled and escaped quotes, NULL and bare numbers are read as MySQL writes them."""
        assert phenx.insert_rows("(1,'It''s','a\\'b\\nc',NULL);") == [
            ["1", "It's", "a'b\nc", None]
        ]

    def test_a_row_short_of_its_columns_is_an_error(self):
        """A row with fewer values than its table's columns is not padded, but refused."""
        dump = DUMP.replace("(1,'Available','Free')", "(1,'Available')")
        with pytest.raises(ValueError, match="protocol_availability_types"):
            phenx.read_dump(io.BytesIO(dump.encode()), phenx.TABLES)

    def test_the_latest_release_is_by_date_then_id(self):
        """Two releases on one day: the later id is the latest."""
        dump = DUMP.replace(
            "(2,'Ver 9.1','2026-02-01 00:00:00')",
            "(2,'Ver 9.1','2026-02-01 00:00:00'),(3,'Ver 9.2','2026-02-01 00:00:00')",
        )
        tables = phenx.read_export(
            phenx.read_dump(io.BytesIO(dump.encode()), phenx.TABLES)
        )
        assert tables["release"].name.tolist() == ["Ver 9.2"]

    def test_export_tables_availability_and_release(self):
        """A variable is open when its protocol is Available; the latest release is kept."""
        dump = phenx.read_dump(io.BytesIO(DUMP.encode()), phenx.TABLES)
        tables = phenx.read_export(dump)
        variables = tables["variables"].set_index("name")
        assert variables.open.to_dict() == {
            "PX1_Abcd": True,
            "PX9_Covid": False,
            "PX2_Efgh": False,
        }
        assert variables.loc["PX2_Efgh", "protocol_visible"] == "0"
        assert variables.loc["PX9_Covid", "availability"] == ""
        assert tables["variables"].id.tolist() == ["2", "3", "10"]
        assert tables["protocols"].description.tolist() == ["It's abcd", "Efgh's"]
        assert tables["protocols"].open.tolist() == [True, False]
        assert tables["mappings"].gap_phv.tolist() == ["phv99999901"]
        assert tables["release"].name.tolist() == ["Ver 9.1"]

    def test_an_export_is_downloaded_once_and_read(self, monkeypatch, tmp_path):
        """The export is downloaded from PhenX's address, kept, and read with its release."""
        monkeypatch.setattr(
            phenx, "downloaded", served({phenx.EXPORT_URL: phenx_zip()})
        )
        manifest = {"sources": [{"name": phenx.EXPORT, "version": "Ver 9.1"}]}
        tables, found, counts = phenx.fetch_source(tmp_path, manifest)
        assert counts["variables"] == len(tables["variables"])
        assert (tmp_path / "Phenx_DB.zip").exists()
        assert tables["replacements"].protocol_new_id.tolist() == ["1"]
        assert found[0]["role"] == "service"
        assert found[0]["version"] == "Ver 9.1"

    def test_a_download_without_the_dump_is_not_kept(self, monkeypatch, tmp_path):
        """A download that is not a zip holding phenx.sql raises, and nothing is kept."""
        monkeypatch.setattr(
            phenx, "downloaded", served({phenx.EXPORT_URL: b"not a zip"})
        )
        manifest = {"sources": [{"name": phenx.EXPORT}]}
        with pytest.raises(ValueError, match="cannot be read as a zip"):
            phenx.fetch_source(tmp_path, manifest)
        assert list(tmp_path.iterdir()) == []

    def test_another_release_than_pinned_is_refused(self, tmp_path):
        """An export of a release that the manifest does not pin is refused."""
        (tmp_path / "Phenx_DB.zip").write_bytes(phenx_zip())
        manifest = {"sources": [{"name": phenx.EXPORT, "version": "Ver 8.0"}]}
        with pytest.raises(
            ValueError, match="is PhenX Ver 9.1, but the manifest pins Ver 8.0"
        ) as raised:
            phenx.fetch_source(tmp_path, manifest)
        assert "follow 'Newer releases'" in str(raised.value)


class TestCommon:
    """The helpers every source uses."""

    def test_pinned_names_a_missing_input(self):
        """A manifest without the named input raises, naming it."""
        with pytest.raises(KeyError, match="Abcd release"):
            common.pinned({"sources": []}, "Abcd release")

    def test_a_new_value_is_closed_and_reported(self, capsys):
        """A value in neither list is not open, and is printed with its count."""
        found = common.open_where(
            pd.Series(["yes", "no", "maybe", "maybe"]), {"yes"}, {"no"}, "abcd"
        )
        assert found.tolist() == [True, False, False, False]
        printed = capsys.readouterr().out
        assert "'maybe' on 2 items" in printed
        assert "'no'" not in printed


TABLE = pd.DataFrame({"id": ["a1", "a2"], "open": [True, False]})


class TestWriteAndRead:
    """Writing a source's tables, and reading them back."""

    def test_round_trip_and_content_hash(self, tmp_path):
        """Tables written as parquet read back, checked by their content's SHA-256."""
        directory = tmp_path / "sources" / "abcd"
        directory.mkdir(parents=True)
        manifest = sources.write_source(
            directory, "abcd", "1", {"items": TABLE}, [], {"items": 3}
        )
        assert manifest["counts"] == {"items": 3}
        assert manifest["files"]["items.parquet"]["csv_sha256"] == sources.table_sha256(
            TABLE
        )
        assert manifest["files"]["items.parquet"]["sha256"] == download.file_sha256(
            directory / "items.parquet"
        )
        tables, read = sources.read_source(tmp_path, "abcd")
        pd.testing.assert_frame_equal(tables["items"], TABLE)
        assert read == manifest

    def test_a_table_checksum_does_not_depend_on_the_system(self, monkeypatch):
        """A table's checksum is the same where lines end with \\r\\n, as on Windows."""
        before = sources.table_sha256(TABLE)
        monkeypatch.setattr("os.linesep", "\r\n")
        assert sources.table_sha256(TABLE) == before

    def test_a_checksum_written_in_chunks_is_that_of_the_whole_text(self):
        """The streamed SHA-256 is that of the whole CSV text, dates and accents too."""
        # This checks that the SHA-256 of 120,000 rows, written piece by piece,
        # equals the SHA-256 of their whole CSV text. With two columns, pandas
        # writes 50,000 rows per chunk. It leaves out the time in a chunk where
        # no value has one. The times fill only the last 10,000 rows, so the
        # last chunk mixes dates and times. Another chunk size would write some
        # dates without a time. The "É" of "Éfgh" takes two bytes in UTF-8, so
        # the bytes are checked too.
        when = pd.to_datetime(
            ["2020-01-01"] * 110_000 + ["2020-01-03 10:00"] * 10_000, format="ISO8601"
        )
        table = pd.DataFrame({"when": when, "name": ["Abcd", "Éfgh"] * 60_000})
        text = table.to_csv(index=False, lineterminator="\n").encode("utf-8")
        assert sources.table_sha256(table) == hashlib.sha256(text).hexdigest()

    def test_a_changed_table_is_refused(self, tmp_path):
        """A table whose content changed since its fetch is refused."""
        directory = tmp_path / "sources" / "abcd"
        directory.mkdir(parents=True)
        sources.write_source(directory, "abcd", "1", {"items": TABLE}, [], {})
        TABLE.head(1).to_parquet(directory / "items.parquet", index=False)
        with pytest.raises(ValueError, match="items.parquet"):
            sources.read_source(tmp_path, "abcd")

    def test_a_rewritten_table_file_is_refused(self, tmp_path):
        """A table written again, with the same content but other bytes, is refused."""
        directory = tmp_path / "sources" / "abcd"
        directory.mkdir(parents=True)
        sources.write_source(directory, "abcd", "1", {"items": TABLE}, [], {})
        TABLE.to_parquet(directory / "items.parquet", index=False, compression="gzip")
        with pytest.raises(ValueError, match="items.parquet differs from its manifest"):
            sources.read_source(tmp_path, "abcd")

    def test_another_version_than_committed_is_a_warning(
        self, monkeypatch, capsys, tmp_path
    ):
        """A fetch whose version is not the committed version is read, with a warning."""
        folder = tmp_path / "manifests"
        folder.mkdir()
        (folder / "abcd.json").write_text(json.dumps({"version": "2"}))
        monkeypatch.setattr(sources, "MANIFESTS", folder)
        directory = tmp_path / "sources" / "abcd"
        directory.mkdir(parents=True)
        sources.write_source(directory, "abcd", "1", {"items": TABLE}, [], {})
        sources.read_source(tmp_path, "abcd")
        assert "is version 1, and the committed manifest is version 2" in (
            capsys.readouterr().err
        )

    def test_an_unfetched_source_names_the_command(self, tmp_path):
        """Reading a source never fetched says how to fetch it."""
        with pytest.raises(FileNotFoundError, match="sources fetch --source abcd"):
            sources.read_source(tmp_path, "abcd")

    def test_a_missing_table_names_the_command(self, tmp_path):
        """A table that the manifest lists but that is gone is named, with the fetch."""
        directory = tmp_path / "sources" / "abcd"
        directory.mkdir(parents=True)
        sources.write_source(directory, "abcd", "1", {"items": TABLE}, [], {})
        (directory / "items.parquet").unlink()
        with pytest.raises(
            FileNotFoundError, match="items.parquet is missing; run: .*--source abcd"
        ):
            sources.read_source(tmp_path, "abcd")

    def test_another_sources_manifest_is_refused(self, tmp_path):
        """A manifest that names another source is not read as this one's."""
        directory = tmp_path / "sources" / "abcd"
        directory.mkdir(parents=True)
        written = sources.write_source(directory, "abcd", "1", {"items": TABLE}, [], {})
        (directory / "manifest.json").write_text(
            json.dumps({**written, "source": "efgh"})
        )
        with pytest.raises(ValueError, match="manifest of source 'efgh'"):
            sources.read_source(tmp_path, "abcd")

    def test_a_manifest_without_its_files_names_the_command(self, tmp_path):
        """A manifest that records no files is an error with the fetch command."""
        directory = tmp_path / "sources" / "abcd"
        directory.mkdir(parents=True)
        written = sources.write_source(directory, "abcd", "1", {"items": TABLE}, [], {})
        del written["files"]
        (directory / "manifest.json").write_text(json.dumps(written))
        with pytest.raises(ValueError, match="KeyError.*run: .*--source abcd"):
            sources.read_source(tmp_path, "abcd")


def fictional_source():
    """A fictional source of one table, read from the manifest's own pins."""
    source = MagicMock()
    source.fetch_source.side_effect = lambda downloads, manifest: (
        {"items": TABLE},
        manifest["sources"],
        {},
    )
    return source


class TestFetch:
    """Fetching sources against their committed manifests."""

    @pytest.fixture
    def folder(self, monkeypatch, tmp_path):
        """Pins-only manifests for two fictional sources, abcd and efgh."""
        folder = tmp_path / "manifests"
        folder.mkdir()
        for name in ("abcd", "efgh"):
            pin = {"role": "release", "name": f"{name} release", "version": "1"}
            (folder / f"{name}.json").write_text(
                json.dumps({"version": "1", "files": {}, "sources": [pin]})
            )
        monkeypatch.setattr(sources, "MANIFESTS", folder)
        monkeypatch.setattr(sources.manifests, "committed_version", lambda *_: None)
        return folder

    def test_a_first_fetch_needs_accepting(self, folder, monkeypatch, tmp_path):
        """Against a pins-only manifest, a fetch differs until accepted, then matches."""
        monkeypatch.setitem(sources.FETCHERS, "abcd", fictional_source())
        inputs = tmp_path / "inputs"
        assert sources.fetch(["abcd"], inputs) == 1
        assert sources.fetch(["abcd"], inputs, update_manifest=True) == 0
        assert (inputs / "sources" / "abcd" / "items.parquet").exists()
        assert sources.fetch(["abcd"], inputs) == 0

    def test_a_failing_source_does_not_stop_the_others(
        self, folder, monkeypatch, capsys, tmp_path
    ):
        """A source that cannot be fetched is reported, and the next one is still fetched."""
        broken = MagicMock()
        broken.fetch_source.side_effect = FileNotFoundError("no abcd download")
        monkeypatch.setitem(sources.FETCHERS, "abcd", broken)
        monkeypatch.setitem(sources.FETCHERS, "efgh", fictional_source())
        inputs = tmp_path / "inputs"
        assert sources.fetch(["abcd", "efgh"], inputs, update_manifest=True) == 1
        assert "abcd not fetched: FileNotFoundError: no abcd download" in (
            capsys.readouterr().err
        )
        assert (inputs / "sources" / "efgh" / "items.parquet").exists()


class TestCommandLine:
    """The `sources fetch` command."""

    def test_an_unknown_source_is_an_error(self):
        """`sources fetch --source` accepts only the sources there are."""
        with pytest.raises(SystemExit):
            main(["sources", "fetch", "--source", "abcd"])

    def test_a_source_that_cannot_be_fetched_is_reported(
        self, monkeypatch, capsys, tmp_path
    ):
        """A source that cannot be fetched names what is missing, and exits 1."""
        monkeypatch.delenv("LOINC_USERNAME", raising=False)
        assert (
            main(["sources", "fetch", "--source", "loinc", "--inputs", str(tmp_path)])
            == 1
        )
        assert "LOINC_USERNAME" in capsys.readouterr().err
