"""caDSR: NCI's data elements, with their names, question texts and values.

caDSR publishes a daily snapshot of its data elements, as a zip of XML
files. A snapshot is pinned by its date. A snapshot can hold an element's
older versions, and exact copies of a record: only the element's latest version
is kept, once. defusedxml parses the XML: it refuses entity declarations and
external references, which a snapshot has none of.
"""

import zipfile
from decimal import Decimal, InvalidOperation

import pandas as pd
from defusedxml.ElementTree import iterparse

from embedding_model.download import downloaded, file_sha256
from embedding_model.sources.common import pinned, report_new

SNAPSHOT = "caDSR released data elements"
ARCHIVE = (
    "https://cadsr.nci.nih.gov/ftp/caDSR_Downloads/CDE/XML_Archive/"
    "releasedCDEsXML-OD.zip.{date}"
)
LICENSE = (
    "NCI's text is free of copyright unless otherwise indicated; the Proprietary "
    "Measure Permission document marks exceptions "
    "(https://www.cancer.gov/policies/copyright-reuse)"
)
CITATION = "embedding_model/docs/licenses.md#cadsr"
# An element with this reference document is part of a proprietary measure,
# whose owner's permission is needed: caDSR's restriction field.
PROPRIETARY = "Proprietary Measure Permission"
# The reference document types that caDSR is known to use. A new type is
# reported, in case it restricts use, as the Proprietary Measure Permission
# does.
DOCUMENT_TYPES = {
    "AHRQ Form",
    "Activity Diagram",
    "Alternate Question Text",
    "Application Standard Question Text",
    "Associations",
    "BBRB Text",
    "BRIDG Mapping Path",
    "CDASH Alternate Question Text",
    "CDASH Implementation Note",
    "CDASH to SDTM Mapping",
    "CDISC CRF Instruction",
    "CDISC EVS Controlled Terminology Update",
    "CDISC Note",
    "COMMENT",
    "Class Diagram",
    "Coding Instructions",
    "DATA_ELEMENT_SOURCE",
    "DESCRIPTION",
    "Data Collection Template",
    "EXAMPLE",
    "Implementation Note",
    "Instructions",
    "Interaction Overview Diagram",
    "LABEL",
    "Measure Sequence Number",
    "NOTE",
    "NRDS Question Text",
    "OMOP CDM ETL Conventions",
    "OMOP CDM User Guidance",
    "Outcome Measure Implementation",
    "PDF",
    "Preferred Question Text",
    PROPRIETARY,
    "REFERENCE",
    "Reference Information",
    "SDTM Core",
    "Standard Alternate Question Text",
    "Tier",
    "Use Case",
    "Use Case Diagram",
    "VALID_VALUE_SOURCE",
    "VD REFERENCE",
    "ePRO Compliant",
}


def text(element, path):
    """The stripped text at a path under an element, or ``""``."""
    found = element.find(path) if element is not None else None
    return (found.text or "").strip() if found is not None else ""


def version(element):
    """An element's version as the decimal number caDSR stores, such as 2.31.

    Args:
        element (dict): The element's row, with its ``public_id`` and
            ``version``.

    Returns:
        Decimal: The version.

    Raises:
        ValueError: If the version is not a number.
    """
    try:
        return Decimal(element["version"])
    except InvalidOperation as error:
        raise ValueError(
            f"caDSR element {element['public_id']} has version "
            f"{element['version']!r}, not a number"
        ) from error


def records(archive):
    """Every data element record in a snapshot, as a dict of lists of rows.

    Args:
        archive (str | Path | file): The snapshot's zip, or its path.

    Yields:
        dict: One record: ``element`` (one row) and its ``alternate_names``,
            ``documents`` and ``values`` (lists of rows).
    """
    with zipfile.ZipFile(archive) as bundle:
        for member in sorted(n for n in bundle.namelist() if n.endswith(".xml")):
            with bundle.open(member) as stream:
                for _, element in iterparse(stream):
                    if element.tag != "DataElement":
                        continue
                    key = {
                        "public_id": text(element, "PUBLICID"),
                        "version": text(element, "VERSION"),
                    }
                    domain = element.find("VALUEDOMAIN")
                    yield {
                        "element": {
                            **key,
                            "long_name": text(element, "LONGNAME"),
                            "short_name": text(element, "PREFERREDNAME"),
                            "definition": text(element, "PREFERREDDEFINITION"),
                            "workflow_status": text(element, "WORKFLOWSTATUS"),
                            "registration_status": text(element, "REGISTRATIONSTATUS"),
                            "context": text(element, "CONTEXTNAME"),
                            "datatype": text(domain, "Datatype"),
                            "value_domain_type": text(domain, "ValueDomainType"),
                        },
                        "alternate_names": [
                            {
                                **key,
                                "name": text(item, "AlternateName"),
                                "type": text(item, "AlternateNameType"),
                                "context": text(item, "ContextName"),
                                "language": text(item, "Language"),
                            }
                            for item in element.findall(
                                "ALTERNATENAMELIST/ALTERNATENAMELIST_ITEM"
                            )
                        ],
                        "documents": [
                            {
                                **key,
                                "type": text(item, "DocumentType"),
                                "name": text(item, "Name"),
                                "text": text(item, "DocumentText"),
                                "language": text(item, "Language"),
                            }
                            for item in element.findall(
                                "REFERENCEDOCUMENTSLIST/REFERENCEDOCUMENTSLIST_ITEM"
                            )
                        ],
                        "values": [
                            {
                                **key,
                                "position": position,
                                "value": text(item, "VALIDVALUE"),
                                "meaning": text(item, "VALUEMEANING"),
                                "meaning_description": text(item, "MEANINGDESCRIPTION"),
                                "meaning_concepts": text(item, "MEANINGCONCEPTS"),
                                "begin_date": text(item, "PVBEGINDATE"),
                                "end_date": text(item, "PVENDDATE"),
                            }
                            for position, item in enumerate(
                                domain.findall(
                                    "PermissibleValues/PermissibleValues_ITEM"
                                )
                                if domain is not None
                                else []
                            )
                        ],
                    }
                    element.clear()


def read_snapshot(archive):
    """Each data element at its latest version, once, with its lists.

    Versions are compared as decimal numbers, as ``version`` reads them. A
    row repeated within one element's lists is kept once. An element is open
    unless it has a Proprietary Measure Permission document. A document type
    not in DOCUMENT_TYPES is reported.

    Args:
        archive (str | Path | file): The snapshot's zip, or its path.

    Returns:
        dict[str, pd.DataFrame]: ``elements``, ``alternate_names``,
            ``documents`` and ``values``, linked by ``public_id`` and
            ``version``.
    """
    latest = {}
    for record in records(archive):
        element = record["element"]
        held = latest.get(element["public_id"])
        if held is None or version(element) > version(held["element"]):
            latest[element["public_id"]] = record
    tables = {
        "elements": pd.DataFrame([r["element"] for r in latest.values()]),
        **{
            table: pd.DataFrame([row for r in latest.values() for row in r[table]])
            for table in ("alternate_names", "documents", "values")
        },
    }
    documents = tables["documents"]
    report_new(
        documents.type, DOCUMENT_TYPES, "reference document type", "elements stay open"
    )
    proprietary = set(documents[documents.type == PROPRIETARY].public_id)
    tables["elements"]["open"] = ~tables["elements"].public_id.isin(proprietary)
    keys = {
        "elements": ["public_id"],
        "alternate_names": ["public_id", "type", "name", "context", "language"],
        "documents": ["public_id", "type", "name", "text"],
        "values": ["public_id", "position"],
    }
    return {
        table: frame.drop_duplicates().sort_values(keys[table], ignore_index=True)
        for table, frame in tables.items()
    }


def fetch_source(downloads, manifest):
    """Fetch the snapshot that the manifest pins.

    Args:
        downloads (Path): Where downloaded files are kept between fetches.
        manifest (dict): The committed manifest; its snapshot entry gives the
            date.

    Returns:
        tuple[dict[str, pd.DataFrame], list[dict]]: The tables and the
            manifest's sources.
    """
    day = pinned(manifest, SNAPSHOT)["version"]
    url = ARCHIVE.format(date=day)
    path = downloaded(url, downloads / f"releasedCDEsXML-OD.zip.{day}")
    tables = read_snapshot(path)
    found = tables["elements"]
    print(f"  {len(found):,} elements, {(~found.open).sum():,} proprietary")
    return tables, [
        {
            "role": "release",
            "name": SNAPSHOT,
            "location": url,
            "version": day,
            "sha256": file_sha256(path),
            "license": LICENSE,
            "citation": CITATION,
        }
    ]
