"""NCIt: concepts, names, definitions and value lists, from its OWL release.

A release is pinned by its version, such as ``26.09d``, and read from EVS's
archive. Every concept is kept, retired ones too, with its status. The
release's history file, in the same folder, says which concept replaced each
retired or obsolete concept, if any.
"""

import csv
import zipfile

import pandas as pd

from embedding_model.download import downloaded
from embedding_model.manifests import file_sha256, pinned
from embedding_model.sources.owl import DEFINED_PARENT, RDFS, grouped, read_owl

RELEASE = "NCIt OWL release"
HISTORY = "NCIt history file"
ARCHIVE = (
    "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/{version}_Release/"
    "Thesaurus_{version}.OWL.zip"
)
# The release's history of concepts created, merged, split or retired.
HISTORY_ARCHIVE = (
    "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/{version}_Release/"
    "subset_cumulative_history_{version}.zip"
)
# The terms of use, as an SPDX license expression. The terms are in
# embedding_model/docs/licenses.md, which CITATION points to.
LICENSE = "CC-BY-4.0"
CITATION = "embedding_model/docs/licenses.md#ncit"
NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
# NCIt's own annotation properties, by their codes in the OWL file.
PREFERRED_NAME = "P108"
DISPLAY_NAME = "P107"
FULL_SYN = "P90"
TERM_TYPE = "P383"
TERM_SOURCE = "P384"
SOURCE_CODE = "P385"
SUBSOURCE = "P386"
DEFINITION = "P97"
ALT_DEFINITION = "P325"
DEFINITION_SOURCE = "P378"
CONCEPT_STATUS = "P310"
SEMANTIC_TYPE = "P106"
IN_SUBSET = "A8"
RETIRED_STATUSES = {"Retired_Concept", "Obsolete_Concept"}


def tag(code):
    """The XML tag of one of NCIt's properties, such as ``P90``."""
    return "{" + NCIT + "}" + code


def first(values, code):
    """The first value of a property, stripped, or ``""``."""
    return (values.get(tag(code)) or [""])[0].strip()


def joined(values):
    """Values stripped and joined by ``|``, in the order given."""
    return "|".join(value.strip() for value in values)


def is_value_list(name):
    """Whether an association gives a concept's permissible values.

    Only associations named ``Has_...Value`` list values, such as
    ``Has_GDC_Value`` or ``Has_PCDC_AML_Authorized_Value``.

    Args:
        name (str): The association's name, its ``Preferred_Name``.

    Returns:
        bool: Whether it is a value list.
    """
    return name.startswith("Has_") and name.endswith("Value")


def read_ncit(stream):
    """NCIt's concepts, synonyms, definitions and value lists.

    Args:
        stream (file): ``Thesaurus.owl``, opened in binary mode.

    Returns:
        dict[str, pd.DataFrame]: ``concepts`` (one row per concept, with its
            preferred name, display names, status, semantic types, subsets
            and parents), ``synonyms`` (each synonym, NCIt's ``FULL_SYN``,
            with its term type, source, source code and subsource),
            ``definitions`` (each with its kind and source) and ``values``
            (each value of each value list, with the list's association
            name). Several values of one field are joined by ``|``. Rows are
            sorted, so a table's content does not depend on the file's order.
    """
    owl = read_owl(stream)
    concepts = {
        iri[len(NCIT) :]: grouped(annotations)
        for iri, annotations in owl["classes"].items()
        if iri.startswith(NCIT)
    }
    name = {code: first(values, PREFERRED_NAME) for code, values in concepts.items()}
    lists = {}
    for iri, annotations in owl["properties"].items():
        list_name = first(grouped(annotations), PREFERRED_NAME)
        if iri.startswith(NCIT) and is_value_list(list_name):
            lists[tag(iri[len(NCIT) :])] = list_name

    def axioms(code, prop, text):
        """The axioms annotating one statement, or one empty axiom.

        A statement without an axiom then still gives one row.
        """
        return owl["axioms"].get((NCIT + code, NCIT + prop, text)) or [{}]

    rows = {"concepts": [], "synonyms": [], "definitions": [], "values": []}
    for code, values in concepts.items():
        parents = [
            parent[len(NCIT) :]
            for parent in values.get(RDFS + "subClassOf", [])
            + values.get(DEFINED_PARENT, [])
            if parent.startswith(NCIT)
        ]
        rows["concepts"].append(
            {
                "code": code,
                "preferred_name": name[code],
                "display_names": joined(values.get(tag(DISPLAY_NAME), [])),
                "status": joined(values.get(tag(CONCEPT_STATUS), [])),
                "semantic_types": joined(values.get(tag(SEMANTIC_TYPE), [])),
                "subsets": joined(
                    name.get(s[len(NCIT) :], s) for s in values.get(tag(IN_SUBSET), [])
                ),
                "parents": joined(dict.fromkeys(parents)),
                "open": True,
            }
        )
        for text in values.get(tag(FULL_SYN), []):
            for axiom in axioms(code, FULL_SYN, text):
                rows["synonyms"].append(
                    {
                        "code": code,
                        "name": text.strip(),
                        "term_type": first(axiom, TERM_TYPE),
                        "source": first(axiom, TERM_SOURCE),
                        "source_code": first(axiom, SOURCE_CODE),
                        "subsource": first(axiom, SUBSOURCE),
                    }
                )
        for prop, kind in (
            (DEFINITION, "DEFINITION"),
            (ALT_DEFINITION, "ALT_DEFINITION"),
        ):
            for text in values.get(tag(prop), []):
                for axiom in axioms(code, prop, text):
                    rows["definitions"].append(
                        {
                            "code": code,
                            "definition": text.strip(),
                            "kind": kind,
                            "source": first(axiom, DEFINITION_SOURCE),
                        }
                    )
        for prop, list_name in lists.items():
            for target in values.get(prop, []):
                value = target[len(NCIT) :]
                rows["values"].append(
                    {
                        "code": code,
                        "association": list_name,
                        "value_code": value,
                        "value": name.get(value, ""),
                    }
                )
    keys = {
        "concepts": ["code"],
        "synonyms": ["code", "name", "term_type", "source", "source_code", "subsource"],
        "definitions": ["code", "kind", "source", "definition"],
        "values": ["code", "association", "value", "value_code"],
    }
    return {
        table: pd.DataFrame(table_rows).sort_values(keys[table], ignore_index=True)
        for table, table_rows in rows.items()
    }


def read_history(path):
    """The records of a release's history file, one row each.

    EVS's ReadMe_history.txt names the fields: the concept's code, the edit's
    action and date, and a reference code. The file writes them around an
    empty second field, and ends each line with a separator.

    Args:
        path (Path): The history file's zip.

    Returns:
        pd.DataFrame: ``code``, ``action``, ``date`` and ``reference``.

    Raises:
        ValueError: If the zip holds no history file, or its lines have
            another number of fields.
    """
    with zipfile.ZipFile(path) as bundle:
        members = [n for n in bundle.namelist() if n.lower().endswith(".txt")]
        if not members:
            raise ValueError(
                f"{path} holds no history file; delete it, then run:"
                " python -m embedding_model sources fetch --source ncit"
            )
        with bundle.open(members[0]) as stream:
            records = pd.read_csv(
                stream,
                sep="|",
                header=None,
                dtype=str,
                keep_default_na=False,
                quoting=csv.QUOTE_NONE,
            )
    if records.shape[1] != 6:
        raise ValueError(
            f"{path}: a history record has 6 fields, and these lines have"
            f" {records.shape[1]}"
        )
    records = records[[0, 2, 3, 4]]
    records.columns = ["code", "action", "date", "reference"]
    return records


def replacements(codes, history, names):
    """What replaced each retired concept, from the release's history file.

    In a merge, one concept survives and the other retires. Each of the two
    gets a merge record that names the survivor. So a merge record of a
    retired concept names its replacement. A retire record names the
    concept's old parent instead, as EVS's ReadMe_history.txt says, so it
    names no replacement.

    Args:
        codes (list[str]): The retired and obsolete concepts' codes.
        history (pd.DataFrame): The history file's records, from
            ``read_history``.
        names (dict[str, str]): Each concept's preferred name, by code.

    Returns:
        pd.DataFrame: One row per merge or retire record of a retired
            concept: ``code``, ``action``, ``date`` (as ``YYYY-MM-DD``), and
            ``replacement_code`` and ``replacement_name`` where the record
            names a replacement.
    """
    records = history[
        history.code.isin(set(codes)) & history.action.isin(["merge", "retire"])
    ]
    merged = (
        (records.action == "merge")
        & ~records.reference.isin(["", "null"])
        & (records.reference != records.code)
    )
    replacement = records.reference.where(merged, "")
    frame = pd.DataFrame(
        {
            "code": records.code,
            "action": records.action,
            "date": pd.to_datetime(records.date, format="%d-%b-%y").dt.strftime(
                "%Y-%m-%d"
            ),
            "replacement_code": replacement,
            "replacement_name": replacement.map(names).fillna(""),
        }
    )
    return frame.sort_values(list(frame.columns), ignore_index=True)


def fetch_source(downloads, manifest):
    """Fetch the NCIt release that the manifest pins, and its history file.

    Args:
        downloads (Path): Where downloaded files are kept between fetches.
        manifest (dict): The committed manifest; its release entry gives the
            version.

    Returns:
        tuple[dict[str, pd.DataFrame], list[dict], dict]: The tables, with
            ``replacements`` added to ``read_ncit``'s, the manifest's sources,
            and the counts that the fetch prints.
    """
    version = pinned(manifest, RELEASE)["version"]
    url = ARCHIVE.format(version=version)
    path = downloaded(url, downloads / f"Thesaurus_{version}.OWL.zip")
    with zipfile.ZipFile(path) as bundle:
        members = [n for n in bundle.namelist() if n.lower().endswith(".owl")]
        if not members:
            raise ValueError(
                f"{path} holds no .owl file; delete it, then run:"
                " python -m embedding_model sources fetch --source ncit"
            )
        with bundle.open(members[0]) as stream:
            tables = read_ncit(stream)
    concepts = tables["concepts"]
    retired = concepts[
        concepts.status.str.split("|").map(
            lambda statuses: bool(RETIRED_STATUSES & set(statuses))
        )
    ].code.tolist()
    history_url = HISTORY_ARCHIVE.format(version=version)
    history_path = downloaded(
        history_url, downloads / f"subset_cumulative_history_{version}.zip"
    )
    names = dict(zip(concepts.code, concepts.preferred_name))
    tables["replacements"] = replacements(retired, read_history(history_path), names)
    replaced = tables["replacements"].replacement_code != ""
    counts = {
        "concepts": len(concepts),
        "retired_or_obsolete": len(retired),
        "replaced": int(tables["replacements"][replaced].code.nunique()),
    }
    print(
        f"  {counts['concepts']:,} concepts, "
        f"{counts['retired_or_obsolete']:,} retired or obsolete, "
        f"{counts['replaced']:,} replaced"
    )
    sources = [
        {
            "role": "release",
            "name": RELEASE,
            "location": url,
            "version": version,
            "sha256": file_sha256(path),
            "license": LICENSE,
            "citation": CITATION,
        },
        {
            "role": "release",
            "name": HISTORY,
            "location": history_url,
            "version": version,
            "sha256": file_sha256(history_path),
            "license": LICENSE,
            "citation": CITATION,
        },
    ]
    return tables, sources, counts
