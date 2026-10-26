"""NCIt: concepts, names, definitions and value lists, from its OWL release.

A release is pinned by its version, such as ``26.09d``, and read from EVS's
archive. Every concept is kept, retired ones too, with its status. For each
retired or obsolete concept, EVS's replacements service says what happened to
it, such as a merge, and which concept replaced it, if any.
"""

import itertools
import json
import zipfile

import pandas as pd

from embedding_model.download import downloaded, fetch, file_sha256, keep, modified
from embedding_model.sources.common import pinned
from embedding_model.sources.owl import DEFINED_PARENT, RDFS, grouped, read_owl

RELEASE = "NCIt OWL release"
HISTORY = "NCIt history: replacements of retired concepts"
ARCHIVE = (
    "https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/{version}_Release/"
    "Thesaurus_{version}.OWL.zip"
)
REPLACEMENTS = "https://api-evsrest.nci.nih.gov/api/v1/history/ncit/replacements"
# How many concepts one call to the replacements service asks about. The codes go
# in the call's URL, so a batch keeps the URL short and the calls few.
HISTORY_BATCH = 100
LICENSE = (
    "CC BY 4.0 (https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.htm)"
)
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
RETIRED = {"Retired_Concept", "Obsolete_Concept"}


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
    found = read_owl(stream)
    concepts = {
        iri[len(NCIT) :]: grouped(annotations)
        for iri, annotations in found["classes"].items()
        if iri.startswith(NCIT)
    }
    name = {code: first(values, PREFERRED_NAME) for code, values in concepts.items()}
    lists = {}
    for iri, annotations in found["properties"].items():
        list_name = first(grouped(annotations), PREFERRED_NAME)
        if iri.startswith(NCIT) and is_value_list(list_name):
            lists[tag(iri[len(NCIT) :])] = list_name

    def axioms(code, prop, text):
        """The axioms annotating one statement, or one empty axiom.

        A statement without an axiom then still gives one row.
        """
        return found["axioms"].get((NCIT + code, NCIT + prop, text)) or [{}]

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
        table: pd.DataFrame(found_rows).sort_values(keys[table], ignore_index=True)
        for table, found_rows in rows.items()
    }


def replacements(codes, path):
    """What replaced each retired concept, from EVS's replacements service.

    The service's answers are kept in one file, so a later fetch reads them
    back.

    Args:
        codes (list[str]): The retired and obsolete concepts' codes.
        path (Path): Where the entries are kept, as JSON.

    Returns:
        pd.DataFrame: One row per entry: ``code``, ``action`` (such as
            ``merge`` or ``retire``), ``date``, and ``replacement_code`` and
            ``replacement_name`` where the concept has a replacement.
    """
    if not path.exists():
        entries = []
        for batch in itertools.batched(codes, HISTORY_BATCH):
            entries += json.loads(fetch(f"{REPLACEMENTS}?list={','.join(batch)}"))
        keep(path, (json.dumps(entries, indent=1) + "\n").encode("utf-8"))
    columns = ["code", "action", "date", "replacementCode", "replacementName"]
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    frame = frame.reindex(columns=columns).fillna("").astype(str)
    frame.columns = ["code", "action", "date", "replacement_code", "replacement_name"]
    return frame.sort_values(list(frame.columns), ignore_index=True)


def fetch_source(downloads, manifest):
    """Fetch the NCIt release that the manifest pins, and its replacements.

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
        member = next(n for n in bundle.namelist() if n.lower().endswith(".owl"))
        with bundle.open(member) as stream:
            tables = read_ncit(stream)
    concepts = tables["concepts"]
    retired = concepts[
        concepts.status.str.split("|").map(
            lambda statuses: bool(RETIRED & set(statuses))
        )
    ].code.tolist()
    kept = downloads / f"replacements_{version}.json"
    found = replacements(retired, kept)
    tables["replacements"] = found
    counts = {
        "concepts": len(concepts),
        "retired_or_obsolete": len(retired),
        "replaced": int(found[found.replacement_code != ""].code.nunique()),
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
            "role": "service",
            "name": HISTORY,
            "location": REPLACEMENTS,
            "fetched": modified(kept),
            "sha256": file_sha256(kept),
            "license": LICENSE,
            "citation": CITATION,
        },
    ]
    return tables, sources, counts
