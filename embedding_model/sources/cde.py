"""The NIH CDE Repository: data elements with their names, definitions and values.

The repository serves only its current state, through a one-call export of
every element: the export that ``ai_harmonization.cde`` also uses, with retired
elements included. Each element keeps its registration status, such as
``Retired``. The export is kept, so a later fetch reads the same elements. Its
count must equal the total that the repository's search reports, or the export
is not kept.
"""

import json

import pandas as pd

from embedding_model.download import downloaded, fetch, modified
from embedding_model.sources.common import open_where
from embedding_model.manifests import file_sha256

EXPORT = "NIH CDE Repository export"
SEARCH = "https://cde.nlm.nih.gov/server/de/search"
# No copyrightStatus value is known to be closed, so a new value is reported,
# and not open until decided.
OPEN_COPYRIGHT = {"Public domain, free to use"}
CLOSED_COPYRIGHT = set()
# The terms of use, as an SPDX license expression. The terms are in
# embedding_model/docs/licenses.md, which CITATION points to.
LICENSE = "LicenseRef-NLM-Web-Policies"
CITATION = "embedding_model/docs/licenses.md#nih-cde"


def search_total():
    """How many elements the repository's search reports, retired ones included."""
    query = {"searchTerm": "", "resultPerPage": 1, "page": 1, "includeRetired": True}
    return json.loads(fetch(SEARCH, json=query))["totalItems"]


def export(path, url, query):
    """The repository's export, downloaded once and kept.

    Args:
        path (Path): Where the export is kept.
        url (str): The export's address.
        query (dict): The export's query, POSTed as JSON.

    Returns:
        list[dict]: The elements.

    Raises:
        ValueError: If a new export's count is not the search's total.
            Nothing is kept, so the next fetch tries again.
    """
    if path.exists():
        return json.loads(path.read_bytes())
    total = search_total()
    read = []

    def check(data):
        """Read the export, and raise unless it holds the search's total."""
        read.append(json.loads(data))
        if len(read[0]) != total:
            raise ValueError(
                f"the export holds {len(read[0]):,} elements and the search "
                f"reports {total:,}; nothing was kept, try again later"
            )

    downloaded(url, path, json=query, check=check)
    return read[0]


def read_export(elements):
    """The export's elements, names, definitions, ids, values and properties.

    An element is open when its ``copyrightStatus`` is public domain and its
    ``noRenderAllowed`` is not true.

    Args:
        elements (list[dict]): The export's records.

    Returns:
        dict[str, pd.DataFrame]: ``elements``, ``designations``,
            ``definitions``, ``ids``, ``values`` and ``properties``, keyed by
            ``tiny_id``, with each list's ``position``.
    """
    rows = {
        "elements": [],
        "designations": [],
        "definitions": [],
        "ids": [],
        "values": [],
        "properties": [],
    }
    for element in elements:
        tiny = element["tinyId"]
        state = element.get("registrationState") or {}
        domain = element.get("valueDomain") or {}
        rows["elements"].append(
            {
                "tiny_id": tiny,
                "steward": (element.get("stewardOrg") or {}).get("name", ""),
                "registration_status": state.get("registrationStatus", ""),
                "administrative_status": state.get("administrativeStatus", ""),
                "copyright_status": element.get("copyrightStatus") or "",
                "no_render_allowed": bool(element.get("noRenderAllowed")),
                "datatype": domain.get("datatype", ""),
            }
        )
        for kind, field in (
            ("designations", "designation"),
            ("definitions", "definition"),
        ):
            for position, entry in enumerate(element.get(kind) or []):
                rows[kind].append(
                    {
                        "tiny_id": tiny,
                        "position": position,
                        field: entry.get(field, ""),
                        "tags": "|".join(entry.get("tags") or []),
                    }
                )
        for entry in element.get("ids") or []:
            rows["ids"].append(
                {
                    "tiny_id": tiny,
                    "source": entry.get("source", ""),
                    "id": entry.get("id", ""),
                    "version": entry.get("version", ""),
                }
            )
        for position, entry in enumerate(domain.get("permissibleValues") or []):
            rows["values"].append(
                {
                    "tiny_id": tiny,
                    "position": position,
                    "permissible_value": entry.get("permissibleValue", ""),
                    "value_meaning_name": entry.get("valueMeaningName", ""),
                    "value_meaning_code": entry.get("valueMeaningCode", ""),
                    "value_meaning_definition": entry.get("valueMeaningDefinition", ""),
                    "code_system_name": entry.get("codeSystemName", ""),
                }
            )
        for entry in element.get("properties") or []:
            rows["properties"].append(
                {
                    "tiny_id": tiny,
                    "key": entry.get("key", ""),
                    "value": str(entry.get("value", "")),
                }
            )
    tables = {table: pd.DataFrame(found) for table, found in rows.items()}
    found = tables["elements"]
    found["open"] = (
        open_where(
            found.copyright_status, OPEN_COPYRIGHT, CLOSED_COPYRIGHT, "copyrightStatus"
        )
        & ~found.no_render_allowed
    )
    keys = {
        "elements": ["tiny_id"],
        "designations": ["tiny_id", "position"],
        "definitions": ["tiny_id", "position"],
        "ids": ["tiny_id", "source", "id", "version"],
        "values": ["tiny_id", "position"],
        "properties": ["tiny_id", "key", "value"],
    }
    return {
        table: frame.sort_values(keys[table], ignore_index=True)
        for table, frame in tables.items()
    }


def fetch_source(downloads, manifest):
    """Fetch the repository's export, or read the one kept.

    Args:
        downloads (Path): Where downloaded files are kept between fetches.
        manifest (dict): The committed manifest. The export is not pinned:
            the repository serves only its current state.

    Returns:
        tuple[dict[str, pd.DataFrame], list[dict], dict]: The tables, the
            manifest's sources, and the counts that the fetch prints.
    """
    # Importing ai_harmonization.cde also imports langchain, which takes several
    # seconds. Importing it here, not at the top of this module, means only an
    # NIH CDE fetch waits for it, not every command.
    from ai_harmonization.cde import SEARCH_EXPORT_QUERY, SEARCH_EXPORT_URL

    path = downloads / "SearchExport.json"
    query = {**SEARCH_EXPORT_QUERY, "includeRetired": True}
    elements = export(path, SEARCH_EXPORT_URL, query)
    tables = read_export(elements)
    found = tables["elements"]
    counts = {"elements": len(found), "not_open": int((~found.open).sum())}
    print(f"  {counts['elements']:,} elements, {counts['not_open']:,} not open")
    source = {
        "role": "service",
        "name": EXPORT,
        "location": SEARCH_EXPORT_URL,
        "fetched": modified(path),
        "sha256": file_sha256(path),
        "license": LICENSE,
        "citation": CITATION,
    }
    return tables, [source], counts
