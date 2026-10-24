"""NCIt: each concept, around its preferred name.

The source side holds the concept's synonyms of the same meaning: NCIt's own
synonyms (SY), other sources' preferred terms (PT), antiquated terms (AQ, AQS),
display names (DN), abbreviations (AB), brand names (BR, FB) and country codes,
such as ``USA`` (CA2, CA3, CS). Number codes (CNU) are not names, so they give
no pairs.

The description is the concept's first NCI definition. When there is none, it
is the first definition from another source. The other definitions are paired
with it. NCIt can end a definition with its source, such as "(NCI)", and this
source is removed.

A concept with value lists gives its pairs once for each distinct list. NCIt
gives the values no order, so they are sorted by name.

EVS's history can name a replacement for a retired or obsolete concept. The
concept then gives its names to the replacement.
"""

import re
from collections import defaultdict

from embedding_model.pairs.common import CURRENT, DRAFT, RETIRED, Item, Text, clean
from embedding_model.sources.common import report_new

# Synonyms (SY), preferred terms (PT), antiquated terms (AQ, AQS), display
# names (DN), abbreviations (AB), brand names (BR, FB) and country codes (CA2,
# CA3, CNU, CS).
SAME_MEANING = {
    "SY",
    "PT",
    "AQ",
    "AQS",
    "DN",
    "AB",
    "BR",
    "FB",
    "CA2",
    "CA3",
    "CNU",
    "CS",
}
# Grammar forms (AD), headers (HD), chemical structures (SN) and drug study
# codes (CN).
NOT_SAME_MEANING = {"AD", "HD", "SN", "CN"}
RETIRED_STATUSES = {"Retired_Concept", "Obsolete_Concept"}
DRAFT_STATUSES = {"Provisional_Concept", "Concept_Pending_Approval"}
CURRENT_STATUSES = {"", "Header_Concept"}
# A source at the end of a definition, as in "(NCI)" or "(NCI04)".
MARKER = re.compile(r"\s*\(([^()]+)\)\s*$")


def status_of(statuses):
    """A concept's status, from NCIt's ``Concept_Status`` values joined by ``|``."""
    found = set(statuses.split("|"))
    if found & RETIRED_STATUSES:
        return RETIRED
    if found & DRAFT_STATUSES:
        return DRAFT
    return CURRENT


def without_source(definition, sources):
    """A definition without a trailing source of NCIt's, such as ``(NCI)``.

    A source with a two-digit year, such as ``(NCI04)``, is removed too. A source
    inside a parenthesis left open, as in ``(for the Abcd (NCI)``, is kept.

    Args:
        definition (str): The definition.
        sources (set[str]): The definition sources that NCIt lists, in upper
            case.

    Returns:
        str: The definition without its source.
    """
    found = MARKER.search(definition)
    if found:
        marker = found.group(1).upper()
        dated = marker[-2:].isdigit() and marker[:-2] in sources
        before = definition[: found.start()]
        left_open = before.count("(") > before.count(")")
        if not left_open and (marker in sources or dated):
            return before
    return definition


def items(tables):
    """NCIt's concepts as items.

    Args:
        tables (dict[str, pd.DataFrame]): The tables ``sources fetch`` wrote.

    Returns:
        list[Item]: One item per concept.
    """
    concepts = tables["concepts"]
    synonyms = tables["synonyms"]
    report_new(
        synonyms.term_type, SAME_MEANING | NOT_SAME_MEANING, "term_type", "not used"
    )
    report_new(
        concepts.status.str.split("|").explode(),
        RETIRED_STATUSES | DRAFT_STATUSES | CURRENT_STATUSES,
        "Concept_Status",
        "read as current",
    )
    texts = defaultdict(list)
    for row in synonyms[synonyms.term_type.isin(SAME_MEANING)].itertuples():
        texts[row.code].append(Text(row.name, row.term_type))
    definitions = tables["definitions"]
    ranked = definitions.assign(
        rank=(definitions.kind != "DEFINITION") * 2 + (definitions.source != "NCI")
    ).sort_values(["code", "rank"], kind="stable")
    sources = set(definitions.source.str.upper()) - {""}
    described = defaultdict(list)
    for row in ranked.itertuples():
        described[row.code].append(without_source(row.definition, sources))
    lists = defaultdict(dict)
    for (code, association), group in tables["values"].groupby(["code", "association"]):
        found = set(map(clean, group.value)) - {""}
        values = tuple(sorted(found, key=lambda value: (value.casefold(), value)))
        if values:
            lists[code].setdefault(values, association)
    found = tables["replacements"]
    replaced_by = defaultdict(list)
    for row in found[found.replacement_code != ""].itertuples():
        replaced_by[row.code].append(row.replacement_code)
    return [
        Item(
            id=row.code,
            canonical=row.preferred_name,
            description=next(iter(described[row.code]), ""),
            descriptions=described[row.code][1:],
            lists=[(name, list(values)) for values, name in lists[row.code].items()],
            texts=texts[row.code],
            status=status_of(row.status),
            replaced_by=replaced_by[row.code],
            open=row.open,
        )
        for row in concepts.itertuples()
    ]
