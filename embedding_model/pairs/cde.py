"""The NIH CDE Repository: each data element, around its first designation.

The source side holds the element's other designations, whatever their tags.
The word "Deprecated" is removed from those that start with it. A designation
is a question text when a tag names a question, such as ``Preferred Question
Text``, or when it ends with a question mark.

The description is the first definition, and the other definitions are paired
with it. NCI's elements copy their definitions from caDSR, so they are cleaned
in the same way as caDSR's.

Each value is its meaning, or its permissible value when it has no meaning, in
the element's order. The codes written before the values, as in ``0= None``,
are removed as ``common.value_labels`` says. The type is the element's own
datatype, such as ``Value List``.
"""

from collections import defaultdict

from embedding_model.pairs.common import (
    CURRENT,
    RETIRED,
    Item,
    Text,
    cadsr_definition,
    undeprecated,
    value_labels,
)
from embedding_model.sources.common import report_new

# The NIH CDE Repository also lists some of caDSR's elements, with NCI as their
# steward. Their definitions are caDSR's, so they go through cadsr_definition(),
# which removes caDSR's curation marks and turns its concept joins into spaces.
# Other stewards' definitions get the usual cleaning only.
CADSR_STEWARD = "NCI"
# The registration statuses that the repository's guides list, and Retired.
STATUSES = {
    "Incomplete": CURRENT,
    "Candidate": CURRENT,
    "Recorded": CURRENT,
    "Qualified": CURRENT,
    "Standard": CURRENT,
    "Preferred Standard": CURRENT,
    "Retired": RETIRED,
}


def by_element(frame, column):
    """A table's texts by element, in the element's own order."""
    ordered = frame.sort_values(["tiny_id", "position"], kind="stable")
    return ordered.groupby("tiny_id")[column].agg(list).to_dict()


def items(tables):
    """The repository's data elements as items.

    Args:
        tables (dict[str, pd.DataFrame]): The tables ``sources fetch`` wrote.

    Returns:
        list[Item]: One item per element.
    """
    elements = tables["elements"]
    report_new(
        elements.registration_status,
        set(STATUSES),
        "registrationStatus",
        "read as current",
    )
    designations = tables["designations"].sort_values(
        ["tiny_id", "position"], kind="stable"
    )
    names = defaultdict(list)
    for row in designations.itertuples():
        designation = undeprecated(row.designation)
        if designation.strip():
            asks = designation.rstrip().endswith("?")
            question = "question" in row.tags.casefold() or asks
            names[row.tiny_id].append(
                Text(designation, row.tags or "designation", question)
            )
    definitions = tables["definitions"]
    copied = definitions.tiny_id.isin(
        elements.tiny_id[elements.steward == CADSR_STEWARD]
    )
    definitions = definitions.assign(
        definition=definitions.definition.where(
            ~copied, definitions.definition.map(cadsr_definition)
        )
    )
    described = by_element(definitions, "definition")
    values = tables["values"]
    labels = values.value_meaning_name.where(
        values.value_meaning_name != "", values.permissible_value
    )
    listed = {
        element: value_labels(found)
        for element, found in by_element(values.assign(label=labels), "label").items()
    }
    found = []
    for row in elements.itertuples():
        own = names[row.tiny_id]
        definitions = described.get(row.tiny_id, [])
        found.append(
            Item(
                id=row.tiny_id,
                canonical=own[0].text if own else "",
                type=row.datatype,
                description=next(iter(definitions), ""),
                descriptions=definitions[1:],
                lists=[("", listed[row.tiny_id])] if row.tiny_id in listed else [],
                texts=own[1:],
                status=STATUSES.get(row.registration_status, CURRENT),
                open=row.open,
            )
        )
    return found
