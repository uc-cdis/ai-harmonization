"""PhenX: each variable, around its description.

Only the variables of protocols that are visible and available are used. A
hidden protocol is an older version of a visible one. A variable is left out
too when the export does not list its protocol.

The canonical is the variable's description, without the quotes around some
of them. Its type is PhenX's own. The source side holds the variable's name,
and each dbGaP study variable that PhenX maps to it as comparable: its name
and its description, each with its studies. Variables that PhenX calls only
related do not mean the same thing, so they are not used.
"""

import re
from collections import defaultdict

from embedding_model.pairs.common import Item, Text, clean, is_name

COMPARABLE = "comparable"
# A code in a value list: a number or a range, as in "9 = Refused", "None = 0"
# or "1 - 7 = times", or else one short token, as in "? = unknown".
NUMBER = re.compile(r"^\(?-?\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\)?$")
TOKEN = re.compile(r"^\S{1,12}$")
# What separates the entries of a value list: "|", a vertical tab, or a comma
# before a coded entry, as in "1 = Yes,2 = No".
ENTRIES = re.compile(r"[|\v]|,(?=\s*\(?-?\d+\)?\s*=)")
# A range of numbers, as in "1900..2011", which lists no values.
RANGE = re.compile(r"^\s*-?[\d.]+\s*\.\.\s*-?[\d.]+\s*$")


def label(entry):
    """An entry's label, without its code, whichever side of "=" the code is on.

    The code is a number or a range, else a short token that is not a name,
    such as ``?`` or ``n/t``, else a token in capitals on the left, such as
    ``NA``. Otherwise the left side is the label and the right side explains
    it, as in ``Yes = Classify as a user``. The code can be "=" itself, as in
    ``= = Cannot be assessed``.

    Args:
        entry (str): One entry, such as ``1 = Never`` or ``Never = 1``.

    Returns:
        str: The label, such as ``Never``.
    """
    separator = " = " if " = " in entry else "="
    left, equals, right = (side.strip() for side in entry.partition(separator))
    if not equals:
        return entry
    if not left:
        return right.removeprefix("=").strip()
    sides = ((left, right), (right, left))
    for code, found in sides:
        if NUMBER.match(code):
            return found
    for code, found in sides:
        if TOKEN.match(code) and not is_name(code):
            return found
    if TOKEN.match(left) and left.isupper():
        return right
    return left


def labels(cell):
    """The labels of a PhenX value list, such as ``9 = Refused|7 = Don't Know``.

    PhenX writes a value list as free text, in no standard format, so it is
    split here. Entries are separated by ``|``, and some by a vertical tab or a
    comma. Each is cleaned before its code is found. A range of numbers, such as
    ``0..110``, lists no values.

    Args:
        cell (str): The variable's ``variable_values``.

    Returns:
        list[str]: The labels, in the list's order.
    """
    entries = [clean(entry) for entry in ENTRIES.split(cell)]
    return [label(entry) for entry in entries if entry and not RANGE.match(entry)]


def unquoted(text):
    """A description without the quotes around it, as in ``"How often?"``."""
    found = text.strip()
    if found.startswith('"') and found.endswith('"') and found.count('"') == 2:
        return found[1:-1]
    return text


def study_texts(mappings):
    """Each variable's dbGaP names and descriptions, each with its studies.

    Args:
        mappings (pd.DataFrame): The comparable mappings.

    Returns:
        dict[str, list[Text]]: The texts, by PhenX variable id.
    """
    studies = defaultdict(dict)
    for row in mappings.itertuples():
        for text, term_type in (
            (row.gap_variable_name, "dbGaP variable name"),
            (row.gap_variable_desc, "dbGaP variable description"),
        ):
            found = studies[row.variable_id].setdefault((text, term_type), set())
            found.add(row.gap_study_accession)
    return {
        variable: [
            Text(text, term_type, study="|".join(sorted(found)))
            for (text, term_type), found in texts.items()
        ]
        for variable, texts in studies.items()
    }


def items(tables):
    """PhenX's variables of visible protocols as items.

    Args:
        tables (dict[str, pd.DataFrame]): The tables ``sources fetch`` wrote.

    Returns:
        list[Item]: One item per variable of a visible protocol.
    """
    variables = tables["variables"]
    used = variables[variables.protocol_visible == "1"]
    mappings = tables["mappings"]
    mapped = study_texts(mappings[mappings.map_level == COMPARABLE])
    found = []
    for row in used.itertuples():
        values = labels(row.variable_values)
        found.append(
            Item(
                id=row.id,
                canonical=unquoted(row.description),
                type=row.type,
                lists=[("", values)] if values else [],
                texts=[Text(row.name, "variable name"), *mapped.get(row.id, [])],
                open=row.open,
            )
        )
    return found
