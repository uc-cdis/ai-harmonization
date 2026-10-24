"""CDISC: each codelist, around its name, and each term, around its preferred term.

A term sits in several codelists, as one row each, so its rows are merged
into one item per concept code. A codelist is its own item: its canonical is
the Codelist Name, and its values are its terms' preferred terms, in the
file's order. For both, the source side holds the submission values and the
synonyms.

The description is the first definition. CDISC notes a definition's source in
parentheses after its last sentence, and this source is removed. A definition
that ends with a question mark is the wording of a questionnaire item. A note
in parentheses may follow the question mark. Such a definition goes on the
source side as a question text.
"""

import re

from embedding_model.pairs.common import Item, Text, clean

SUBMISSION_VALUE = "CDISC Submission Value"
SYNONYMS = "CDISC Synonym(s)"
DEFINITION = "CDISC Definition"
# CDISC notes a definition's source in parentheses after its last sentence or
# a bracketed reference, as in "A thing. (NCI)" or "A thing. [ICH] (CDISC
# Glossary)". A source can hold parentheses of its own, as "(21 CFR 3(b))".
NOTE = re.compile(r"([.?!\]])\s*\(((?:[^()]|\([^()]*\))*)\)\s*$")
# A source is a few words; a longer note is content.
SOURCE_WORDS = 6
QUESTION = re.compile(r"\?\s*(?:\([^()]*\))?\s*\.?\s*$")


def is_question(definition):
    """Whether a definition is a question's wording: it ends with ``?``.

    A note in parentheses, and a full stop, may follow the question mark.
    """
    return bool(QUESTION.search(definition))


def without_source(definition):
    """A definition without the source that CDISC notes after it, as ``(NCI)``."""
    found = NOTE.search(definition)
    if found and len(found.group(2).split()) <= SOURCE_WORDS:
        return definition[: found.start()] + found.group(1)
    return definition


def merged(code, rows, canonical, lists):
    """One item from all the rows of one concept code.

    Args:
        code (str): The concept code.
        rows (pd.DataFrame): Its rows, in the file's order.
        canonical (str): The column that holds the canonical.
        lists (list[tuple[str, list[str]]]): Its value lists.

    Returns:
        Item: The item.
    """
    texts = [Text(value, SUBMISSION_VALUE) for value in rows[SUBMISSION_VALUE]]
    for cell in rows[SYNONYMS]:
        texts += [
            Text(name.strip(), SYNONYMS)
            for name in clean(cell).split(";")
            if name.strip()
        ]
    definitions = list(dict.fromkeys(rows[DEFINITION]))
    texts += [
        Text(text, DEFINITION, question=True)
        for text in definitions
        if is_question(text)
    ]
    described = [
        without_source(text) for text in definitions if text and not is_question(text)
    ]
    return Item(
        id=code,
        canonical=next((name for name in rows[canonical] if name), ""),
        description=next(iter(described), ""),
        descriptions=described[1:],
        lists=lists,
        texts=texts,
        open=bool(rows.open.all()),
    )


def items(tables):
    """CDISC's codelists and terms as items.

    Args:
        tables (dict[str, pd.DataFrame]): The tables ``sources fetch`` wrote.

    Returns:
        list[Item]: One item per codelist, and one per term.
    """
    terms = tables["terms"]
    is_codelist = terms["Codelist Code"] == ""
    members = terms[~is_codelist]
    values = {
        codelist: list(
            dict.fromkeys(name for name in group["NCI Preferred Term"] if name)
        )
        for codelist, group in members.groupby("Codelist Code", sort=False)
    }
    found = [
        merged(
            code,
            rows,
            "Codelist Name",
            [("", values[code])] if values.get(code) else [],
        )
        for code, rows in terms[is_codelist].groupby("Code", sort=False)
    ]
    found += [
        merged(code, rows, "NCI Preferred Term", [])
        for code, rows in members.groupby("Code", sort=False)
    ]
    return found
