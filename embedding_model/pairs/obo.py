"""The ontologies: each term, around its label.

The source side holds the term's EXACT synonyms, of every synonym type.
RELATED, BROAD and NARROW synonyms do not mean the same thing, so they are
not used. The description is the term's first definition, and any other
definition is paired with it. An obsolete term with a "term replaced by" gives
its label and synonyms to the replacement. The word "obsolete" that marks an
obsolete term's texts is removed.
"""

import re
from collections import defaultdict

from embedding_model.pairs.common import CURRENT, RETIRED, Item, Text
from embedding_model.sources.common import report_new

SAME_MEANING = {"EXACT"}
# How an obsolete term's texts are marked: "obsolete abcd", "OBSOLETE. An
# abcd." or "Thin abcd (obsolete)".
OBSOLETE = re.compile(r"^\s*obsoleted?\b[.:]?\s*|\s*\(obsolete\)\s*$", re.IGNORECASE)
NOT_SAME_MEANING = {"RELATED", "BROAD", "NARROW"}


def unmarked(text, obsolete):
    """An obsolete term's text without the word that marks it obsolete."""
    return OBSOLETE.sub("", text) if obsolete else text


def items(tables):
    """The ontologies' terms as items.

    Args:
        tables (dict[str, pd.DataFrame]): The tables ``sources fetch`` wrote.

    Returns:
        list[Item]: One item per term.
    """
    synonyms = tables["synonyms"]
    report_new(synonyms.scope, SAME_MEANING | NOT_SAME_MEANING, "scope", "not used")
    texts = defaultdict(list)
    for row in synonyms[synonyms.scope.isin(SAME_MEANING)].itertuples():
        texts[row.id].append(Text(row.name, row.synonym_types or row.scope))
    definitions = tables["definitions"]
    described = definitions.groupby("id").definition.agg(list).to_dict()
    return [
        Item(
            id=row.id,
            canonical=unmarked(row.label, row.obsolete),
            description=unmarked(described.get(row.id, [""])[0], row.obsolete),
            descriptions=[
                unmarked(text, row.obsolete) for text in described.get(row.id, [])[1:]
            ],
            texts=[
                Text(unmarked(text.text, row.obsolete), text.term_type)
                for text in texts[row.id]
            ],
            status=RETIRED if row.obsolete else CURRENT,
            replaced_by=[found for found in row.replaced_by.split("|") if found],
            open=row.open,
        )
        for row in tables["terms"].itertuples()
    ]
