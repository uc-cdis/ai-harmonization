"""LOINC: each code, around its long common name.

The source side holds the code's short name and display name, and its
question texts: the survey question, and its wording on each open form row.
RELATEDNAMES2 and the consumer names are not used, since LOINC says they are
not synonyms. The description is the code's definition.

The values are the code's default answer list, in the list's own order. A
code's default list has no panel context, whatever its link type. The codes
written before the answers are removed as ``common.value_labels`` says.

A deprecated or discouraged code that ``MapTo.csv`` maps to another code
gives its names to that code. The word "Deprecated" that LOINC puts before a
deprecated code's names is removed.
"""

from collections import defaultdict

import pandas as pd

from embedding_model.pairs.common import (
    CURRENT,
    DRAFT,
    RETIRED,
    Item,
    Text,
    undeprecated,
    value_labels,
)
from embedding_model.sources.common import report_new

STATUSES = {
    "ACTIVE": CURRENT,
    "TRIAL": DRAFT,
    "DEPRECATED": RETIRED,
    "DISCOURAGED": RETIRED,
}


def unmarked(name, status):
    """A deprecated code's name without the "Deprecated" that LOINC puts first."""
    return undeprecated(name) if status == "DEPRECATED" else name


def answer_lists(answers, links):
    """Each code's default answer list, named by its id and link type.

    Args:
        answers (pd.DataFrame): ``answers``, one row per answer.
        links (pd.DataFrame): ``answer_links``, one row per code and list.

    Returns:
        dict[str, list[tuple[str, list[str]]]]: Each code's default list, for
            the codes whose default list has answers.
    """
    order = pd.to_numeric(answers.SequenceNumber, errors="coerce")
    ordered = answers.assign(order=order).sort_values(
        ["AnswerListId", "order"], kind="stable"
    )
    grouped = ordered.groupby("AnswerListId").DisplayText.agg(list)
    texts = {
        list_id: value_labels(list_answers) for list_id, list_answers in grouped.items()
    }
    lists = {}
    for row in links[links.ApplicableContext == ""].itertuples():
        if texts.get(row.AnswerListId):
            name = f"{row.AnswerListId} ({row.AnswerListLinkType})"
            lists[row.LoincNumber] = [(name, texts[row.AnswerListId])]
    return lists


def items(tables):
    """LOINC's codes as items.

    Args:
        tables (dict[str, pd.DataFrame]): The tables ``sources fetch`` wrote.

    Returns:
        list[Item]: One item per code.
    """
    codes = tables["codes"]
    report_new(codes.STATUS, set(STATUSES), "STATUS", "read as current")
    forms = tables["forms"]
    worded = defaultdict(list)
    for row in forms[forms.open & (forms.DisplayNameForForm != "")].itertuples():
        worded[row.Loinc].append(
            Text(row.DisplayNameForForm, "DisplayNameForForm", question=True)
        )
    lists = answer_lists(tables["answers"], tables["answer_links"])
    replaced_by = tables["map_to"].groupby("LOINC").MAP_TO.agg(list).to_dict()
    return [
        Item(
            id=row.LOINC_NUM,
            canonical=unmarked(row.LONG_COMMON_NAME, row.STATUS),
            description=row.DefinitionDescription,
            lists=lists.get(row.LOINC_NUM, []),
            texts=[
                Text(unmarked(row.SHORTNAME, row.STATUS), "SHORTNAME"),
                Text(row.DisplayName, "DisplayName"),
                Text(row.SURVEY_QUEST_TEXT, "SURVEY_QUEST_TEXT", question=True),
                *worded[row.LOINC_NUM],
            ],
            status=STATUSES.get(row.STATUS, CURRENT),
            replaced_by=replaced_by.get(row.LOINC_NUM, []),
            open=row.open,
        )
        for row in codes.itertuples()
    ]
