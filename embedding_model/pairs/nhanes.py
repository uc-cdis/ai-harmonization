"""NHANES: each variable of each data file, around its codebook text.

A variable in one data file is one item, since each data file has its own
codebook. The canonical is the codebook's English Text: the question asked,
without the form's number for it, or what the variable measures. When a
codebook has none, the SAS label is used, or else the variable list's
description. The source side holds the variable's name, and its SAS label and
description where they differ.

The values are the codebook's value labels, in its order. Three labels
describe the codebook's rows rather than a value, so they are left out:
``Range of Values``, ``Value was recorded`` and ``Missing``.
"""

from embedding_model.pairs.common import Item, Text, clean, without_item_number

KEY = ["cycle", "data_file", "name"]
NOT_VALUES = {"Range of Values", "Value was recorded", "Missing"}


def items(tables):
    """NHANES' variables as items, one per variable and data file.

    Args:
        tables (dict[str, pd.DataFrame]): The tables ``sources fetch`` wrote.

    Returns:
        list[Item]: One item per variable and data file.
    """
    variables = tables["variables"].drop_duplicates(KEY)
    both = variables.merge(
        tables["codebook"], on=KEY, how="outer", suffixes=("_list", "_codebook")
    )
    both = both.fillna({"open_list": True, "open_codebook": True}).fillna("")
    values = tables["values"].sort_values([*KEY, "position"], kind="stable")
    values = values.assign(label=values.label.map(clean))
    values = values[~values.label.isin(NOT_VALUES)]
    listed = values.groupby(KEY).label.agg(list).to_dict()
    found = []
    for row in both.itertuples():
        key = (row.cycle, row.data_file, row.name)
        found.append(
            Item(
                id="/".join(key),
                canonical=without_item_number(clean(row.english_text))
                or row.sas_label
                or row.description,
                lists=[("", listed[key])] if key in listed else [],
                texts=[
                    Text(row.name, "variable name"),
                    Text(row.sas_label, "SAS label"),
                    Text(row.description, "variable list description"),
                ],
                open=bool(row.open_list and row.open_codebook),
            )
        )
    return found
