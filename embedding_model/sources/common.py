"""What the source readers share: a field's new values, and whether an item is open.

``report_new`` prints the values of a field that a reader does not know yet.
``open_where`` says which items a restriction field allows to be used.
"""


def report_new(values, known, field, outcome):
    """Print each value of a field that is not among the known values, with its count.

    Args:
        values (pd.Series): The field, one value per item.
        known (set[str]): The values already decided on.
        field (str): The field's name, for the report.
        outcome (str): What a new value means for its items, such as
            ``not used``.
    """
    for value, count in values[~values.isin(known)].value_counts().items():
        print(f"  new value of {field}: {value!r} on {count:,} items; {outcome}")


def open_where(values, allowed, closed, field):
    """Whether each item's restriction field holds a value known to allow use.

    An item is used only when its field holds one of the ``allowed`` values.
    Any other value means it is not used. A value in neither ``allowed`` nor
    ``closed`` is new, so it is printed with its count, for someone to
    decide on.

    Args:
        values (pd.Series): The field, one value per item.
        allowed (set[str]): The values known to be open.
        closed (set[str]): The values known to be closed.
        field (str): The field's name, for the report.

    Returns:
        pd.Series: True where the item may be used.
    """
    report_new(values, allowed | closed, field, "not used")
    return values.isin(allowed)
