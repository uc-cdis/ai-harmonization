"""What the source readers share: a pin from the manifest, and a restriction field."""


def pinned(manifest, name):
    """One of a source's inputs, as the committed manifest pins it.

    Args:
        manifest (dict): The source's committed manifest.
        name (str): The input's name in the manifest's ``sources``.

    Returns:
        dict: The manifest's entry for the input: its version, SHA-256 or
            both.

    Raises:
        KeyError: If the manifest has no input of that name.
    """
    for source in manifest.get("sources", []):
        if source.get("name") == name:
            return source
    raise KeyError(f"the manifest has no source named {name!r}")


def report_new(values, known, field, outcome):
    """Print each value of a field that is not among the known ones, with its count.

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
        closed (set[str]): The values known not to be.
        field (str): The field's name, for the report.

    Returns:
        pd.Series: True where the item may be used.
    """
    report_new(values, allowed | closed, field, "not used")
    return values.isin(allowed)
