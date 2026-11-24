"""What the benchmark builders share.

``bookkeeping_fields`` gives a node's fields that are not targets, and
``target_row`` writes a dictionary property as a row of a benchmark's targets,
the same way for the GDC and the BDC benchmarks.
"""

from ai_harmonization.formatters import VALUE_SEPARATOR


def bookkeeping_fields(node_schema):
    """A node's bookkeeping fields, which are not targets: its ``systemProperties`` and its ``type``.

    The data commons fills a node's ``systemProperties``, for example ``id``,
    ``created_datetime`` and ``state``. A node's ``type`` holds the node's own
    name.

    Args:
        node_schema (dict): The node's schema in the dictionary.

    Returns:
        set[str]: The names of the node's bookkeeping fields.
    """
    return {"type", *(node_schema.get("systemProperties") or [])}


def target_row(node_name, prop):
    """A dictionary property's row in a benchmark's targets, keyed by ``benchmarks.TARGET_COLUMNS``.

    A type that lists several kinds, for example ``["string", "null"]``, is
    written as its kinds other than ``null``, joined by ``/``.

    Args:
        node_name (str): The property's node.
        prop (Property): The property, as ``SimpleDataModel`` reads it.

    Returns:
        dict: The target's property key, type, description and values.
    """
    target_type = prop.type
    if isinstance(target_type, list):
        target_type = "/".join(str(kind) for kind in target_type if kind != "null")
    return {
        "target_property": f"{node_name}.{prop.name}",
        "target_type": str(target_type),
        "target_description": prop.description,
        "target_values": VALUE_SEPARATOR.join(
            str(value) for value in prop.values or []
        ),
    }
