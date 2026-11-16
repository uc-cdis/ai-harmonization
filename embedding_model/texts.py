"""How a property is written as one text, the same way in every step.

A model reads a target property, or a source property or variable, as one
text. ``score`` writes the benchmarks' texts this way, and ``pairs prepare``
writes the training pairs this way. So, by default, a model trains and is
scored on texts in one format.
"""

from ai_harmonization.formatters import (
    format_values,
    get_node_property_as_name_description,
    get_node_property_as_name_type_description,
    get_node_property_as_name_type_description_values,
    get_node_property_as_name_type_values,
    values_segment,
)
from ai_harmonization.simple_data_model import Node, Property


def stdelement(node, prop):
    """A property in the format of StdElement-Mapper's NCIt training pairs.

    Its model card shows ``name: ... | description: ... | values: a; b``, with
    no node and no type. The model is private, so its card needs access too.
    With this format, ``score`` scores the model in the format of its training
    pairs.

    The approaches read a target's node.property from its text. This format
    has no node, so it is not in ai_harmonization.formatters.

    Args:
        node (Node): Not part of this format.
        prop (Property): The property. The text keeps the first
            MAX_VALUES_IN_PROMPT of its values.

    Returns:
        str: The text.
    """
    parts = [f"name: {prop.name}"]
    if prop.description:
        parts.append(f"description: {prop.description}")
    values = format_values(prop.values, separator="; ")
    if values:
        parts.append(f"values: {values}")
    return " | ".join(parts)


def compact(node, prop):
    """A property as ``node.name (type): description Values: a | b``, each if present.

    It is the format of ``name_type_description_values``, with one change:
    compact writes the description and its ``": "`` only when the property has
    a description. The formatters treat values the same way. A property with no
    description and no values ends at its type, for example ``a.doi (string)``.
    A property with values only reads ``a.sex (string) Values: F | M``.

    The approaches read a target's node.property from its text, up to the
    first colon. A text in this format may have no colon, so compact is not in
    ai_harmonization.formatters.

    Args:
        node (Node): The node.
        prop (Property): The property. The text keeps the first
            MAX_VALUES_IN_PROMPT of its values.

    Returns:
        str: The text.
    """
    description = f": {prop.description}" if prop.description else ""
    return f"{node.name}.{prop.name} ({prop.type}){description}{values_segment(prop)}"


# Each format is a formatter: (Node, Property) -> text. Four are the formatters
# of ai_harmonization.formatters, under their own names, and this module
# defines the other two. The default, compact, is the format that this
# pipeline trains its models in. compare also compares models in compact by
# default.
FORMATS = {
    "compact": compact,
    "name_description": get_node_property_as_name_description,
    "name_type_description": get_node_property_as_name_type_description,
    "name_type_description_values": get_node_property_as_name_type_description_values,
    "name_type_values": get_node_property_as_name_type_values,
    "stdelement": stdelement,
}
DEFAULT_FORMAT = "compact"


def recorded_format(manifest, manifest_path, next_step):
    """The text format that a manifest holds for its texts, one of FORMATS.

    The pairs, a corpus and a run each record the format in ``settings``.

    Args:
        manifest (dict): The manifest.
        manifest_path (Path): Where the manifest is, for the messages.
        next_step (str): The messages' last part: what to do when the format
            is missing or unknown, for example ``run: python -m embedding_model
            pairs prepare --source ncit``.

    Returns:
        str: The format's name.

    Raises:
        ValueError: If the manifest holds no format, or one that is not in
            FORMATS.
    """
    settings = manifest.get("settings")
    text_format = settings.get("format") if isinstance(settings, dict) else None
    if text_format is None:
        raise ValueError(
            f"{manifest_path} holds no text format for its texts. {next_step}"
        )
    if not isinstance(text_format, str) or text_format not in FORMATS:
        raise ValueError(
            f"{manifest_path} holds the text format {text_format!r}, which is"
            f" not one of {', '.join(FORMATS)}. {next_step}"
        )
    return text_format


def property_text(formatter, node, name, type_, description, values):
    """The text of a target property or a source property/variable, by a formatter.

    A source variable's node would be its dbGaP table, or the paper that it
    comes from. That node says nothing about what the variable means, so the
    variable's text has no node. So the text starts at the variable's name,
    without the formatters' leading dot.

    Args:
        formatter (Callable[[Node, Property], str]): A value of FORMATS.
        node (str): The node, or "" for a source variable.
        name (str): The target property's or source property/variable's name.
        type_ (str): Its type.
        description (str): Its description.
        values (list[str] | None): Its values.

    Returns:
        str: The text.
    """
    written = formatter(
        Node(name=node, description="", links=[], properties=[]),
        Property(name=name, description=description, type=type_, values=values),
    )
    return written if node else written.removeprefix(".")
