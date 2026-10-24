"""How a property is written as one text, the same way in every step.

A model reads a target property, or a source property or variable, as one
text. Scoring writes the benchmarks' texts this way, and ``pairs prepare``
writes the training pairs this way. So a model is trained and scored on texts
of one shape.
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
    """A property in the shape of StdElement-Mapper's NCIt training pairs.

    The model card shows ``name: ... | description: ... | values: a; b``, with
    no node and no type. This format scores the model in the shape of its
    training pairs.

    It is not in ai_harmonization.formatters: the approaches read a target's
    node.property back from its text, and this shape has no node.

    Args:
        node (Node): Not part of this shape.
        prop (Property): The property. The first MAX_VALUES_IN_PROMPT of its
            values are kept.

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

    It is the shape of ``name_type_description_values``, with one change: the
    description and its ``": "`` are written only when the property has a
    description. The formatters treat values the same way. A property with no
    description and no values ends at its type, as ``a.doi (string)``. A
    property with values only reads ``a.sex (string) Values: F | M``.

    It is not in ai_harmonization.formatters: the approaches read a target's
    node.property back from its text, up to the first colon, and a text in
    this shape may have no colon.

    Args:
        node (Node): The node.
        prop (Property): The property. The first MAX_VALUES_IN_PROMPT of its
            values are kept.

    Returns:
        str: The text.
    """
    description = f": {prop.description}" if prop.description else ""
    return f"{node.name}.{prop.name} ({prop.type}){description}{values_segment(prop)}"


# Each format is a formatter: (Node, Property) -> text. Four are the formatters
# of ai_harmonization.formatters, under their own names, and two are defined
# here. The default, compact, is the shape that this pipeline's models are
# trained on. Models are also compared in it.
FORMATS = {
    "compact": compact,
    "name_description": get_node_property_as_name_description,
    "name_type_description": get_node_property_as_name_type_description,
    "name_type_description_values": get_node_property_as_name_type_description_values,
    "name_type_values": get_node_property_as_name_type_values,
    "stdelement": stdelement,
}
DEFAULT_FORMAT = "compact"


def property_text(formatter, node, name, type_, description, values):
    """The text of a target property or a source property/variable, by a formatter.

    A source variable is written without a node: its node would be its dbGaP
    table or GDC paper, which says nothing about what it means. Its text then
    starts at its name, without the formatters' leading dot.

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
