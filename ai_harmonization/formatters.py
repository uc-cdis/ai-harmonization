"""
Document formatters for multi-prompt similarity search.

Each formatter converts a (Node, Property) pair into a string used for both
target index documents and source variable queries. Using the same formatter
on both sides ensures the embedding space is symmetric.

The label of the variant that produced a match is recorded in the
``prompt_variant`` output column. Pass these to
``MultiPromptSimilaritySearch(prompt_variants={'A': ..., 'B': ...})``.

Prompt variants:
  A — name + description                          get_node_property_as_name_description
  B — name + type + description                   get_node_property_as_name_type_description
  C — name + type + description + enum values     get_node_property_as_name_type_description_values
  D — name + type + enum values (no description)  get_node_property_as_name_type_values

Variants C and D cap how many values they include; see MAX_VALUES_IN_PROMPT.
"""

from ai_harmonization.simple_data_model import get_node_property_as_string

# What joins a property's values wherever a list of them is rendered as one
# string: the embedded prompt text, the mapping CSVs, and the review widget.
#
# A pipe, not a comma, for two reasons. Some values contain ", " themselves,
# which makes a comma-joined list impossible to split back into values — and
# the review widget has only the joined string to work from. And experiments
# confirmed it matches better.
VALUE_SEPARATOR = " | "

# How many of a property's permissible values go into a prompt.
#
# An uncapped list crowds out the property's name, type and description, and is
# not kept in full anyway: text past the embedding model's input length is
# discarded wherever the tokenizer reaches it, mid-value. Capping puts that
# boundary at a whole value, and at the same place on both sides of the
# comparison, since one formatter builds both the target documents and the
# source queries. 12 covers all but a handful of real value lists.
#
# Only prompt text is affected; the model and the curator's CSV keep every
# value.
MAX_VALUES_IN_PROMPT = 12


def format_values(values, limit=MAX_VALUES_IN_PROMPT):
    """Join a property's permissible values for use in a prompt.

    Args:
        values (list | None): Permissible values, or None.
        limit (int): Maximum number to include.

    Returns:
        str: The values joined by VALUE_SEPARATOR, or "" when there are none.
    """
    if not values:
        return ""
    return VALUE_SEPARATOR.join(str(value) for value in values[:limit])


def values_segment(prop):
    """Return the trailing ``" Values: ..."`` segment for a property.

    The variants that include values all append them the same way — as an
    optional tail with its own leading space — so the segment is built once
    here rather than each variant deciding for itself.

    Args:
        prop (Property): The property whose values to render.

    Returns:
        str: ``" Values: a | b | c"``, or ``""`` when the property has none.
    """
    values = format_values(prop.values)
    return f" Values: {values}" if values else ""


def get_node_property_as_name_description(node, prop):
    """Variant A: slot identifier + description (no type)."""
    return f"{node.name}.{prop.name}: {prop.description}"


# Variant B: slot identifier + type + description.
# Identical to ai_harmonization.simple_data_model.get_node_property_as_string;
# aliased here so all four prompt-variant formatters live in one place.
get_node_property_as_name_type_description = get_node_property_as_string


def get_node_property_as_name_type_description_values(node, prop):
    """Variant C: slot identifier + type + description + enum values.

    Values are capped at MAX_VALUES_IN_PROMPT.
    """
    return (
        f"{node.name}.{prop.name} ({prop.type}): "
        f"{prop.description}{values_segment(prop)}"
    )


def get_node_property_as_name_type_values(node, prop):
    """Variant D: slot identifier + type + enum values — no description.

    Omitting the description matches on field name, type and value categories
    only. This helps where the two descriptions disagree or where one side is
    boilerplate: a dbGaP variable described as "Severity of emphysema based
    upon the degree of parenchymal involvement" and a target slot described as
    "A subjective assessment of the severity of the condition" share almost no
    wording, while their value ladders line up closely.

    The trailing colon is load-bearing: ``get_node_prop_type_desc_from_string``
    splits the slot identifier off at the first colon, so without it the parser
    either latches onto the colon in "Values:" and returns a property name of
    "sex (SexEnum) Values", or fails outright and falls back to using the whole
    document as the slot key. Both corrupt the target name silently. The
    ``(type)`` segment, by contrast, is optional as far as that parser is
    concerned — variant A omits it.
    """
    return f"{node.name}.{prop.name} ({prop.type}):{values_segment(prop)}"
