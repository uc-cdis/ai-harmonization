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
"""

from ai_harmonization.simple_data_model import get_node_property_as_string

_SEP = ", "


def get_node_property_as_name_description(node, prop):
    """Variant A: slot identifier + description (no type)."""
    return f"{node.name}.{prop.name}: {prop.description}"


# Variant B: slot identifier + type + description.
# Identical to ai_harmonization.simple_data_model.get_node_property_as_string;
# aliased here so all four prompt-variant formatters live in one place.
get_node_property_as_name_type_description = get_node_property_as_string


def get_node_property_as_name_type_description_values(node, prop):
    """Variant C: slot identifier + type + description + enum values."""
    enum_ctx = f" Values: {_SEP.join(prop.values)}" if prop.values else ""
    return f"{node.name}.{prop.name} ({prop.type}): {prop.description}{enum_ctx}"


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
    parts = [f"{node.name}.{prop.name} ({prop.type}):"]
    if prop.values:
        parts.append(f"Values: {_SEP.join(prop.values)}")
    return " ".join(parts)
