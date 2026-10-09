"""Tests for ai_harmonization.formatters — document formatter functions."""

import pytest

from ai_harmonization.simple_data_model import (
    Node,
    Property,
    get_node_prop_type_desc_from_string,
)
from ai_harmonization.formatters import (
    MAX_VALUES_IN_PROMPT,
    VALUE_SEPARATOR,
    format_values,
    get_node_property_as_name_description,
    get_node_property_as_name_type_description,
    get_node_property_as_name_type_description_values,
    get_node_property_as_name_type_values,
)


@pytest.fixture
def node():
    """A "subject" node with a description and no links or properties."""
    return Node(name="subject", description="A study subject", links=[], properties=[])


@pytest.fixture
def prop_plain():
    """An integer property with a description and no enum values."""
    return Property(
        name="age", description="Age at enrollment in years", type="integer"
    )


@pytest.fixture
def prop_enum():
    """An encoded string property with a description and three enum values."""
    return Property(
        name="sex",
        description="Biological sex of the participant",
        type="string/encoded",
        values=["Male", "Female", "Unknown"],
    )


class TestVariantA:
    def test_contains_name_and_description(self, node, prop_plain):
        """Variant A is exactly the slot identifier, a colon and the description."""
        result = get_node_property_as_name_description(node, prop_plain)
        assert result == "subject.age: Age at enrollment in years"

    def test_no_type_in_output(self, node, prop_plain):
        """Variant A leaves out the type, so no "(type)" segment appears."""
        result = get_node_property_as_name_description(node, prop_plain)
        assert "(" not in result


class TestVariantB:
    def test_contains_type(self, node, prop_plain):
        """Variant B adds the type in parentheses after the slot identifier."""
        result = get_node_property_as_name_type_description(node, prop_plain)
        assert "subject.age (integer):" in result

    def test_contains_description(self, node, prop_plain):
        """Variant B keeps the description alongside the type."""
        result = get_node_property_as_name_type_description(node, prop_plain)
        assert "Age at enrollment in years" in result


class TestVariantC:
    def test_includes_enum_values(self, node, prop_enum):
        """Variant C lists the enum values after "Values:", joined by the separator."""
        result = get_node_property_as_name_type_description_values(node, prop_enum)
        expected = VALUE_SEPARATOR.join(["Male", "Female", "Unknown"])
        assert f"Values: {expected}" in result

    def test_includes_description(self, node, prop_enum):
        """Variant C keeps the description alongside the values."""
        result = get_node_property_as_name_type_description_values(node, prop_enum)
        assert "Biological sex" in result

    def test_no_values_clause_when_none(self, node, prop_plain):
        """A property without enum values gets no empty "Values:" clause from C."""
        result = get_node_property_as_name_type_description_values(node, prop_plain)
        assert "Values:" not in result


class TestVariantD:
    def test_omits_description(self, node, prop_enum):
        """Variant D leaves the description out even when the property has values."""
        result = get_node_property_as_name_type_values(node, prop_enum)
        assert "Biological sex" not in result

    def test_includes_enum_values(self, node, prop_enum):
        """Variant D is exactly identifier, type, colon and the "Values:" clause."""
        result = get_node_property_as_name_type_values(node, prop_enum)
        expected = VALUE_SEPARATOR.join(["Male", "Female", "Unknown"])
        assert result == f"subject.sex (string/encoded): Values: {expected}"

    def test_colon_always_present(self, node, prop_plain):
        """Variant D keeps the colon after the type when the property has no values."""
        result = get_node_property_as_name_type_values(node, prop_plain)
        assert "subject.age (integer):" in result

    def test_no_values_reduces_to_identifier_and_type(self, node, prop_plain):
        """Most bdchm slots have no enum, so D carries only name and range."""
        result = get_node_property_as_name_type_values(node, prop_plain)
        assert result == "subject.age (integer):"

    @pytest.mark.parametrize("prop_fixture", ["prop_enum", "prop_plain"])
    def test_output_round_trips_through_the_slot_key_parser(
        self, node, prop_fixture, request
    ):
        """The trailing colon is load-bearing. Without it the parser latches onto
        the colon in "Values:" and mangles the property name, silently corrupting
        the target slot in the mapping CSV."""
        prop = request.getfixturevalue(prop_fixture)
        formatted = get_node_property_as_name_type_values(node, prop)
        parsed_node, parsed_prop, parsed_type, _ = get_node_prop_type_desc_from_string(
            formatted
        )
        assert (parsed_node, parsed_prop) == (node.name, prop.name)
        assert parsed_type == prop.type

    def test_description_is_dropped_whatever_it_contains(self, node):
        """Variant D carries no description, so nothing in one can leak through."""
        prop = Property(
            name="bmi", description="Body mass index measured in kg/m2", type="float"
        )
        assert (
            get_node_property_as_name_type_values(node, prop) == "subject.bmi (float):"
        )


class TestValueCap:
    """The cap keeps target documents inside the embedding model's input
    length, so it has to apply to the source and target sides alike."""

    @pytest.fixture
    def prop_many_values(self):
        """An encoded property with three times MAX_VALUES_IN_PROMPT values."""
        return Property(
            name="observation_type",
            description="What was observed",
            type="string/encoded",
            values=[f"code {i}" for i in range(MAX_VALUES_IN_PROMPT * 3)],
        )

    def test_joins_at_most_the_limit(self, prop_many_values):
        """format_values keeps the first MAX_VALUES_IN_PROMPT values, in order."""
        joined = format_values(prop_many_values.values)
        assert (
            joined.split(VALUE_SEPARATOR)
            == prop_many_values.values[:MAX_VALUES_IN_PROMPT]
        )

    def test_limit_is_a_parameter(self, prop_many_values):
        """An explicit limit replaces the default cap and keeps that many values."""
        joined = format_values(prop_many_values.values, limit=3)
        assert joined == VALUE_SEPARATOR.join(["code 0", "code 1", "code 2"])

    def test_both_value_bearing_variants_cap_alike(self, node, prop_many_values):
        """Variants C and D format the target documents and the source queries,
        so one cap here is what keeps the two sides comparable."""
        for formatter in (
            get_node_property_as_name_type_description_values,
            get_node_property_as_name_type_values,
        ):
            result = formatter(node, prop_many_values)
            shown = result.split("Values: ")[1]
            assert (
                shown.split(VALUE_SEPARATOR)
                == prop_many_values.values[:MAX_VALUES_IN_PROMPT]
            )
