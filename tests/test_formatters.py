"""Tests for ai_harmonization.formatters — document formatter functions."""

import pytest

from ai_harmonization.simple_data_model import (
    Node,
    Property,
    get_node_prop_type_desc_from_string,
)
from ai_harmonization.formatters import (
    get_node_property_as_name_description,
    get_node_property_as_name_type_description,
    get_node_property_as_name_type_description_values,
    get_node_property_as_name_type_values,
)


@pytest.fixture
def node():
    return Node(name="subject", description="A study subject", links=[], properties=[])


@pytest.fixture
def prop_plain():
    return Property(
        name="age", description="Age at enrollment in years", type="integer"
    )


@pytest.fixture
def prop_enum():
    return Property(
        name="sex",
        description="Biological sex of the participant",
        type="string/encoded",
        values=["Male", "Female", "Unknown"],
    )


class TestVariantA:
    def test_contains_name_and_description(self, node, prop_plain):
        result = get_node_property_as_name_description(node, prop_plain)
        assert result == "subject.age: Age at enrollment in years"

    def test_no_type_in_output(self, node, prop_plain):
        result = get_node_property_as_name_description(node, prop_plain)
        assert "(" not in result


class TestVariantB:
    def test_contains_type(self, node, prop_plain):
        result = get_node_property_as_name_type_description(node, prop_plain)
        assert "subject.age (integer):" in result

    def test_contains_description(self, node, prop_plain):
        result = get_node_property_as_name_type_description(node, prop_plain)
        assert "Age at enrollment in years" in result


class TestVariantC:
    def test_includes_enum_values(self, node, prop_enum):
        result = get_node_property_as_name_type_description_values(node, prop_enum)
        assert "Values: Male, Female, Unknown" in result

    def test_includes_description(self, node, prop_enum):
        result = get_node_property_as_name_type_description_values(node, prop_enum)
        assert "Biological sex" in result

    def test_no_values_clause_when_none(self, node, prop_plain):
        result = get_node_property_as_name_type_description_values(node, prop_plain)
        assert "Values:" not in result


class TestVariantD:
    def test_omits_description(self, node, prop_enum):
        result = get_node_property_as_name_type_values(node, prop_enum)
        assert "Biological sex" not in result

    def test_includes_enum_values(self, node, prop_enum):
        result = get_node_property_as_name_type_values(node, prop_enum)
        assert result == "subject.sex (string/encoded): Values: Male, Female, Unknown"

    def test_colon_always_present(self, node, prop_plain):
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
