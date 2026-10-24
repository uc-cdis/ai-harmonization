"""Tests for embedding_model.texts: how a property is written as one text."""

from ai_harmonization.formatters import (
    MAX_VALUES_IN_PROMPT,
    get_node_property_as_name_type_description_values,
)
from ai_harmonization.simple_data_model import Node, Property
from embedding_model import texts


class TestFormats:
    """Each format writes a property the same way as its formatter."""

    def test_a_target_is_written_by_the_formatter(self):
        """A target property's text is what the formatter writes when called directly."""
        expected = get_node_property_as_name_type_description_values(
            Node(name="a", description="", links=[], properties=[]),
            Property(name="age", description="Age", type="integer", values=["1"]),
        )
        formatter = texts.FORMATS["name_type_description_values"]
        written = texts.property_text(formatter, "a", "age", "integer", "Age", ["1"])
        assert written == expected

    def test_a_source_has_no_leading_dot(self):
        """A source variable has no node, so its text starts with its name."""
        formatter = texts.FORMATS[texts.DEFAULT_FORMAT]
        written = texts.property_text(
            formatter, "", "AGE01", "integer", "Age at exam", ["1"]
        )
        assert written.startswith("AGE01")

    def test_stdelement_shape(self):
        """StdElement text joins its parts with pipes, and its first 12 values with ;.

        An empty description is left out.
        """
        values = [str(v) for v in range(MAX_VALUES_IN_PROMPT + 3)]
        written = texts.property_text(
            texts.stdelement, "a", "age", "integer", "Age", values
        )
        assert written.startswith("name: age | description: Age | values: 0; 1; ")
        assert written.count(";") == MAX_VALUES_IN_PROMPT - 1
        assert (
            texts.property_text(texts.stdelement, "a", "age", "integer", "", None)
            == "name: age"
        )

    def test_a_source_keeps_its_own_leading_dot(self):
        """Only the formatter's dot is dropped, not a dot the name starts with."""
        formatter = texts.FORMATS[texts.DEFAULT_FORMAT]
        written = texts.property_text(formatter, "", ".AGE", "integer", "Age", None)
        assert written.startswith(".AGE ")


class TestCompact:
    """The default format writes each part only when the property has it."""

    def test_neither_description_nor_values_ends_at_the_type(self):
        """A property with no description and no values leaves no separator behind."""
        assert texts.property_text(texts.compact, "a", "doi", "string", "", None) == (
            "a.doi (string)"
        )

    def test_values_without_a_description_follow_the_type(self):
        """Values follow the type when there is no description."""
        assert texts.property_text(texts.compact, "", "sex", "string", "", ["F"]) == (
            "sex (string) Values: F"
        )

    def test_only_the_first_12_values_are_written(self):
        """The values are cut to the first 12, as the formatters cut them."""
        values = [f"v{number}" for number in range(MAX_VALUES_IN_PROMPT + 3)]
        written = texts.property_text(texts.compact, "", "abcd", "string", "", values)
        assert written == "abcd (string) Values: " + " | ".join(
            values[:MAX_VALUES_IN_PROMPT]
        )

    def test_with_a_description_it_is_the_formatters_shape(self):
        """With a description, the text is what name_type_description_values writes."""
        for values in (["F", "M"], None):
            assert texts.property_text(
                texts.compact, "a", "sex", "string", "Abcd: efgh", values
            ) == texts.property_text(
                texts.FORMATS["name_type_description_values"],
                "a",
                "sex",
                "string",
                "Abcd: efgh",
                values,
            )

    def test_the_other_formats_are_the_formatters_own(self):
        """Every other format keeps what its formatter writes, separator included."""
        written = texts.property_text(
            texts.FORMATS["name_type_description_values"],
            "a",
            "doi",
            "string",
            "",
            None,
        )
        assert written == "a.doi (string): "
