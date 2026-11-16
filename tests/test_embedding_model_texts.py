"""Tests for embedding_model.texts: how a property is written as one text."""

from ai_harmonization.formatters import MAX_VALUES_IN_PROMPT
from embedding_model import texts


class TestFormats:
    """How a format writes a property as one text."""

    def test_a_target_text_is_node_property_type_description_and_values(self):
        """A target property's text is node.property (type): its description, then its values."""
        formatter = texts.FORMATS["name_type_description_values"]
        written = texts.property_text(
            formatter, "demographic", "sex", "string", "Sex", ["F", "M"]
        )
        assert written == "demographic.sex (string): Sex Values: F | M"

    def test_a_source_has_no_leading_dot(self):
        """A source variable has no node, so property_text starts its text with its name."""
        formatter = texts.FORMATS[texts.DEFAULT_FORMAT]
        written = texts.property_text(
            formatter, "", "AGE01", "integer", "Age at exam", None
        )
        assert written.startswith("AGE01")

    def test_stdelement_shape(self):
        """stdelement joins a text's parts with pipes, and its first 12 values with ``;``.

        stdelement drops an empty description.
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
        """property_text drops only the formatter's dot, not a dot that the name starts with."""
        formatter = texts.FORMATS[texts.DEFAULT_FORMAT]
        written = texts.property_text(formatter, "", ".AGE", "integer", "Age", None)
        assert written.startswith(".AGE ")


class TestCompact:
    """The default format writes the description and the values only when the property has them."""

    def test_neither_description_nor_values_ends_at_the_type(self):
        """A property with no description and no values ends at its type, with no separator."""
        assert texts.property_text(texts.compact, "a", "doi", "string", "", None) == (
            "a.doi (string)"
        )

    def test_values_without_a_description_follow_the_type(self):
        """Values follow the type when there is no description."""
        assert texts.property_text(
            texts.compact, "", "sex", "string", "", ["F", "M"]
        ) == ("sex (string) Values: F | M")

    def test_only_the_first_12_values_are_written(self):
        """compact keeps the first 12 values, as the formatters do."""
        values = [f"v{number}" for number in range(MAX_VALUES_IN_PROMPT + 3)]
        written = texts.property_text(
            texts.compact, "", "abcd_property", "string", "", values
        )
        assert written == "abcd_property (string) Values: " + " | ".join(
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
        """property_text keeps the name_type_description_values formatter's text, ": " included."""
        written = texts.property_text(
            texts.FORMATS["name_type_description_values"],
            "a",
            "doi",
            "string",
            "",
            None,
        )
        assert written == "a.doi (string): "
