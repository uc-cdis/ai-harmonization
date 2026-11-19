"""Tests for SimpleDataModel.from_linkml_yaml() and from_gdc_model() in ai_harmonization.simple_data_model."""

import json
import textwrap

import pytest
from pydantic import ValidationError

from ai_harmonization.simple_data_model import (
    DEFAULT_PROPERTY_DESCRIPTION,
    SimpleDataModel,
)


MINIMAL_SCHEMA = textwrap.dedent(
    """\
    classes:
      Subject:
        description: A study participant
        attributes:
          age:
            description: Age at enrollment
            range: integer
          sex:
            description: Biological sex
            range: SexEnum
      Empty: {}
    enums:
      SexEnum:
        permissible_values:
          Male: {}
          Female: {}
          Unknown: {}
    slots: {}
"""
)

INHERITANCE_SCHEMA = textwrap.dedent(
    """\
    classes:
      Base:
        attributes:
          id:
            description: Identifier
            range: string
      Child:
        is_a: Base
        attributes:
          extra:
            description: Extra field
            range: string
"""
)

MIXIN_SCHEMA = textwrap.dedent(
    """\
    classes:
      Timestamped:
        mixin: true
        attributes:
          created_at:
            description: Creation timestamp
            range: string
      Record:
        mixins:
          - Timestamped
        attributes:
          value:
            description: Record value
            range: string
"""
)

ENUM_INHERITANCE_SCHEMA = textwrap.dedent(
    """\
    classes:
      Sample:
        attributes:
          status:
            description: Status
            range: ExtendedStatus
    enums:
      BaseStatus:
        permissible_values:
          Active: {}
          Inactive: {}
      ExtendedStatus:
        inherits:
          - BaseStatus
        permissible_values:
          Pending: {}
"""
)

# A class whose fields come from the schema-level `slots` section rather than
# from inline `attributes`. bdchm uses both: 225 inline attributes plus 7 shared
# slot definitions referenced 35 times, so both paths have to resolve.
GLOBAL_SLOTS_SCHEMA = textwrap.dedent(
    """\
    classes:
      Observation:
        slots:
          - observation_id
          - value
    slots:
      observation_id:
        description: Primary key
        range: string
      value:
        description: Observed value
        range: float
"""
)

CYCLIC_SCHEMA = textwrap.dedent(
    """\
    classes:
      A:
        is_a: B
        attributes:
          a_field:
            description: From A
            range: string
      B:
        is_a: A
        attributes:
          b_field:
            description: From B
            range: string
"""
)


class TestFromLinkmlYaml:
    def test_parses_non_empty_classes(self):
        """A class with attributes becomes a node."""
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        names = [n.name for n in model.nodes]
        assert "Subject" in names

    def test_empty_class_excluded(self):
        """A class with no attributes yields no node rather than an empty one."""
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        names = [n.name for n in model.nodes]
        assert "Empty" not in names

    def test_node_description_preserved(self):
        """The class description carries over to the node."""
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        subject = next(n for n in model.nodes if n.name == "Subject")
        assert subject.description == "A study participant"

    def test_enum_values_resolved(self):
        """A slot whose range is an enum gets that enum's permissible values."""
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        subject = next(n for n in model.nodes if n.name == "Subject")
        sex = next(p for p in subject.properties if p.name == "sex")
        assert set(sex.values) == {"Male", "Female", "Unknown"}

    def test_non_enum_range_has_no_values(self):
        """A slot whose range is not an enum leaves values as None."""
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        subject = next(n for n in model.nodes if n.name == "Subject")
        age = next(p for p in subject.properties if p.name == "age")
        assert age.values is None

    def test_is_a_inheritance(self):
        """A child class carries its is_a parent's attributes as well as its own."""
        model = SimpleDataModel.from_linkml_yaml(INHERITANCE_SCHEMA)
        child = next(n for n in model.nodes if n.name == "Child")
        prop_names = {p.name for p in child.properties}
        assert "id" in prop_names
        assert "extra" in prop_names

    def test_mixin_inheritance(self):
        """A class carries its mixin's attributes as well as its own."""
        model = SimpleDataModel.from_linkml_yaml(MIXIN_SCHEMA)
        record = next(n for n in model.nodes if n.name == "Record")
        prop_names = {p.name for p in record.properties}
        assert "created_at" in prop_names
        assert "value" in prop_names

    def test_enum_inherits_chain(self):
        """An enum that inherits carries its parent's values as well as its own."""
        model = SimpleDataModel.from_linkml_yaml(ENUM_INHERITANCE_SCHEMA)
        sample = next(n for n in model.nodes if n.name == "Sample")
        status = next(p for p in sample.properties if p.name == "status")
        assert set(status.values) == {"Active", "Inactive", "Pending"}

    def test_global_slots_resolved(self):
        """Slot references take their range and description from the slots section."""
        model = SimpleDataModel.from_linkml_yaml(GLOBAL_SLOTS_SCHEMA)
        observation = next(n for n in model.nodes if n.name == "Observation")
        by_name = {p.name: p for p in observation.properties}
        assert set(by_name) == {"observation_id", "value"}
        assert by_name["value"].type == "float"
        assert by_name["observation_id"].description == "Primary key"

    def test_undescribed_field_gets_default_description(self):
        """A field without a description gets DEFAULT_PROPERTY_DESCRIPTION."""
        model = SimpleDataModel.from_linkml_yaml(
            "classes:\n  Thing:\n    attributes:\n      key: {}\n"
        )
        thing = next(n for n in model.nodes if n.name == "Thing")
        assert thing.properties[0].description == DEFAULT_PROPERTY_DESCRIPTION

    def test_missing_range_defaults_to_string(self):
        """A field without a range is typed "string"."""
        model = SimpleDataModel.from_linkml_yaml(
            "classes:\n  Thing:\n    attributes:\n      key:\n        description: A key\n"
        )
        thing = next(n for n in model.nodes if n.name == "Thing")
        assert thing.properties[0].type == "string"

    def test_inheritance_cycle_does_not_recurse_forever(self):
        """An is_a cycle terminates, with A still picking up B's attributes."""
        model = SimpleDataModel.from_linkml_yaml(CYCLIC_SCHEMA)
        by_name = {n.name: n for n in model.nodes}
        assert {p.name for p in by_name["A"].properties} == {"a_field", "b_field"}

    @pytest.mark.parametrize("empty_yaml", ["", "{}", "classes: {}"])
    def test_empty_schema_returns_empty_model(self, empty_yaml):
        """Empty YAML, an empty map and an empty classes section all give no nodes."""
        model = SimpleDataModel.from_linkml_yaml(empty_yaml)
        assert model.nodes == []


class TestPermissibleValueForms:
    """LinkML accepts a map or a bare list of permissible values; both must load.

    bdchm uses the map form, with a description and a meaning CURIE on every
    value. The list form is equally valid LinkML, so a schema written that way
    has to resolve to the same values rather than failing on the shorthand.
    """

    MAP_FORM = textwrap.dedent(
        """\
        classes:
          Subject:
            attributes:
              status: {range: StatusEnum}
        enums:
          StatusEnum:
            permissible_values:
              ACTIVE: {}
              INACTIVE: {}
        """
    )

    RICH_MAP_FORM = textwrap.dedent(
        """\
        classes:
          Subject:
            attributes:
              status: {range: StatusEnum}
        enums:
          StatusEnum:
            permissible_values:
              ACTIVE:
                description: Currently enrolled
                meaning: OMOP:1234
              INACTIVE:
                description: No longer enrolled
                meaning: OMOP:5678
        """
    )

    LIST_FORM = textwrap.dedent(
        """\
        classes:
          Subject:
            attributes:
              status: {range: StatusEnum}
        enums:
          StatusEnum:
            permissible_values:
              - ACTIVE
              - INACTIVE
        """
    )

    @staticmethod
    def _status_values(schema):
        model = SimpleDataModel.from_linkml_yaml(schema)
        subject = next(n for n in model.nodes if n.name == "Subject")
        return next(p for p in subject.properties if p.name == "status").values

    @pytest.mark.parametrize("form", ["MAP_FORM", "RICH_MAP_FORM", "LIST_FORM"])
    def test_all_forms_yield_the_same_values(self, form):
        """Map, rich map and list forms all give the same values, in schema order."""
        assert self._status_values(getattr(self, form)) == ["ACTIVE", "INACTIVE"]

    def test_empty_permissible_values_leaves_values_unset(self):
        """An enum that lists no permissible values leaves values as None."""
        schema = (
            "classes:\n  S:\n    attributes:\n      f: {range: E}\nenums:\n  E: {}\n"
        )
        model = SimpleDataModel.from_linkml_yaml(schema)
        assert model.nodes[0].properties[0].values is None


class TestFromGdcModel:
    DICTIONARY = {
        "_definitions": {"ignored": {"type": "string"}},
        "demographic": {
            "description": "Demographic data.",
            "properties": {
                "race": {
                    "type": ["string", "null"],
                    "description": "Self-reported race.",
                    "enum": ["asian", "white"],
                },
                "age_at_index": {
                    "description": "Age at index.",
                    "oneOf": [{"type": "integer"}, {"type": "null"}],
                },
                "vital_status": {
                    "description": "Alive or dead.",
                    "oneOf": [{"enum": ["Alive", "Dead"]}, {"type": "null"}],
                },
                "state": {"common": {"description": "The record's state."}},
                "sites": {"type": "array", "items": {"enum": ["Lung", "Liver"]}},
                "cases": {"anyOf": [{"items": {}}]},
            },
        },
    }

    def properties(self):
        """The demographic node's properties, by name."""
        model = SimpleDataModel.from_gdc_model(self.DICTIONARY)
        (node,) = model.nodes
        return {p.name: p for p in node.properties}

    def test_every_property_is_kept_in_order(self):
        """Every property is a Property, links included, in the dictionary's order."""
        assert list(self.properties()) == [
            "race",
            "age_at_index",
            "vital_status",
            "state",
            "sites",
            "cases",
        ]

    def test_fields_the_property_gives_are_used_as_given(self):
        """A property's own type, description and enum are kept unchanged."""
        race = self.properties()["race"]
        assert race.type == ["string", "null"]
        assert race.description == "Self-reported race."
        assert race.values == ["asian", "white"]

    def test_missing_fields_come_from_where_the_dictionary_keeps_them(self):
        """Types and values under oneOf, descriptions under common, arrays' items enums."""
        props = self.properties()
        assert props["age_at_index"].type == "integer"
        assert props["vital_status"].type == "enum"
        assert props["vital_status"].values == ["Alive", "Dead"]
        assert props["state"].description == "The record's state."
        assert props["sites"].values == ["Lung", "Liver"]

    def test_a_link_stays_empty(self):
        """A link property has no description, type or values of its own."""
        cases = self.properties()["cases"]
        assert (cases.description, cases.type, cases.values) == ("", "", None)

    def test_more_fallbacks_and_their_order(self):
        """A bare enum is an enum; the own description beats common, which beats term."""
        dictionary = {
            "case": {
                "description": "A case.",
                "properties": {
                    "gender": {"enum": ["female", "male"]},
                    "own": {
                        "description": "Own words.",
                        "common": {"description": "Shared words."},
                    },
                    "shared": {
                        "common": {"description": "Shared words."},
                        "term": {"description": "Term words."},
                    },
                    "termed": {"term": {"description": "Term words."}},
                },
            }
        }
        (node,) = SimpleDataModel.from_gdc_model(dictionary).nodes
        props = {p.name: p for p in node.properties}
        assert props["gender"].type == "enum"
        assert props["own"].description == "Own words."
        assert props["shared"].description == "Shared words."
        assert props["termed"].description == "Term words."

    def test_one_of_types_and_values_are_not_repeated(self):
        """Types and values gathered from several oneOf options appear once each."""
        spec = {
            "oneOf": [
                {"type": "string", "enum": ["A", "B"]},
                {"type": "string", "enum": ["A", "C"]},
                {"type": "null"},
            ]
        }
        prop = SimpleDataModel.from_gdc_model({"n": {"properties": {"p": spec}}})
        (prop,) = prop.nodes[0].properties
        assert (prop.type, prop.values) == ("string", ["A", "B", "C"])

    @pytest.mark.parametrize(
        "spec",
        [
            {"items": True, "type": "array"},
            {"items": "enum"},
            {"oneOf": [{"enum": None}, {"type": "null"}]},
            {"oneOf": True},
            {"term": "_terms.yaml#/race"},
            {"common": ["not", "a", "definition"]},
            {"oneOf": [{"type": ["string", "null"]}, {"type": "integer"}]},
        ],
    )
    def test_unexpected_shapes_do_not_raise(self, spec):
        """A fallback that finds something other than a definition is ignored, not an error."""
        SimpleDataModel.from_gdc_model({"n": {"properties": {"p": spec}}})

    def test_a_null_description_is_rejected(self):
        """A property's own null description fails validation, so the dispatcher tries another parser."""
        with pytest.raises(ValidationError):
            SimpleDataModel.from_gdc_model(
                {"n": {"properties": {"p": {"description": None}}}}
            )

    def test_the_dispatcher_keeps_this_parse_for_a_gen3_dictionary(self):
        """get_from_unknown_json_format returns from_gdc_model's nodes and properties."""
        expected = SimpleDataModel.from_gdc_model(self.DICTIONARY)
        model = SimpleDataModel.get_from_unknown_json_format(
            json.dumps(self.DICTIONARY)
        )
        assert [(n.name, [p.name for p in n.properties]) for n in model.nodes] == [
            (n.name, [p.name for p in n.properties]) for n in expected.nodes
        ]
