"""Tests for SimpleDataModel.from_linkml_yaml() in ai_harmonization.simple_data_model."""

import textwrap

import pytest

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
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        names = [n.name for n in model.nodes]
        assert "Subject" in names

    def test_empty_class_excluded(self):
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        names = [n.name for n in model.nodes]
        assert "Empty" not in names

    def test_node_description_preserved(self):
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        subject = next(n for n in model.nodes if n.name == "Subject")
        assert subject.description == "A study participant"

    def test_enum_values_resolved(self):
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        subject = next(n for n in model.nodes if n.name == "Subject")
        sex = next(p for p in subject.properties if p.name == "sex")
        assert set(sex.values) == {"Male", "Female", "Unknown"}

    def test_non_enum_range_has_no_values(self):
        model = SimpleDataModel.from_linkml_yaml(MINIMAL_SCHEMA)
        subject = next(n for n in model.nodes if n.name == "Subject")
        age = next(p for p in subject.properties if p.name == "age")
        assert age.values is None

    def test_is_a_inheritance(self):
        model = SimpleDataModel.from_linkml_yaml(INHERITANCE_SCHEMA)
        child = next(n for n in model.nodes if n.name == "Child")
        prop_names = {p.name for p in child.properties}
        assert "id" in prop_names
        assert "extra" in prop_names

    def test_mixin_inheritance(self):
        model = SimpleDataModel.from_linkml_yaml(MIXIN_SCHEMA)
        record = next(n for n in model.nodes if n.name == "Record")
        prop_names = {p.name for p in record.properties}
        assert "created_at" in prop_names
        assert "value" in prop_names

    def test_enum_inherits_chain(self):
        model = SimpleDataModel.from_linkml_yaml(ENUM_INHERITANCE_SCHEMA)
        sample = next(n for n in model.nodes if n.name == "Sample")
        status = next(p for p in sample.properties if p.name == "status")
        assert set(status.values) == {"Active", "Inactive", "Pending"}

    def test_global_slots_resolved(self):
        model = SimpleDataModel.from_linkml_yaml(GLOBAL_SLOTS_SCHEMA)
        observation = next(n for n in model.nodes if n.name == "Observation")
        by_name = {p.name: p for p in observation.properties}
        assert set(by_name) == {"observation_id", "value"}
        assert by_name["value"].type == "float"
        assert by_name["observation_id"].description == "Primary key"

    def test_undescribed_field_gets_default_description(self):
        model = SimpleDataModel.from_linkml_yaml(
            "classes:\n  Thing:\n    attributes:\n      key: {}\n"
        )
        thing = next(n for n in model.nodes if n.name == "Thing")
        assert thing.properties[0].description == DEFAULT_PROPERTY_DESCRIPTION

    def test_missing_range_defaults_to_string(self):
        model = SimpleDataModel.from_linkml_yaml(
            "classes:\n  Thing:\n    attributes:\n      key:\n        description: A key\n"
        )
        thing = next(n for n in model.nodes if n.name == "Thing")
        assert thing.properties[0].type == "string"

    def test_inheritance_cycle_does_not_recurse_forever(self):
        model = SimpleDataModel.from_linkml_yaml(CYCLIC_SCHEMA)
        by_name = {n.name: n for n in model.nodes}
        assert {p.name for p in by_name["A"].properties} == {"a_field", "b_field"}

    @pytest.mark.parametrize("empty_yaml", ["", "{}", "classes: {}"])
    def test_empty_schema_returns_empty_model(self, empty_yaml):
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
        assert self._status_values(getattr(self, form)) == ["ACTIVE", "INACTIVE"]

    def test_empty_permissible_values_leaves_values_unset(self):
        schema = (
            "classes:\n  S:\n    attributes:\n      f: {range: E}\nenums:\n  E: {}\n"
        )
        model = SimpleDataModel.from_linkml_yaml(schema)
        assert model.nodes[0].properties[0].values is None
