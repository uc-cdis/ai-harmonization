"""Tests for the multi-prompt merge logic and the document_formatter hook.

These avoid building real vector stores: MultiPromptSimilaritySearch is
assembled from stub indexes via from_indexes(), and the formatter hook is
exercised through get_data_model_as_langchain_documents().
"""

import pytest

from ai_harmonization.harmonization_approaches.similarity_inmem import (
    MultiPromptSimilaritySearch,
    TargetSlotMatch,
)
from ai_harmonization.simple_data_model import (
    Node,
    Property,
    SimpleDataModel,
    get_data_model_as_langchain_documents,
    get_node_property_as_string,
)


class StubIndex:
    """Returns canned matches, recording the query text it was given."""

    def __init__(self, matches, document_formatter=None):
        self._matches = matches
        self.document_formatter = document_formatter or get_node_property_as_string
        self.embedding_function = None
        self.queries = []

    def find_similar_target_slots(self, query_text, **kwargs):
        self.queries.append(query_text)
        matches = self._matches
        limit = kwargs.get("k")
        return matches[:limit] if limit is not None else matches


def match(slot_key, similarity, target_description="a description"):
    return TargetSlotMatch(
        slot_key=slot_key,
        similarity=similarity,
        target_description=target_description,
    )


@pytest.fixture
def source_node():
    return Node(name="pht999999", description="", links=[], properties=[])


@pytest.fixture
def source_property():
    return Property(name="AGE", description="Age at enrollment", type="integer")


class TestFromIndexes:
    def test_rejects_empty_indexes(self):
        with pytest.raises(ValueError):
            MultiPromptSimilaritySearch.from_indexes({})

    def test_keeps_variant_labels(self):
        search = MultiPromptSimilaritySearch.from_indexes(
            {"A": StubIndex([]), "B": StubIndex([])}
        )
        assert set(search.indexes) == {"A", "B"}


class TestGetSuggestionsForProperty:
    def test_merges_across_variants(self, source_node, source_property):
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex([match("TargetClass.field_a", 0.7)]),
                "B": StubIndex([match("OtherClass.field_e", 0.6)]),
            }
        )
        suggestions = search.get_suggestions_for_property(source_node, source_property)
        slots = {f"{s.target_node}.{s.target_property}" for s in suggestions}
        assert slots == {"TargetClass.field_a", "OtherClass.field_e"}

    def test_deduplicates_by_slot_keeping_best_similarity(
        self, source_node, source_property
    ):
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex([match("TargetClass.field_a", 0.55)]),
                "B": StubIndex([match("TargetClass.field_a", 0.91)]),
            }
        )
        suggestions = search.get_suggestions_for_property(source_node, source_property)
        assert len(suggestions) == 1
        assert suggestions[0].similarity == 0.91

    def test_records_the_winning_variant(self, source_node, source_property):
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex([match("TargetClass.field_a", 0.55)]),
                "B": StubIndex([match("TargetClass.field_a", 0.91)]),
            }
        )
        suggestions = search.get_suggestions_for_property(source_node, source_property)
        assert suggestions[0].target_additional_metadata["prompt_variant"] == "B"

    def test_sorted_by_similarity_descending(self, source_node, source_property):
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex(
                    [
                        match("TargetClass.field_a", 0.4),
                        match("TargetClass.field_b", 0.95),
                        match("OtherClass.field_e", 0.62),
                    ]
                ),
            }
        )
        suggestions = search.get_suggestions_for_property(source_node, source_property)
        assert [s.similarity for s in suggestions] == [0.95, 0.62, 0.4]

    def test_k_caps_the_merged_result(self, source_node, source_property):
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex(
                    [
                        match("TargetClass.field_a", 0.9),
                        match("TargetClass.field_c", 0.8),
                    ]
                ),
                "B": StubIndex(
                    [match("OtherClass.field_e", 0.7), match("OtherClass.field_f", 0.6)]
                ),
            }
        )
        suggestions = search.get_suggestions_for_property(
            source_node, source_property, k=3
        )
        assert len(suggestions) == 3

    def test_each_variant_queried_with_its_own_formatter(
        self, source_node, source_property
    ):
        index_a = StubIndex(
            [], document_formatter=lambda n, p: f"{n.name}.{p.name}: {p.description}"
        )
        index_b = StubIndex([], document_formatter=lambda n, p: f"ONLY NAME {p.name}")
        search = MultiPromptSimilaritySearch.from_indexes({"A": index_a, "B": index_b})

        search.get_suggestions_for_property(source_node, source_property)

        assert index_a.queries == ["pht999999.AGE: Age at enrollment"]
        assert index_b.queries == ["ONLY NAME AGE"]

    def test_source_fields_copied_onto_suggestions(self, source_node):
        source_property = Property(
            name="SEX",
            description="Biological sex",
            type="string/encoded",
            additional_metadata={"value_labels": ["1=Male", "2=Female"]},
        )
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex(
                    [match("TargetClass.field_c", 0.88, "Sex of the participant")]
                ),
            }
        )
        suggestion = search.get_suggestions_for_property(source_node, source_property)[
            0
        ]

        assert suggestion.source_node == "pht999999"
        assert suggestion.source_property == "SEX"
        assert suggestion.source_description == "Biological sex"
        assert suggestion.source_additional_metadata["value_labels"] == [
            "1=Male",
            "2=Female",
        ]
        assert suggestion.target_description == "Sex of the participant"

    def test_slot_key_splits_on_the_last_dot(self, source_node, source_property):
        """Target node names can contain dots; only the property is split off."""
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex([match("schema.TargetClass.field_a", 0.8)]),
            }
        )
        suggestion = search.get_suggestions_for_property(source_node, source_property)[
            0
        ]
        assert suggestion.target_node == "schema.TargetClass"
        assert suggestion.target_property == "field_a"

    def test_no_matches_yields_no_suggestions(self, source_node, source_property):
        search = MultiPromptSimilaritySearch.from_indexes({"A": StubIndex([])})
        assert search.get_suggestions_for_property(source_node, source_property) == []


class TestIterAndBatchInterface:
    @pytest.fixture
    def source_model(self):
        return SimpleDataModel(
            nodes=[
                Node(
                    name="pht001",
                    description="",
                    links=[],
                    properties=[
                        Property(name="AGE", description="Age", type="integer"),
                        Property(name="SEX", description="Sex", type="string"),
                    ],
                ),
                Node(
                    name="pht002",
                    description="",
                    links=[],
                    properties=[
                        Property(
                            name="BMI", description="Body mass index", type="float"
                        ),
                    ],
                ),
            ]
        )

    @pytest.fixture
    def search(self):
        return MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex(
                    [
                        match("TargetClass.field_a", 0.8),
                        match("TargetClass.field_c", 0.7),
                    ]
                ),
            }
        )

    def test_iter_yields_one_group_per_property(self, search, source_model):
        groups = list(search.iter_suggestions_by_property(source_model))
        assert len(groups) == 3
        assert all(len(group) == 2 for group in groups)

    def test_get_harmonization_suggestions_flattens(self, search, source_model):
        result = search.get_harmonization_suggestions(source_model)
        assert len(result.suggestions) == 6

    def test_conforms_to_the_benchmark_call_signature(self, search, source_model):
        """The benchmark harness calls this with both models as keywords."""
        result = search.get_harmonization_suggestions(
            input_source_model=source_model,
            input_target_model=source_model,
            k=1,
        )
        assert len(result.suggestions) == 3

    def test_to_simlified_dataframe_round_trip(self, search, source_model):
        df = search.get_harmonization_suggestions(source_model).to_simlified_dataframe()
        assert list(df["Original Node.Property"])[0] == "pht001.AGE"
        assert "Similarity" in df.columns


class TestDocumentFormatterHook:
    @pytest.fixture
    def target_model(self):
        return SimpleDataModel(
            nodes=[
                Node(
                    name="TargetClass",
                    description="",
                    links=[],
                    properties=[
                        Property(
                            name="field_a", description="Age in years", type="integer"
                        ),
                    ],
                ),
            ]
        )

    def test_defaults_to_node_property_as_string(self, target_model):
        documents = get_data_model_as_langchain_documents(target_model)
        assert (
            documents[0].page_content == "TargetClass.field_a (integer): Age in years"
        )

    def test_custom_formatter_controls_embedded_text(self, target_model):
        documents = get_data_model_as_langchain_documents(
            target_model, document_formatter=lambda n, p: f"{n.name}.{p.name}"
        )
        assert documents[0].page_content == "TargetClass.field_a"

    def test_description_kept_in_metadata_even_when_omitted_from_text(
        self, target_model
    ):
        """Variant D leaves the description out of the embedded text, so the
        review CSV has to read it back from document metadata."""
        documents = get_data_model_as_langchain_documents(
            target_model,
            document_formatter=lambda n, p: f"{n.name}.{p.name} ({p.type}):",
        )
        assert "Age in years" not in documents[0].page_content
        assert documents[0].metadata["description"] == "Age in years"

    def test_one_document_per_property(self, target_model):
        target_model.nodes[0].properties.append(
            Property(name="field_b", description="Sex", type="string")
        )
        assert len(get_data_model_as_langchain_documents(target_model)) == 2
