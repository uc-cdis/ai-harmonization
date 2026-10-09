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

    def __init__(self, matches, document_formatter=None, input_target_model=None):
        self._matches = matches
        self.document_formatter = document_formatter or get_node_property_as_string
        self.embedding_function = None
        self.input_target_model = input_target_model
        self.queries = []

    def find_similar_target_slots(self, query_text, **kwargs):
        """Stand-in vector search: log the query, return the canned matches up to k."""
        self.queries.append(query_text)
        matches = self._matches
        limit = kwargs.get("k")
        return matches[:limit] if limit is not None else matches


def match(slot_key, similarity, target_description="a description"):
    """Build the TargetSlotMatch an index returns for one target slot."""
    return TargetSlotMatch(
        slot_key=slot_key,
        similarity=similarity,
        target_description=target_description,
    )


@pytest.fixture
def source_node():
    """An empty source node with no description or properties."""
    return Node(name="pht999999", description="", links=[], properties=[])


@pytest.fixture
def source_property():
    """An integer source variable with a name and description."""
    return Property(name="AGE", description="Age at enrollment", type="integer")


class TestFromIndexes:
    def test_rejects_empty_indexes(self):
        """An empty index mapping raises ValueError."""
        with pytest.raises(ValueError):
            MultiPromptSimilaritySearch.from_indexes({})

    def test_keeps_variant_labels(self):
        """Each index stays keyed by the variant label it was passed under."""
        search = MultiPromptSimilaritySearch.from_indexes(
            {"A": StubIndex([]), "B": StubIndex([])}
        )
        assert set(search.indexes) == {"A", "B"}


class TestGetSuggestionsForProperty:
    def test_merges_across_variants(self, source_node, source_property):
        """Suggestions combine the matches from every variant."""
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
        """A slot matched by two variants appears once, with the higher similarity."""
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
        """The merged suggestion's prompt_variant is the variant that scored higher."""
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex([match("TargetClass.field_a", 0.55)]),
                "B": StubIndex([match("TargetClass.field_a", 0.91)]),
            }
        )
        suggestions = search.get_suggestions_for_property(source_node, source_property)
        assert suggestions[0].target_additional_metadata["prompt_variant"] == "B"

    def test_sorted_by_similarity_descending(self, source_node, source_property):
        """Suggestions come back best first, whatever order the index returned."""
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
        """With k=3, the four distinct slots from two variants are cut to three."""
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
        """Each variant formats the source query with its own document_formatter."""
        index_a = StubIndex(
            [], document_formatter=lambda n, p: f"{n.name}.{p.name}: {p.description}"
        )
        index_b = StubIndex([], document_formatter=lambda n, p: f"ONLY NAME {p.name}")
        search = MultiPromptSimilaritySearch.from_indexes({"A": index_a, "B": index_b})

        search.get_suggestions_for_property(source_node, source_property)

        assert index_a.queries == ["pht999999.AGE: Age at enrollment"]
        assert index_b.queries == ["ONLY NAME AGE"]

    def test_source_fields_copied_onto_suggestions(self, source_node):
        """A suggestion carries its source variable's name, description, value
        labels and variable_id."""
        source_property = Property(
            name="SEX",
            description="Biological sex",
            type="string/encoded",
            additional_metadata={
                "value_labels": ["1=Male", "2=Female"],
                "variable_id": "phv99999902.v1",
            },
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
        assert suggestion.source_additional_metadata["variable_id"] == (
            "phv99999902.v1"
        )
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
        """When no variant matches anything the result is an empty list."""
        search = MultiPromptSimilaritySearch.from_indexes({"A": StubIndex([])})
        assert search.get_suggestions_for_property(source_node, source_property) == []


class TestIterAndBatchInterface:
    @pytest.fixture
    def source_model(self):
        """Two source tables holding three variables between them."""
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
        """A one-variant search whose index returns two matches for any query."""
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
        """One group per source variable across all tables, each with both matches."""
        groups = list(search.iter_suggestions_by_property(source_model))
        assert len(groups) == 3
        assert all(len(group) == 2 for group in groups)

    def test_get_harmonization_suggestions_flattens(self, search, source_model):
        """The batch call flattens the groups into one list of six suggestions."""
        result = search.get_harmonization_suggestions(source_model)
        assert len(result.suggestions) == 6

    def test_conforms_to_the_benchmark_call_signature(self, search, source_model):
        """The benchmark harness calls this with both models as keywords, and
        HarmonizationApproach declares input_target_model required, so passing
        it must keep working."""
        search.input_target_model = source_model
        result = search.get_harmonization_suggestions(
            input_source_model=source_model,
            input_target_model=source_model,
            k=1,
        )
        assert len(result.suggestions) == 3

    def test_passing_the_same_target_model_is_accepted(self, source_model):
        """The benchmark harness and the curation notebooks pass the target
        model explicitly, and it is the model the instance embedded."""
        search = MultiPromptSimilaritySearch.from_indexes(
            {"A": StubIndex([match("TargetClass.field_a", 0.8)])}
        )
        search.input_target_model = source_model
        result = search.get_harmonization_suggestions(
            source_model, input_target_model=source_model
        )
        assert len(result.suggestions) == 3

    def test_passing_a_different_target_model_raises(self, source_model):
        """Answering against the embedded target while the caller asked for
        another one would be a silently wrong result."""
        search = MultiPromptSimilaritySearch.from_indexes(
            {"A": StubIndex([match("TargetClass.field_a", 0.8)])}
        )
        search.input_target_model = source_model
        other = SimpleDataModel(
            nodes=[
                Node(
                    name="somethingelse",
                    description="",
                    links=[],
                    properties=[
                        Property(name="x", description="x", type="string"),
                    ],
                )
            ]
        )
        with pytest.raises(ValueError, match="different input_target_model"):
            search.get_harmonization_suggestions(source_model, input_target_model=other)

    def test_from_indexes_takes_the_target_model_from_its_indexes(self, source_model):
        """Instances built this way still know their target, so the check in
        get_harmonization_suggestions applies to them too."""
        search = MultiPromptSimilaritySearch.from_indexes(
            {
                "A": StubIndex(
                    [match("TargetClass.field_a", 0.8)],
                    input_target_model=source_model,
                )
            }
        )
        assert search.input_target_model is source_model

    def test_to_simlified_dataframe_round_trip(self, search, source_model):
        """The simplified frame shows source node.property and a Similarity column."""
        df = search.get_harmonization_suggestions(source_model).to_simlified_dataframe()
        assert list(df["Original Node.Property"])[0] == "pht001.AGE"
        assert "Similarity" in df.columns


class TestDocumentFormatterHook:
    @pytest.fixture
    def target_model(self):
        """A target model with a single described integer slot."""
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
        """With no formatter, document text is "node.property (type): description"."""
        documents = get_data_model_as_langchain_documents(target_model)
        assert (
            documents[0].page_content == "TargetClass.field_a (integer): Age in years"
        )

    def test_custom_formatter_controls_embedded_text(self, target_model):
        """A custom formatter's output is the document text, verbatim."""
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
        """Each property of a node becomes its own document."""
        target_model.nodes[0].properties.append(
            Property(name="field_b", description="Sex", type="string")
        )
        assert len(get_data_model_as_langchain_documents(target_model)) == 2
