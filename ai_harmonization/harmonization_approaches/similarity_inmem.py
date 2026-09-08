import logging
from typing import Callable, Dict, Iterator, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ai_harmonization.harmonization_approaches.base import (
    HarmonizationApproach,
    HarmonizationSuggestions,
    SingleHarmonizationSuggestion,
)
from ai_harmonization.simple_data_model import (
    Node,
    Property,
    SimpleDataModel,
    get_data_model_as_langchain_documents,
    get_node_prop_type_desc_from_string,
    get_node_property_as_string,
)
from ai_harmonization.utils import (
    TEMP_DIR,
    add_documents_to_vectorstore,
    get_langchain_vectorstore_and_persistent_client,
    get_similar_documents,
)


class ExistingVectorstoreException(BaseException):
    pass


class SuggestionInfo(TypedDict):
    node: Node
    property: Property
    matches: List[Document]


class TargetSlotMatch(TypedDict):
    """A single target slot returned by a similarity query."""

    slot_key: str
    similarity: float
    target_description: str


class SimilaritySearchInMemoryVectorDb(HarmonizationApproach):

    def __init__(
        self,
        vectordb_persist_directory_name: str,
        input_target_model: SimpleDataModel,
        embedding_function: HuggingFaceEmbeddings | None = None,
        force_vectorstore_recreation: bool = False,
        batch_size: int | None = None,
        document_formatter: Optional[Callable[[Node, Property], str]] = None,
    ):
        """
        Args:
            vectordb_persist_directory_name: Chroma collection / persist directory name.
            input_target_model: The target schema to embed and search against.
            embedding_function: Embedding model. Defaults to all-mpnet-base-v2.
            force_vectorstore_recreation: Delete and rebuild an existing collection.
            batch_size: Number of documents to add per batch.
            document_formatter: Turns a (Node, Property) pair into the text that
                gets embedded. Defaults to ``get_node_property_as_string``.
                The same formatter is applied to source properties at query
                time, so both sides of the comparison share one text format.
        """
        super().__init__()

        self.embedding_function = embedding_function or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.document_formatter = document_formatter or get_node_property_as_string
        self.vectordb_persist_directory_name = vectordb_persist_directory_name
        vectorstore, persistent_client = (
            get_langchain_vectorstore_and_persistent_client(
                persist_directory_name=self.vectordb_persist_directory_name,
                embedding_function=self.embedding_function,
            )
        )
        self.vectorstore = vectorstore
        self.persistent_client = persistent_client

        try:
            self._add_target_to_vector_database(
                input_target_model,
                force_recreation=force_vectorstore_recreation,
                batch_size=batch_size,
            )
        except ExistingVectorstoreException:
            logging.info("vectorstore already exists, NOT recreating...")
            pass

    def _add_target_to_vector_database(
        self, input_target_model, force_recreation=False, batch_size=None
    ):
        if force_recreation:
            all_ids = self.vectorstore.get()["ids"]
            if all_ids:
                self.vectorstore.delete(ids=all_ids)

        if len(self.vectorstore.get()["ids"]) != 0:
            raise ExistingVectorstoreException(
                f"Vectorstore in persist directory {self.vectordb_persist_directory_name} already has data here: {TEMP_DIR}, "
                "aborting re-adding. Delete the persist directory, use a new one, or force recreation."
            )

        target_docs = get_data_model_as_langchain_documents(
            input_target_model, document_formatter=self.document_formatter
        )

        add_documents_to_vectorstore(
            target_docs, self.vectorstore, self.persistent_client, batch_size
        )

    def find_similar_target_slots(
        self, query_text: str, **kwargs
    ) -> List[TargetSlotMatch]:
        """
        Search the embedded target model with raw text.

        This is the single-query primitive underneath
        ``get_harmonization_suggestions``; use that instead when you have a
        source model rather than a bare string.

        Args:
            query_text: Text to embed and search with.
            **kwargs: Passed through to the vectorstore, e.g. ``k`` and
                ``score_threshold``.

        Returns:
            List[TargetSlotMatch]: Matching target slots, best first.
        """
        matches = []
        for document, similarity in get_similar_documents(
            self.vectorstore, query_text, **kwargs
        ):
            node_name, property_name, _, parsed_description = (
                get_node_prop_type_desc_from_string(document.page_content)
            )
            matches.append(
                TargetSlotMatch(
                    slot_key=(
                        f"{node_name}.{property_name}"
                        if node_name
                        else document.page_content
                    ),
                    similarity=similarity,
                    # Formatters may omit the description from the embedded
                    # text, so prefer the copy kept in document metadata.
                    target_description=(
                        document.metadata.get("description")
                        or parsed_description
                        or document.page_content
                    ),
                )
            )
        return matches

    def get_harmonization_suggestions(
        self,
        input_source_model: SimpleDataModel,
        input_target_model: Optional[SimpleDataModel] = None,
        **kwargs,
    ) -> HarmonizationSuggestions:
        """
        Args:
            input_source_model: The model whose properties get mapped.
            input_target_model: Unused. The target model is embedded in the
                vectorstore at construction time; the parameter is kept to
                satisfy the HarmonizationApproach interface.
            **kwargs: Passed through to the vectorstore, e.g. ``k`` and
                ``score_threshold``.
        """
        suggestions_for_output_model = self._get_suggestions_for_source_model(
            input_source_model, **kwargs
        )
        suggestions = []
        for suggestion_info in suggestions_for_output_model:
            source_property = suggestion_info["property"]
            source_node = suggestion_info["node"]
            suggested_docs = suggestion_info["matches"]

            source_node_name, source_prop_name, source_prop_type, source_prop_desc = (
                source_node.name,
                source_property.name,
                source_property.type,
                source_property.description,
            )

            source_additional_metadata = {}
            source_additional_metadata["type"] = source_prop_type

            for single_suggested_doc, similarity in suggested_docs:
                target_text = single_suggested_doc.page_content

                (
                    target_node_name,
                    target_prop_name,
                    target_prop_type,
                    target_prop_desc,
                ) = get_node_prop_type_desc_from_string(target_text)

                target_additional_metadata = {}
                target_additional_metadata["type"] = target_prop_type

                single_suggestion = SingleHarmonizationSuggestion(
                    source_node=source_node_name,
                    source_property=source_prop_name,
                    source_description=source_prop_desc,
                    source_additional_metadata=source_additional_metadata,
                    target_node=target_node_name,
                    target_property=target_prop_name,
                    target_description=(
                        single_suggested_doc.metadata.get("description")
                        or target_prop_desc
                    ),
                    target_additional_metadata=target_additional_metadata,
                    similarity=similarity,
                )
                suggestions.append(single_suggestion)

        return HarmonizationSuggestions(suggestions=suggestions)

    def _get_suggestions_for_source_model(
        self, input_source_model: SimpleDataModel, **kwargs
    ) -> List[SuggestionInfo]:
        suggestions_for_output_model = []
        for node in input_source_model.nodes:
            for node_property in node.properties:
                suggestion_info = {
                    "node": node,
                    "property": node_property,
                    "matches": [],
                }

                source_query = self.document_formatter(node, node_property)
                matches = get_similar_documents(
                    self.vectorstore,
                    source_query,
                    **kwargs,
                )

                suggestion_info["matches"] = matches

                if matches:
                    suggestions_for_output_model.append(suggestion_info)

        if not suggestions_for_output_model:
            raise Exception("Cannot parse node properties")

        return suggestions_for_output_model


class MultiPromptSimilaritySearch(HarmonizationApproach):
    """
    Runs the same similarity search several times over differently-worded
    prompts, then keeps the best match per target slot.

    One SimilaritySearchInMemoryVectorDb is built per prompt variant, each
    embedding the target model with its own formatter. A source property is
    queried against every variant using that variant's formatter, so both
    sides of each comparison share one text format. Results are merged by
    target slot, keeping the highest similarity across variants, and the
    winning variant's label is recorded on each suggestion under
    ``target_additional_metadata['prompt_variant']``.

    Combining variants matters because no single wording wins everywhere:
    enum values help match categorical variables, while measurement units
    help match lab results.
    """

    def __init__(
        self,
        input_target_model: SimpleDataModel,
        prompt_variants: Dict[str, Callable[[Node, Property], str]],
        embedding_function: HuggingFaceEmbeddings | None = None,
        collection_name_prefix: str = "target_model",
        force_vectorstore_recreation: bool = False,
        batch_size: int | None = None,
    ):
        """
        Args:
            input_target_model: The target schema to embed, once per variant.
            prompt_variants: Maps a short label (e.g. ``'A'``, recorded in the
                ``prompt_variant`` output column) to a ``(Node, Property) -> str``
                formatter. See ``ai_harmonization.formatters``.
            embedding_function: Embedding model, shared by every variant.
            collection_name_prefix: Chroma collections are named
                ``{prefix}_variant_{label}``. Change the prefix when the target
                schema changes, or an existing collection is silently reused.
            force_vectorstore_recreation: Rebuild collections that already exist.
                Use this after editing a formatter or the target schema.
            batch_size: Number of documents to add per batch.
        """
        super().__init__()

        if not prompt_variants:
            raise ValueError("prompt_variants must contain at least one formatter.")

        self.embedding_function = embedding_function or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.indexes: Dict[str, SimilaritySearchInMemoryVectorDb] = {
            label: SimilaritySearchInMemoryVectorDb(
                vectordb_persist_directory_name=(
                    f"{collection_name_prefix}_variant_{label.lower()}"
                ),
                input_target_model=input_target_model,
                embedding_function=self.embedding_function,
                force_vectorstore_recreation=force_vectorstore_recreation,
                batch_size=batch_size,
                document_formatter=formatter,
            )
            for label, formatter in prompt_variants.items()
        }

    @classmethod
    def from_indexes(
        cls, indexes: Dict[str, SimilaritySearchInMemoryVectorDb]
    ) -> "MultiPromptSimilaritySearch":
        """
        Build from indexes that already exist, instead of embedding the target
        model again.

        Args:
            indexes: Maps a prompt variant label to an index built with that
                variant's ``document_formatter``.

        Returns:
            MultiPromptSimilaritySearch: Merging over the given indexes.
        """
        if not indexes:
            raise ValueError("indexes must contain at least one index.")

        instance = cls.__new__(cls)
        instance.indexes = dict(indexes)
        instance.embedding_function = next(iter(indexes.values())).embedding_function
        return instance

    def get_suggestions_for_property(
        self, source_node: Node, source_property: Property, **kwargs
    ) -> List[SingleHarmonizationSuggestion]:
        """
        Map one source property against every prompt variant.

        Args:
            source_node: The node containing the source property.
            source_property: The source property to map.
            **kwargs: Passed through to each variant's vectorstore. ``k`` caps
                both the per-variant query size and the merged result size.

        Returns:
            List[SingleHarmonizationSuggestion]: Deduplicated by target slot,
                highest similarity first, at most ``k`` entries.
        """
        best_by_slot: Dict[str, tuple[TargetSlotMatch, str]] = {}

        for label, index in self.indexes.items():
            query_text = index.document_formatter(source_node, source_property)
            for match in index.find_similar_target_slots(query_text, **kwargs):
                current = best_by_slot.get(match["slot_key"])
                if current is None or match["similarity"] > current[0]["similarity"]:
                    best_by_slot[match["slot_key"]] = (match, label)

        ranked = sorted(
            best_by_slot.values(), key=lambda pair: pair[0]["similarity"], reverse=True
        )
        limit = kwargs.get("k")
        if limit is not None:
            ranked = ranked[:limit]

        return [
            SingleHarmonizationSuggestion(
                source_node=source_node.name,
                source_property=source_property.name,
                source_description=source_property.description,
                source_additional_metadata={
                    "type": source_property.type,
                    "value_labels": (source_property.additional_metadata or {}).get(
                        "value_labels", []
                    ),
                },
                target_node=match["slot_key"].rsplit(".", 1)[0],
                target_property=match["slot_key"].rsplit(".", 1)[-1],
                target_description=match["target_description"],
                target_additional_metadata={"prompt_variant": label},
                similarity=match["similarity"],
            )
            for match, label in ranked
        ]

    def iter_suggestions_by_property(
        self, input_source_model: SimpleDataModel, **kwargs
    ) -> Iterator[List[SingleHarmonizationSuggestion]]:
        """
        Yield suggestions one source property at a time.

        Use this instead of ``get_harmonization_suggestions`` for long runs
        where results should be written out incrementally rather than held in
        memory until every property is mapped.

        Yields:
            List[SingleHarmonizationSuggestion]: Suggestions for one property,
                highest similarity first.
        """
        for node in input_source_model.nodes:
            for node_property in node.properties:
                yield self.get_suggestions_for_property(node, node_property, **kwargs)

    def get_harmonization_suggestions(
        self,
        input_source_model: SimpleDataModel,
        input_target_model: Optional[SimpleDataModel] = None,
        **kwargs,
    ) -> HarmonizationSuggestions:
        """
        Args:
            input_source_model: The model whose properties get mapped.
            input_target_model: Unused. The target model is embedded in the
                vectorstores at construction time; the parameter is kept to
                satisfy the HarmonizationApproach interface.
            **kwargs: Passed through to each variant's vectorstore, e.g. ``k``.
        """
        suggestions = [
            suggestion
            for property_suggestions in self.iter_suggestions_by_property(
                input_source_model, **kwargs
            )
            for suggestion in property_suggestions
        ]
        return HarmonizationSuggestions(suggestions=suggestions)
