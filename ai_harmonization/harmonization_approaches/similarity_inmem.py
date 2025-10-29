import logging
from typing import List, TypedDict

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


class SimilaritySearchInMemoryVectorDb(HarmonizationApproach):

    def __init__(
        self,
        vectordb_persist_directory_name: str,
        input_target_model: SimpleDataModel,
        embedding_function: HuggingFaceEmbeddings | None = None,
        force_vectorstore_recreation: bool = False,
        batch_size: int | None = None,
    ):
        super().__init__()

        self.embedding_function = embedding_function or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
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

        target_docs = get_data_model_as_langchain_documents(input_target_model)

        add_documents_to_vectorstore(
            target_docs, self.vectorstore, self.persistent_client, batch_size
        )

    def get_harmonization_suggestions(
        self,
        input_source_model: SimpleDataModel,
        input_target_model: SimpleDataModel,
        **kwargs,
    ):
        # note: k and score_threshold are in kwargs
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
                    target_description=target_prop_desc,
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

                source_query = get_node_property_as_string(node, node_property)
                matches = get_similar_documents(
                    self.vectorstore,
                    source_query,
                    **kwargs,
                )

                suggestion_info["matches"] = matches

                if matches:
                    suggestions_for_output_model.append(suggestion_info)

        if not suggestions_for_output_model:
            raise Exception(f"Cannot parse node properties")

        return suggestions_for_output_model
