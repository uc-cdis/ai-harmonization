import ast
import json
import logging
import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings

from harmonization.cde import (
    EXAMPLE_AI_MODEL_OUTPUT_JSON,
    TEMP_DIR,
    add_documents_to_vectorstore,
    get_cdes_as_langchain_documents,
    get_data_model_as_langchain_documents,
    get_gen3_data_model_as_langchain_documents,
    get_gen3_data_model_as_node_prop_descriptions,
    get_interactive_table_for_suggestions,
    get_langchain_vectorstore_and_persistent_client,
    get_similar_documents,
    get_similar_documents_as_string,
)
from harmonization.harmonization_approaches.harmonization import (
    HarmonizationApproach,
    HarmonizationSuggestions,
    SingleHarmonizationSuggestion,
)


class SimilaritySearchInMemoryVectorDb(HarmonizationApproach):

    def __init__(
        self,
        vectordb_persist_directory_name: str,
        input_target_model_type: str = "gen3",
        embedding_function: HuggingFaceEmbeddings = None,
    ):
        super().__init__()

        if input_target_model_type == "gen3":
            logging.info(f"Treating input target as type: 'gen3'")
        else:
            raise NotImplementedError(
                f"input_target_model_type of '{input_target_model_type}' is not supported at this time."
            )

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

    def add_target_to_vector_database(self, input_target_model, force_recreation=False):
        if force_recreation:
            all_ids = self.vectorstore.get()["ids"]
            if all_ids:
                self.vectorstore.delete(ids=all_ids)

        if len(self.vectorstore.get()["ids"]) != 0:
            raise Exception(
                f"Vectorstore in persist directory {self.vectordb_persist_directory_name} already has data here: {TEMP_DIR}, "
                "aborting re-adding. Delete the persist directory or use a new one."
            )

        target_docs = get_gen3_data_model_as_langchain_documents(input_target_model)

        add_documents_to_vectorstore(
            target_docs, self.vectorstore, self.persistent_client
        )

    def get_harmonization_suggestions(
        self, input_source_model, input_target_model, score_threshold=0.5, k=4, **kwargs
    ):
        suggestions_for_output_model = self._get_suggestions_for_ai_model_output(
            input_source_model, score_threshold=score_threshold, k=k, **kwargs
        )
        suggestions = []
        for node_property_desc, suggested_docs in suggestions_for_output_model.items():
            source_node_prop = node_property_desc.split(": ")[0]
            source_description = ":".join(node_property_desc.split(": ")[1:])

            source_node = source_node_prop.split(".")[0]
            source_property = ".".join(source_node_prop.split(".")[1:])

            for single_suggested_doc, similarity in suggested_docs:
                target_text = single_suggested_doc.page_content

                target_node_prop = target_text.split(": ")[0]
                target_description = ":".join(target_text.split(": ")[1:])

                target_node = target_node_prop.split(".")[0]
                target_property = ".".join(target_node_prop.split(".")[1:])

                single_suggestion = SingleHarmonizationSuggestion(
                    source_node=source_node,
                    source_property=source_property,
                    source_description=source_description,
                    target_node=target_node,
                    target_property=target_property,
                    target_description=target_description,
                    similarity=similarity,
                )
                suggestions.append(single_suggestion)

        return HarmonizationSuggestions(suggestions=suggestions)

    def _get_suggestions_for_ai_model_output(self, input_source_model, **kwargs):
        suggestions_for_output_model = {}

        for node in input_source_model.get("nodes", []):
            for node_property in node.get("properties", []):
                node_property_desc = f'{node.get("name", "unknown")}.{node_property.get("name", "unknown")}: {node_property.get("description", "")}'
                matches = get_similar_documents(
                    self.vectorstore, node_property_desc, **kwargs
                )
                if matches:
                    suggestions_for_output_model[node_property_desc] = matches

        if not suggestions_for_output_model:
            for node, node_info in input_source_model.items():
                node_properties = getattr(node_info, "properties", None) or node_info
                if type(node_properties) == list:
                    for node_property in node_properties:
                        node_property_desc = f'{node_info.get("name", "unknown")}.{node_property.get("name", "unknown")}: {node_property.get("description", "")}'
                        matches = get_similar_documents(
                            self.vectorstore, node_property_desc, **kwargs
                        )
                        if matches:
                            suggestions_for_output_model[node_property_desc] = matches
                elif type(node_properties) == dict:
                    for _, node_property in node_info.get("properties", {}).items():
                        node_property_desc = f'{node_info.get("name", "unknown")}.{node_property.get("name", "unknown")}: {node_property.get("description", "")}'
                        matches = get_similar_documents(
                            self.vectorstore, node_property_desc, **kwargs
                        )
                        if matches:
                            suggestions_for_output_model[node_property_desc] = matches
                else:
                    raise Exception(f"Cannot parse node properties")
        return suggestions_for_output_model
