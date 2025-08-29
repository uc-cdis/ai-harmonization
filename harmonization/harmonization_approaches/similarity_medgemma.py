import logging

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

from langchain_huggingface import HuggingFaceEmbeddings

from harmonization.harmonization_approaches.base import (
    HarmonizationApproach,
    HarmonizationSuggestions,
    SingleHarmonizationSuggestion,
)


class SimilaritySearchInMedgemma(HarmonizationApproach):

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-pt",
        target_model_type: str = "gen3",
        embedding_function: HuggingFaceEmbeddings = None,
    ):
        super().__init__()

        if target_model_type == "gen3":
            logging.info(f"Treating input target as type: 'gen3'")
        else:
            raise NotImplementedError(
                f"input_target_model_type of '{input_target_model_type}' is not supported at this time."
            )

        self.embedding_function = embedding_function or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map="auto"
        )
        self.model = model.eval()
        self.calculate_cosine_similarity = torch.nn.CosineSimilarity(dim=0)

    def get_harmonization_suggestions(
        self, input_source_model, input_target_model, score_threshold=0.5, k=4, **kwargs
    ):
        suggestions_for_output_model = self._get_suggestions_for_ai_model_output(
            input_source_model,
            input_target_model,
            score_threshold=score_threshold,
            k=k,
            **kwargs,
        )
        # WIP from here
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
                )
                suggestions.append(single_suggestion)
        return HarmonizationSuggestions(suggestions=suggestions)

    def _get_node_property_desc_list(model):
        node_property_desc_list = []
        for node, node_info in model.items():
            node_properties = getattr(node_info, "properties", None) or node_info
            if type(node_properties) == list:
                node_property_desc = f'{node_info.get("name", "unknown")}.{node_property.get("name", "unknown")}: {node_property.get("description", "")}'
                node_property_desc_list.extend(node_property_desc)
            elif type(node_properties) == dict:
                for _, node_property in node_info.get("properties", {}).items():
                    node_property_desc = f'{node_info.get("name", "unknown")}.{node_property.get("name", "unknown")}: {node_property.get("description", "")}'
                    node_property_desc_list.extend(node_property_desc)
            else:
                raise Exception(f"Cannot parse node properties. Node failed:/n{node}")
        return node_property_desc_list

    def _get_cls_embedding(text):
        text_inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(model.device)
        with torch.no_grad():
            text_outputs = model(**text_inputs, output_hidden_states=True)
            hidden_states = text_outputs.hidden_states
            final_hidden_state = hidden_states[-1]
            cls_embedding = final_hidden_state.mean(dim=1).squeeze()
            print(f"Embedding size: {cls_embedding.shape}")
            return cls_embedding

    def _get_suggestions_for_ai_model_output(
        self, source_model, target_model, score_threshold=0.5, **kwargs
    ):
        suggestions_for_output_model = {}
        source_node_property_desc_list = self._get_node_property_desc_list(source_model)
        target_node_property_desc_list = self._get_node_property_desc_list(target_model)
        with torch.no_grad():
            for source_node_property_desc in source_node_property_desc_list:
                suggestions_for_output_model[source_node_property_desc] = []
                source_cls_embedding = self._get_cls_embedding(
                    source_node_property_desc
                )
                for target_node_property_desc in target_node_property_desc_list:
                    target_cls_embedding = self._get_cls_embedding(
                        target_node_property_desc
                    )
                    cosine_similarity = self.calculate_cosine_similarity(
                        source_cls_embedding, target_cls_embedding
                    )
                    if cosine_similarity >= score_threshold:
                        # Should we append similarity score as well?
                        suggestions_for_output_model[source_node_property_desc].append(
                            target_node_property_desc
                        )
        return suggestions_for_output_model
