import logging

import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel, Field
from transformers import AutoConfig
import torch

from langchain_huggingface import HuggingFaceEmbeddings

from harmonization.cde import (
    TEMP_DIR,
    add_documents_to_vectorstore,
    get_gen3_data_model_as_langchain_documents,
    get_langchain_vectorstore_and_persistent_client,
    get_similar_documents,
)
from harmonization.harmonization_approaches.base import (
    HarmonizationApproach,
    HarmonizationSuggestions,
    SingleHarmonizationSuggestion,
)


class ExistingVectorstoreException(BaseException):
    pass


class RetrievalAugmentedGeneration(HarmonizationApproach):

    def __init__(
        self,
        vectordb_persist_directory_name: str,
        input_target_model: dict,
        input_target_model_type: str = "gen3",
        embedding_function: HuggingFaceEmbeddings | None = None,
        force_vectorstore_recreation: bool = False,
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

        try:
            self._add_target_to_vector_database(
                input_target_model, force_recreation=force_vectorstore_recreation
            )
        except ExistingVectorstoreException:
            logging.info("vectorstore already exists, NOT recreating...")
            pass

    def _add_target_to_vector_database(
        self, input_target_model, force_recreation=False
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

        target_docs = get_gen3_data_model_as_langchain_documents(input_target_model)

        add_documents_to_vectorstore(
            target_docs, self.vectorstore, self.persistent_client
        )

    def get_harmonization_suggestions(
        self, input_source_model, input_target_model, **kwargs
    ):
        # note: k and score_threshold are in kwargs
        suggestions_for_output_model = self._get_suggestions_for_ai_model_output(
            input_source_model, **kwargs
        )
        suggestions = []
        llm_helper = LLMHelper()
        for node_property_desc, suggested_docs in suggestions_for_output_model.items():
            source_node_prop = node_property_desc.split(": ")[0]
            source_description = ":".join(node_property_desc.split(": ")[1:])

            source_node = source_node_prop.split(".")[0]
            source_property = ".".join(source_node_prop.split(".")[1:])

            sorted_suggestions_list = self._convert_documents_to_sorted_list(suggested_docs)

            llm_suggestion = llm_helper.get_decision(
                                        source_node, 
                                        source_property, 
                                        source_description, 
                                        sorted_suggestions_list
                                        )
            
            if llm_suggestion != None:
                for i, d in enumerate(sorted_suggestions_list):
                    if d["node"] == llm_suggestion["node"] and d["property"] == llm_suggestion["property"]:
                        item = sorted_suggestions_list.pop(i)   # remove from current position
                        sorted_suggestions_list.insert(0, item) # put at the front
                        break

            for i, d in enumerate(sorted_suggestions_list, start=1):  # start=1 makes ranking go from 1..n
                d["ranking"] = i

            for item in sorted_suggestions_list:
                single_suggestion = SingleHarmonizationSuggestion(
                    source_node=source_node,
                    source_property=source_property,
                    source_description=source_description,
                    target_node=item["node"],
                    target_property=item["property"],
                    target_description=item["description"],
                    similarity=item["similarity"],
                    ranking=item["ranking"]
                )
                suggestions.append(single_suggestion)

        return HarmonizationSuggestions(suggestions=suggestions)
    

    def _convert_documents_to_sorted_list(self, docs):
        data = []
        for single_doc, similarity in docs:
            text = single_doc.page_content

            node_prop = text.split(": ")[0]
            description = ":".join(text.split(": ")[1:])

            node = node_prop.split(".")[0]
            property = ".".join(node_prop.split(".")[1:])

            data.append({"node": node, "property": property, "description": description, "similarity": similarity})

        sorted_data = sorted(data, key=lambda x: x["similarity"], reverse=True)
        return sorted_data


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
    

class BestMatch(BaseModel):
        node: str = Field(..., description="The name of the node")
        property: str = Field(..., description="The name of the property")
        #description: str = Field(..., description="The description of the property")
        #confidence: float = Field(..., description="The confidence score")

#class MatchResult(BaseModel):
#    no_match: bool = Field(..., description="True if there is no match")
#    reason: str = Field(..., description="Describe the reason for the result")
#    best_match: BestMatch

class LLMHelper:
    def __init__(self):
        self.llm = self.load_llm()

    def load_llm(self):
        model_name = "meta-llama/Llama-3.1-8B-Instruct"

        model_config = AutoConfig.from_pretrained(model_name)
        num_heads = model_config.num_attention_heads
        n_gpus = torch.cuda.device_count()
        valid_divisors = [d for d in range(1, n_gpus+1) if num_heads % d == 0]
        tensor_parallel_size = max(valid_divisors)

        return LLM(
            model=model_name, 
            max_model_len=2048,
            dtype="float16",             # saves memory vs bfloat16
            max_num_seqs=1,              # how many requests at a time
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=2048
            )

    json_schema = BestMatch.model_json_schema()
    guided_decoding_params_json = GuidedDecodingParams(json=json_schema)
    sampling_params_json = SamplingParams(
        guided_decoding=guided_decoding_params_json,
        max_tokens=256,
        temperature=0.0
    )

    query_template = """
You are a helpful schema-matching assistant that outputs in JSON. Given an INPUT and a few target CANDIDATES, choose the best mapping. You have to choose one from the CANDIDATES.

INPUT:
Node: {input_node}
Property: {input_property}
Description: {input_description}

CANDIDATES:
{candidates}

{invalid_note}
"""
    

    results = []

    def get_decision(self, 
                     source_node, 
                     source_property, 
                     source_description, 
                     suggestions,
                     max_retries: int = 3):
        
        cand_strs = []
        node_property_pairs = set()
        for item in suggestions:
            node_property_pairs.add((item["node"],item["property"]))
            cand_strs.append(
                f"Node: {item["node"]}\nProperty: {item["property"]}\nDescription: {item["description"]}\nScore: {item["similarity"]:.3f}"
            )
        cand_str = "\n\n".join(cand_strs)
        invalid_answers = []

        for attempt in range(max_retries):
            invalid_note = ""
            if invalid_answers:
                invalid_note = f"Previous invalid answers: {', '.join(invalid_answers)}. Do NOT repeat them."


            query = self.__class__.query_template.format(input_node=source_node, 
                                        input_property=source_property,
                                        input_description=source_description,
                                        candidates=cand_str,
                                        invalid_note=invalid_note)
            
            outputs = self.llm.generate(query, sampling_params=self.__class__.sampling_params_json)

            decision = None
            node_property_pair = None
            try:
                decision = json.loads(outputs[0].outputs[0].text)
                node_property_pair = (decision["node"], decision["property"])
            except Exception as e:
                invalid_answers.append(outputs[0].outputs[0].text)
            
            if decision and node_property_pair:
                if node_property_pair in node_property_pairs:
                    return decision
                else:
                    invalid_answers.append(str(decision))

        return None
