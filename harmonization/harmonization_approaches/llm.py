from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.lora.request import LoRARequest
from pydantic import BaseModel, Field
from transformers import AutoConfig
import torch
import json
import logging


DEFAULT_QUERY_TEMPLATE = """
You are a helpful schema-matching assistant that outputs in JSON. Given an INPUT and some target CANDIDATES, choose the best mapping. You have to choose one from the CANDIDATES.

INPUT:
Node: {input_node}
Property: {input_property}
Description: {input_description}

CANDIDATES:
{candidates}

{invalid_note}
"""


class BestMatch(BaseModel):
    node: str = Field(..., description="The name of the node")
    property: str = Field(..., description="The name of the property")


class LLMClient:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_model_len: int = 4096,
        dtype: str = "float16",
        max_num_seqs: int = 1,
        max_num_batched_tokens: int = 2048,
        gpu_memory_utilization: float = 0.9,
        inference_max_tokens: int = 4096,
        inference_temperature: float = 0.0,
        inference_max_retries: int = 3,
        lora_model_path: str | None = None,
        query_template: str | None = None,
    ):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization

        if lora_model_path:
            logging.debug(f"lora_model_path: {lora_model_path}")
            self.enable_lora = True
            # only handle one lora model now
            self.lora_request = LoRARequest("default_name", 1, lora_model_path)
        else:
            self.enable_lora = False
            self.lora_request = None

        self.llm = self.load_llm()

        json_schema = BestMatch.model_json_schema()
        guided_decoding_params_json = GuidedDecodingParams(json=json_schema)
        self.sampling_params_json = SamplingParams(
            guided_decoding=guided_decoding_params_json,
            max_tokens=inference_max_tokens,
            temperature=inference_temperature,
        )

        self.inference_max_retries = inference_max_retries

        if query_template:
            self.query_template = query_template
        else:
            self.query_template = DEFAULT_QUERY_TEMPLATE


    def load_llm(self):
        if torch.cuda.is_available():
            model_config = AutoConfig.from_pretrained(self.model_name)
            num_heads = model_config.num_attention_heads
            n_gpus = torch.cuda.device_count()
            valid_divisors = [d for d in range(1, n_gpus+1) if num_heads % d == 0]
            tensor_parallel_size = max(valid_divisors)
        else:
            tensor_parallel_size = 1

        return LLM(
            model=self.model_name, 
            max_model_len=self.max_model_len,
            dtype=self.dtype,             
            max_num_seqs=self.max_num_seqs,              # how many requests at a time
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=self.max_num_batched_tokens,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_lora=self.enable_lora,
            )
    

    def get_decision(self, group_df):

        source_node = group_df["source_node"].iloc[0]
        source_property = group_df["source_property"].iloc[0]
        source_description = group_df["source_description"].iloc[0]

        cand_strs = []
        node_property_pairs = set()
        for index, row in group_df.iterrows():
            node_property_pairs.add((row["target_node"], row["target_property"]))
            cand_strs.append(
                f"Node: {row["target_node"]}\nProperty: {row["target_property"]}\nDescription: {row["target_description"]}"
            )
        cand_str = "\n\n".join(cand_strs)
        invalid_answers = []
        
        for attempt in range(self.inference_max_retries):
            invalid_note = ""
            if invalid_answers:
                invalid_note = f"Previous invalid answers: {', '.join(invalid_answers)}. Do NOT repeat them."

            query = self.query_template.format(
                input_node=source_node, 
                input_property=source_property,
                input_description=source_description,
                candidates=cand_str,
                invalid_note=invalid_note
            )
            
            outputs = self.llm.generate(
                query,
                sampling_params=self.sampling_params_json,
                lora_request=self.lora_request,
            )

            decision = None
            node_property_pair = None
            try:
                decision = json.loads(outputs[0].outputs[0].text)
                node_property_pair = (decision["node"], decision["property"])
            except Exception as e:
                logging.debug(
                    f"Invalid LLM output(invalid json): {outputs[0].outputs[0].text}. " 
                    f"Process detail: mapping {source_node},{source_property} to {node_property_pairs}"
                )
                invalid_answers.append(outputs[0].outputs[0].text)
            
            if decision and node_property_pair:
                if node_property_pair in node_property_pairs:
                    return decision
                else:
                    logging.debug(
                        f"Invalid LLM output(invalid answer): {decision}. " 
                        f"Process detail: mapping {source_node},{source_property} to {node_property_pairs}"
                    )
                    invalid_answers.append(str(decision))
        
        return None
