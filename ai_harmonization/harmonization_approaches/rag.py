from harmonization.harmonization_approaches.base import (
    HarmonizationApproach,
    HarmonizationSuggestions,
    SingleHarmonizationSuggestion,
)

from harmonization.harmonization_approaches.similarity_inmem import (
    SimilaritySearchInMemoryVectorDb,
)

from harmonization.harmonization_approaches.llm import (
    LLMClient,
)

from harmonization.simple_data_model import (
    SimpleDataModel,
)

import pandas as pd 
import logging

class RetrievalAugmentedGeneration(HarmonizationApproach):

    def __init__(
        self,
        similarity_search_approach_class = None,
        llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        llm_max_model_len: int = 4096,
        llm_dtype: str = "bfloat16",
        llm_max_num_seqs: int = 1,
        llm_max_num_batched_tokens: int = 2048,
        llm_gpu_memory_utilization: float = 0.9,
        llm_lora_model_path: str | None = None,
        inference_max_tokens: int = 4096,
        inference_temperature: float = 0.0,
        inference_max_retries: int = 3,
        inference_top_k: int = 5,
        max_suggestions_num: int = 10,
        query_template: str | None = None,
    ):
        super().__init__()
        self.similarity_search_approach_class = similarity_search_approach_class
        self.llm_model_name = llm_model_name
        self.llm_max_model_len = llm_max_model_len
        self.llm_dtype = llm_dtype
        self.llm_max_num_seqs = llm_max_num_seqs
        self.llm_max_num_batched_tokens = llm_max_num_batched_tokens
        self.llm_gpu_memory_utilization = llm_gpu_memory_utilization
        self.llm_lora_model_path = llm_lora_model_path
        self.inference_max_tokens = inference_max_tokens
        self.inference_temperature = inference_temperature
        self.inference_max_retries = inference_max_retries
        self.inference_top_k = inference_top_k
        self.max_suggestions_num = max_suggestions_num
        self.query_template = query_template


    def get_harmonization_suggestions(
        self,
        input_source_model: SimpleDataModel,
        input_target_model: SimpleDataModel,
        **kwargs
    ):
        similarity_search_suggestions = self.similarity_search_approach_class.get_harmonization_suggestions(
            input_source_model=input_source_model,
            input_target_model=input_target_model,
            **kwargs
        )

        similarity_search_output_df = similarity_search_suggestions.to_dataframe()
        
        llm_client = LLMClient(
            model_name=self.llm_model_name,
            max_model_len=self.llm_max_model_len,
            dtype=self.llm_dtype,
            max_num_seqs=self.llm_max_num_seqs,
            max_num_batched_tokens=self.llm_max_num_batched_tokens,
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
            inference_max_tokens=self.inference_max_tokens,
            inference_temperature=self.inference_temperature,
            inference_max_retries=self.inference_max_retries,
            inference_top_k=self.inference_top_k,
            lora_model_path=self.llm_lora_model_path,
            query_template=self.query_template,
        )

        suggestions = []
        for (source_node, source_property), group_df in similarity_search_output_df.groupby(["source_node", "source_property"]):
            outputs = llm_client.inference(group_df)
            if outputs is not None:
                # 1. Mark rows that are in outputs
                group_df["in_outputs"] = group_df.apply(
                    lambda row: (row["target_node"], row["target_property"]) in outputs, axis=1
                )
                logging.debug(f"group_df: {group_df[["target_node", "target_property", "similarity", "ranking", "in_outputs"]]}")
                n = len(outputs)
                # 2. Rank entries in outputs by similarity (descending)
                in_outputs_df = group_df[group_df["in_outputs"]].sort_values("similarity", ascending=False).copy()
                in_outputs_df["ranking"] = range(1, n+1)
                logging.debug(f"in_outputs_df: {in_outputs_df[["target_node", "target_property", "similarity", "ranking", "in_outputs"]]}")

                # 3. Rank the rest by similarity (descending), starting at n+1
                rest_df = group_df[~group_df["in_outputs"]].sort_values("similarity", ascending=False).copy()
                rest_df["ranking"] = range(n+1, n+len(rest_df)+1)
                logging.debug(f"rest_df: {rest_df[["target_node", "target_property", "similarity", "ranking", "in_outputs"]]}")

                # 4. Concatenate and restore original DataFrame
                final_df = pd.concat([in_outputs_df, rest_df])
                group_df.update(final_df)
                group_df = group_df.drop("in_outputs", axis=1)
                logging.debug(f"final group_df: {group_df[["target_node", "target_property", "similarity", "ranking"]]}")
            else:
                group_df["ranking"] = (
                    group_df["similarity"]
                    .rank(method="first", ascending=False)
                    .astype(int)
                )
            suggestions_df = group_df[group_df["ranking"] <= self.max_suggestions_num]
            logging.debug(f"suggestions_df: {suggestions_df[["target_node", "target_property", "similarity", "ranking"]]}")

            for index, row in suggestions_df.iterrows():
                single_suggestion = SingleHarmonizationSuggestion(
                    source_node=row["source_node"],
                    source_property=row["source_property"],
                    source_description=row["source_description"],
                    source_additional_metadata=row["source_additional_metadata"],
                    target_node=row["target_node"],
                    target_property=row["target_property"],
                    target_description=row["target_description"],
                    target_additional_metadata=row["target_additional_metadata"],
                    similarity=row["similarity"],
                    ranking=row["ranking"],
                )
                suggestions.append(single_suggestion)

        return HarmonizationSuggestions(suggestions=suggestions)
