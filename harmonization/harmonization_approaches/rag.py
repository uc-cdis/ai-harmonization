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

class RetrievalAugmentedGeneration(HarmonizationApproach):

    def __init__(
        self,
        similarity_search_approach_class = None,
        llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        llm_max_model_len: int = 2048,
        llm_dtype: str = "float16",
        llm_max_num_seqs: int = 1,
        llm_max_num_batched_tokens: int = 2048,
        llm_gpu_memory_utilization: float = 0.9,
        llm_lora_model_path: str | None = None,
        inference_max_tokens: int = 256,
        inference_temperature: float = 0.0,
        inference_max_retries: int = 3,
        max_suggestions_num: int = 5,
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
            lora_model_path=self.llm_lora_model_path,
            query_template=self.query_template,
        )

        suggestions = []
        for (source_node, source_property), group_df in similarity_search_output_df.groupby(["source_node", "source_property"]):
            decision = llm_client.get_decision(group_df)
            if decision is not None:
                group_df.loc[(group_df["target_node"] == decision["node"]) & (group_df["target_property"] == decision["property"]), 'ranking'] = 1
                mask = group_df["ranking"] != 1  
                start_rank = 1
                group_df.loc[mask, "ranking"] = (
                    group_df.loc[mask, "similarity"]
                    .rank(method="first", ascending=False)
                    .astype(int) + start_rank
                )
            else:
                group_df["ranking"] = (
                    group_df["similarity"]
                    .rank(method="first", ascending=False)
                    .astype(int)
                )
            suggestions_df = group_df[group_df["ranking"] <= self.max_suggestions_num]

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
