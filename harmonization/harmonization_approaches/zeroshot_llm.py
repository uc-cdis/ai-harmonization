import logging
import os
import json
from abc import ABC

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from typing import List
from pydantic import BaseModel


from harmonization.harmonization_approaches.base import HarmonizationApproach


class Mapping(BaseModel):
    node: str
    property: str
    description: str


class ResponseSchema(BaseModel):
    response: List[Mapping]


class ZeroShotLLMHarmonization(HarmonizationApproach, ABC):

    def __init__(
        self,
        model_name: str,
        prompt_path: str,
        input_source_model: str,
        input_target_model: str,
        input_target_model_type: str = "gen3",
    ):
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.input_source_model = input_source_model
        self.input_target_model = input_target_model
        self.prompt = self.create_prompt()
        self.llm = None

    def create_prompt(self):
        """
        1. Read in prompt format from path.
        2. Replace placeholder values with the input source model which we're trying to harmonize
        3. Replace placeholder value with the input target model which is the target of harmonization
        """
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompt file not found at: {self.prompt_path}")

        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        try:
            prompt = prompt.replace(
                "input_source_model_placeholder", self.input_source_model
            )
        except:
            raise ValueError(
                "Can't input source model to prompt, input_source_model_placeholder is not in prompt"
            )

        try:
            prompt = prompt.replace(
                "input_target_model_placeholder", self.input_target_model
            )
        except:
            raise ValueError(
                "Can't input source model to prompt, input_target_model_placeholder is not in prompt"
            )

        return prompt

    def load_model(self):

        print(f"Loading model - {self.model_name} - with vLLM")
        self.llm = LLM(model=self.model_name)
        print("Model loaded succesfully")

    def run_inference(self, max_tokens=512, temperature=0.3):

        ref_schema = ResponseSchema.model_json_schema()
        decoding_params = GuidedDecodingParams(json=json.dumps(ref_schema))

        if self.llm is None:
            raise ValueError("No model has been loaded. Call 'load_model()' first.")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            guided_decoding=decoding_params,
        )

        outputs = self.llm.generate(self.prompt, sampling_params)

        return outputs[0].outputs[0].text
