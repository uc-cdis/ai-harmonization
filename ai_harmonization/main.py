import json
import os
import time

from ai_harmonization.harmonization_approaches.similarity_inmem import (
    SimilaritySearchInMemoryVectorDb,
)
from ai_harmonization.harmonization_benchmark import get_metrics_for_approach
from ai_harmonization.jsonl import (
    jsonl_to_csv,
    split_harmonization_jsonl_by_input_target_model,
)
