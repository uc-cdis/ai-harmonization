# AI Harmonization

This contains code and related artifacts for powering an AI-assisted data model harmonization tool. It also contains the infrastructure for abstracting approaches, benchmarking them, and generating training data.

## Usage

```
pip install ai_harmonization
```

> NOTE: Tested on Python 3.12

### **AI-Assisted Data Curation Toolkit**

- **[Demo](./jupyter/ai_assisted_data_curation.ipynb)**

### Harmonization Benchmarking and AI Training

- [Example of Evaluating a Harmonization Approach](./jupyter/harmonization_approach_evaluation.ipynb)
- [Example Creation of a Synthetic Benchmark for Evaluation](./jupyter/harmonization_synth_benchmark_creation.ipynb)
- [Example Creation of a Real Benchmark for Evaluation](./jupyter/harmonization_real_benchmark_creation.ipynb)
- [Example Creation of a Training Data for Harmonization](./jupyter/harmonization_training_data.ipynb)

## Details

Reference `ai_harmonization.harmonization_approaches.similarity_inmem.SimilaritySearchInMemoryVectorDb` to understand how to build new harmonization approaches.

Abtractions have been built to allow the evaluation of different overall approaches. There is a base class to implement for a new `HarmonizationApproach`.

There are 2 Pydantic data types defined to standardize the interface. `HarmonizationSuggestions` is:

```json
{
    "suggestions": List of `SingleHarmonizationSuggestion`
}
```

and `SingleHarmonizationSuggestion` is:

```python
class SingleHarmonizationSuggestion(BaseModel):
    source_node: str
    source_property: str
    source_description: str
    target_node: str
    target_property: str
    target_description: str
    similarity: float = None
```

This format is convertable to [A Simple Standard for Sharing Ontological Mappings (SSSOM)](https://github.com/mapping-commons/SSSOM) and happens by default when calling `get_metrics_for_approach`.

An example of how to implement a new approach. Create a new file: `harmonization/harmonization_approaches/new_approach.py`

```python
from ai_curation.harmonization_approaches.harmonization import (
    HarmonizationApproach,
    HarmonizationSuggestions,
    SingleHarmonizationSuggestion,
)

class NewApproachExample(HarmonizationApproach):

    def __init__(
        self,
    ):
        super().__init__()
        # TODO


    def get_harmonization_suggestions(
        self, input_source_model, input_target_model, **kwargs
    ):
        # TODO
        return HarmonizationSuggestions(suggestions=suggestions)
```

> Note that the current `SimilaritySearchInMemoryVectorDb` already supports providing a new embedding algorithm.

And now, if you have a benchmark and want to evaluate the new approach:

```python
from ai_harmonization.harmonization_benchmark import get_metrics_for_approach
from ai_harmonization.harmonization_approaches.new_approach import (
    NewApproachExample,
)

new_approach = NewApproachExample()

output_filename = get_metrics_for_approach(
    benchmark_filepath="path/to/benchmark.jsonl",
    harmonization_approach=new_approach,
    output_filename="output.tsv",
    metrics_column_name="custom_metrics",
)
```

## Benchmark Details

JSONL file, each row is a separate test.

Each test should have 3 keys: `input_source_model` in JSON, with desire to harmonize to  `input_target_model` in JSON, and we expect known mapping defined in `harmonized_mapping` (which is a TSV represented as a string, with 2 columns `ai_model_node_prop_desc` and `harmonized_model_node_prop_desc`).

Example harmonized mapping:

| ai_model_node_prop_desc | harmonized_model_node_prop_desc |
|---|---|
| PatientData.AgeAtDiagnosis: The age of the patient at the time of diagnosis. | standard_patient_record.age_at_diagnosis: Age in years when diagnosed with the condition. |
| PatientData.LastVisitDate: The date of the last visit to a healthcare provider. | standard_patient_record.last_visit_date: Date of most recent medical appointment or consultation. |
| OutpatSkinCheck.AnchorDateOffset: Days between the specified anchor date and the patient's last completed skin test date. | outpat_v_skin_test.DaysFromAnchorDateToEventDate: |

The output of the `get_metrics_for_approach` generates a new file with appended metrics per test case in the benchmark.

## Local Setup

Python 3.12 is recommended, newer versions may also work.

This uses [uv](https://docs.astral.sh/uv/) (faster than `pip` and `poetry`, easier to work with).

```
uv sync
uv pip install -e .
```

### Tests

```
uv run pytest tests/
```

> Note: Tests are fairly minimal at the moment

### Other

If for some reason you need to run detect-secrets separately from pre-commit and you have a ton of data or datasets, you can explicitly exclude those directories (they should already ignored by git):

```
detect-secrets scan --exclude-files '.*/data/.*|.*/datasets/.*|.*/output/.*'
```
