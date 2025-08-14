from harmonization.harmonization_benchmark import (
    get_metrics_for_approach,
    HarmonizationApproach,
)
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from typing import List

from harmonization.harmonization_approaches.harmonization import (
    SingleHarmonizationSuggestion,
    HarmonizationSuggestions,
)


@pytest.fixture
def example_suggestions():
    return HarmonizationSuggestions(
        suggestions=[
            SingleHarmonizationSuggestion(
                source_node="Table1",
                source_property="foo",
                source_description="Foo field",
                target_node="TableA",
                target_property="bar",
                target_description="Bar field",
                similarity=0.88,
            ),
            SingleHarmonizationSuggestion(
                source_node="Table2",
                source_property="baz",
                source_description="Baz field",
                target_node="TableB",
                target_property="buzz",
                target_description="Buzz field",
                similarity=0.92,
            ),
        ]
    )


def test_get_metrics_for_approach(
    tmp_path: Path, example_suggestions: List[SingleHarmonizationSuggestion]
):
    # Create a sample benchmark file with JSONL content
    sample_row = {
        "input_source_model": """
{"a": 1}
""".strip(),
        "input_target_model": """
{"b": 2}
""".strip(),
        "harmonized_mapping": """
foo\tbar
x\ty
""",
    }
    benchmark_filepath = tmp_path / "benchmark.jsonl"
    with open(benchmark_filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_row) + "\n")

    # Prepare output file path
    output_filename = tmp_path / "metrics_benchmark.jsonl"

    # Mock HarmonizationApproach and get_metrics_for_test_case
    mock_approach = MagicMock(spec=HarmonizationApproach)
    mock_approach.get_harmonization_suggestions.return_value = example_suggestions

    with patch(
        "harmonization.harmonization_benchmark.get_metrics_for_test_case",
        return_value={"accuracy": 0.9},
    ):
        get_metrics_for_approach(
            str(benchmark_filepath),
            mock_approach,
            output_filename=str(output_filename),
            metrics_column_name="custom_metrics",
        )

    # Validate the output file content
    with open(output_filename, encoding="utf-8") as f:
        written_row = json.loads(f.readline())
        assert "custom_metrics" in written_row
        assert written_row["custom_metrics"] == {"accuracy": 0.9}
        assert json.loads(written_row["input_source_model"]) == {"a": 1}
