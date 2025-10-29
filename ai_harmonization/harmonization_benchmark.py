import csv
import io
import json
import os
import re
from collections import Counter
from typing import Callable, Dict

from ai_harmonization.harmonization_approaches.base import (
    HarmonizationApproach,
    HarmonizationSuggestions,
    SingleHarmonizationSuggestion,
    harmonization_suggestions_to_sssom,
)
from ai_harmonization.simple_data_model import (
    SimpleDataModel,
    get_node_prop_type_desc_from_string,
)


def get_metrics_for_approach(
    benchmark_filepath: str,
    harmonization_approach: HarmonizationApproach,
    output_filename: str | None = None,
    metrics_column_name: str = "custom_metrics",
    output_sssom_per_row: bool = False,
    output_tsvs_per_row: bool = False,
    output_expected_results_per_row: bool = False,
    source_model_to_simple_model_converter_function: (
        Callable[[str], SimpleDataModel] | None
    ) = None,
    target_model_to_simple_model_converter_function: (
        Callable[[str], SimpleDataModel] | None
    ) = None,
    **kwargs,
) -> str:
    """
    Calculates and outputs metrics for a harmonization approach on a benchmark.

    This function processes a benchmark file, applies a given harmonization
    approach, and outputs various metrics, including accuracy, precision, recall,
    and F1-score. It also provides options to output SSSOM (Simple Semantic
    Similarity Output Measure) and TSV (Tab-Separated Values) files for each row
    of the benchmark.

    Args:
        benchmark_filepath: Path to the benchmark file (JSON Lines format).
        harmonization_approach: The harmonization approach to evaluate.
        output_filename: Path to the output file for metrics (JSON Lines format).
            If None, a default filename is generated based on the benchmark file.
        metrics_column_name: The name of the column to store metrics in the
            output file.
        output_sssom_per_row: Whether to output SSSOM files for each row.
        output_tsvs_per_row: Whether to output TSV files for each row.
        output_expected_results_per_row: Whether to output expected results as TSV
            files for each row.
        source_model_to_simple_model_converter_function: A function to convert the
            source model to a SimpleDataModel. If None, a default converter is used.
        target_model_to_simple_model_converter_function: A function to convert the
            target model to a SimpleDataModel. If None, a default converter is used.
        **kwargs: Additional keyword arguments to pass to the harmonization approach.

    Returns:
        The path to the output file containing the metrics.
    """
    source_model_to_simple_model_converter_function = (
        source_model_to_simple_model_converter_function
        or SimpleDataModel.get_from_unknown_json_format
    )
    target_model_to_simple_model_converter_function = (
        target_model_to_simple_model_converter_function
        or SimpleDataModel.get_from_unknown_json_format
    )

    if not output_filename:
        dir_name = os.path.dirname(benchmark_filepath)
        base_name = os.path.basename(benchmark_filepath)
        output_filename = os.path.abspath(
            os.path.join(dir_name, "metrics_" + base_name)
        )

    dir_name = os.path.dirname(output_filename)
    sssom_output_dir_name = os.path.join(dir_name, "sssom_per_row")
    tsvs_output_dir_name = os.path.join(dir_name, "sssom_clean_tsvs_per_row")
    expected_results_output_dir_name = os.path.join(
        dir_name, "expected_results_per_row"
    )

    with open(output_filename, "w") as output_file:
        with open(benchmark_filepath, "r", encoding="utf-8") as input_file:
            for i, line in enumerate(input_file):
                row = json.loads(line)
                try:
                    input_source_model = json.loads(row["input_source_model"])
                except Exception:
                    input_source_model = row["input_source_model"]

                input_source_model = source_model_to_simple_model_converter_function(
                    json.dumps(input_source_model)
                )

                try:
                    input_target_model = json.loads(row["input_target_model"])
                except Exception:
                    input_target_model = row["input_target_model"]

                input_target_model = target_model_to_simple_model_converter_function(
                    json.dumps(input_target_model)
                )

                harmonized_mapping = row["harmonized_mapping"]
                expected_harmonization_suggestions = (
                    get_harmonization_suggestions_from_harmonized_mapping(
                        harmonized_mapping
                    )
                )
                if output_expected_results_per_row:
                    filename = os.path.join(
                        expected_results_output_dir_name, f"row_{i}.sssom.tsv"
                    )
                    harmonization_suggestions_to_sssom(
                        expected_harmonization_suggestions.suggestions,
                        filename=filename,
                        exclude_required_comments=True,
                    )

                suggestions = harmonization_approach.get_harmonization_suggestions(
                    input_source_model=input_source_model,
                    input_target_model=input_target_model,
                    **kwargs,
                )

                # output standard SSSOM per row
                if output_sssom_per_row:
                    filename = os.path.join(sssom_output_dir_name, f"row_{i}.sssom.tsv")
                    harmonization_suggestions_to_sssom(
                        suggestions.suggestions, filename=filename
                    )

                if output_tsvs_per_row:
                    filename = os.path.join(tsvs_output_dir_name, f"row_{i}.sssom.tsv")
                    harmonization_suggestions_to_sssom(
                        suggestions.suggestions,
                        filename=filename,
                        exclude_required_comments=True,
                    )

                metrics = get_metrics_for_test_case(
                    suggestions, expected_harmonization_suggestions
                )
                row[metrics_column_name] = metrics
                line_to_write = json.dumps(row) + "\n"
                output_file.write(line_to_write)
    return output_filename


def get_harmonization_suggestions_from_harmonized_mapping(
    harmonized_mapping: str,
) -> HarmonizationSuggestions:
    suggestions = []

    # Use io.StringIO to treat the string as a file-like object
    tsv_file = io.StringIO(harmonized_mapping)
    reader = csv.reader(tsv_file, delimiter="\t")

    # Skip the header row
    next(reader)

    for node_prop_mapping in reader:
        source_model_node_prop_type_desc, target_model_node_prop_type_desc = (
            node_prop_mapping
        )

        source_node_name, source_prop_name, source_prop_type, source_prop_desc = (
            get_node_prop_type_desc_from_string(source_model_node_prop_type_desc)
        )
        target_node_name, target_prop_name, target_prop_type, target_prop_desc = (
            get_node_prop_type_desc_from_string(target_model_node_prop_type_desc)
        )

        source_additional_metadata = {}
        source_additional_metadata["type"] = source_prop_type

        target_additional_metadata = {}
        target_additional_metadata["type"] = target_prop_type

        single_suggestion = SingleHarmonizationSuggestion(
            source_node=source_node_name,
            source_property=source_prop_name,
            source_description=source_prop_desc,
            source_additional_metadata=source_additional_metadata,
            target_node=target_node_name,
            target_property=target_prop_name,
            target_description=target_prop_desc,
            target_additional_metadata=target_additional_metadata,
        )
        suggestions.append(single_suggestion)

    return HarmonizationSuggestions(suggestions=suggestions)


def get_metrics_for_test_case(
    suggestions: HarmonizationSuggestions, expected_mappings: HarmonizationSuggestions
) -> Dict:
    HarmonizationSuggestions.model_validate(suggestions)
    HarmonizationSuggestions.model_validate(expected_mappings)

    def to_pair(s: SingleHarmonizationSuggestion):
        return (s.source_node, s.source_property, s.target_node, s.target_property)

    suggestion_pairs = set(to_pair(s) for s in suggestions.suggestions)
    expected_pairs = set(to_pair(e) for e in expected_mappings.suggestions)

    correct_pairs = suggestion_pairs & expected_pairs
    missing_pairs = expected_pairs - suggestion_pairs
    missing_mappings = [
        f"{pair[0]}.{pair[1]} -> {pair[2]}.{pair[3]}" for pair in missing_pairs
    ]

    n_correct = len(correct_pairs)
    n_suggested = len(suggestion_pairs)
    n_expected = len(expected_pairs)

    # (node.property -> node.property) accuracy
    # e.g. recall
    accuracy = n_correct / n_expected if n_expected else 0.0

    # how many suggestions were correct
    precision = n_correct / n_suggested if n_suggested else 0.0

    # how many expected were found
    recall = n_correct / n_expected if n_expected else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "n_suggested": n_suggested,
        "n_expected": n_expected,
        "n_correct": n_correct,
        "overall_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "missing_mappings": missing_mappings,
    }
