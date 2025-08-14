import csv
import io
import json
import os
from collections import Counter
from typing import Dict

from harmonization.harmonization_approaches.harmonization import (
    HarmonizationApproach,
    HarmonizationSuggestions,
    SingleHarmonizationSuggestion,
    harmonization_suggestions_to_sssom,
)


def get_metrics_for_approach(
    benchmark_filepath: str,
    harmonization_approach: HarmonizationApproach,
    output_filename: str | None = None,
    metrics_column_name: str = "custom_metrics",
    output_sssom_per_row: bool = True,
) -> str:
    if not output_filename:
        dir_name = os.path.dirname(benchmark_filepath)
        base_name = os.path.basename(benchmark_filepath)
        output_filename = os.path.abspath(
            os.path.join(dir_name, "metrics_" + base_name)
        )

    dir_name = os.path.dirname(output_filename)
    sssom_output_dir_name = os.path.join(dir_name, "sssom_per_row")

    with open(output_filename, "w") as output_file:
        with open(benchmark_filepath, "r", encoding="utf-8") as input_file:
            for i, line in enumerate(input_file):
                row = json.loads(line)
                input_source_model = json.loads(row["input_source_model"])
                input_target_model = json.loads(row["input_target_model"])
                harmonized_mapping = row["harmonized_mapping"]
                expected_harmonization_suggestions = (
                    get_harmonization_suggestions_from_harmonized_mapping(
                        harmonized_mapping
                    )
                )

                suggestions = harmonization_approach.get_harmonization_suggestions(
                    input_source_model=input_source_model,
                    input_target_model=input_target_model,
                )

                # output standard SSSOM per row
                if output_sssom_per_row:
                    filename = os.path.join(sssom_output_dir_name, f"row_{i}.sssom.tsv")
                    harmonization_suggestions_to_sssom(
                        suggestions.suggestions, filename=filename
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
        ai_model_node_prop_desc, harmonized_model_node_prop_desc = node_prop_mapping

        source_node_prop = ai_model_node_prop_desc.split(": ")[0]
        source_description = ":".join(ai_model_node_prop_desc.split(": ")[1:])

        source_node = source_node_prop.split(".")[0]
        source_property = ".".join(source_node_prop.split(".")[1:])

        target_node_prop = harmonized_model_node_prop_desc.split(": ")[0]
        target_description = ":".join(harmonized_model_node_prop_desc.split(": ")[1:])

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

    n_correct = len(correct_pairs)
    n_suggested = len(suggestion_pairs)
    n_expected = len(expected_pairs)

    # (node.property -> node.property) accuracy
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
    }
