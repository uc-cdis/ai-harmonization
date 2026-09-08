"""
dbGaP data dictionary parsing and mapping output utilities.

Parses dbGaP data_dict.xml and var_report.xml files into SimpleDataModel objects
for use as source variables in harmonization. Also provides helpers for finding
study files and writing output CSV rows.
"""

import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from ai_harmonization.simple_data_model import SimpleDataModel, Node, Property
from ai_harmonization.styles import STRONG_SIMILARITY, VERY_STRONG_SIMILARITY


DBGAP_TYPE_MAP = {
    "integer": "integer",
    "decimal": "float",
    "float": "float",
    "num": "float",
    "numeric": "float",
    "string": "string",
    "text": "string",
    "encoded value": "string/encoded",
    "date": "datetime",
    "datetime": "datetime",
}

CSV_HEADERS = [
    "Original Node.Property",
    "Suggested Target Node.Property",
    "Similarity",
    "Target Description",
    "Target Values",
    "Original Description",
    "Original Values",
    "study_id",
    "source_table_id",
    "source_variable_name",
    "prompt_variant",
    "rank",
]


def parse_dbgap_table(dict_path, report_path=None):
    """Parse a dbGaP data_dict XML (+ optional var_report XML) into a SimpleDataModel.

    Args:
        dict_path (str): Path to the *_data_dict.xml file.
        report_path (str | None): Path to the matching *_var_report.xml file.
            Used to resolve property types; skipped if None or missing.

    Returns:
        tuple[str, SimpleDataModel]: (table_id, model).
            Property.values contains value meanings (for embedding prompts);
            Property.additional_metadata['value_labels'] contains code=meaning
            pairs (for CSV display).
    """
    root = ET.parse(dict_path).getroot()
    table_id = root.attrib.get("id", os.path.basename(dict_path))

    var_types = {}
    if report_path and os.path.exists(report_path):
        try:
            report_root = ET.parse(report_path).getroot()
            for var_node in report_root.findall(".//variable"):
                v_name = var_node.findtext("name")
                v_type_node = var_node.find(".//type")
                if v_name and v_type_node is not None and v_type_node.text:
                    var_types[v_name.strip()] = DBGAP_TYPE_MAP.get(
                        v_type_node.text.strip().lower(), "string"
                    )
        except Exception as e:
            print(
                f"Warning: could not parse var report {os.path.basename(report_path)}: {e}"
            )

    properties = []
    for var in root.findall(".//variable"):
        var_name = (var.findtext("name") or "").strip()
        if not var_name:
            continue

        value_meanings, value_labels = [], []
        for val in var.findall("value")[:5]:
            if not val.text:
                continue
            meaning = val.text.strip()
            value_meanings.append(meaning)
            code = val.attrib.get("code")
            value_labels.append(f"{code}={meaning}" if code else meaning)

        properties.append(
            Property(
                name=var_name,
                description=(
                    var.findtext("description") or "No description available."
                ).strip(),
                type=var_types.get(var_name, "string"),
                values=value_meanings or None,
                additional_metadata=(
                    {"value_labels": value_labels} if value_labels else None
                ),
            )
        )

    return table_id, SimpleDataModel(
        nodes=[
            Node(
                name=table_id,
                description="",
                links=[],
                properties=properties,
            )
        ]
    )


def find_study_metadata_files(studies_dir, study_id):
    """Return (study_files_path, dict_files, var_report_by_pht) for a study.

    Args:
        studies_dir (str): Root directory holding one subdirectory per study.
        study_id (str): e.g. 'phs000557.v7.p2.c1'

    Returns:
        tuple[str, list[str], dict[str, str]]: The study's metadata directory,
            the *_data_dict.xml filenames in it, and a map from pht accession
            to the full path of the matching *_var_report.xml.

    Raises:
        FileNotFoundError: If the study's metadata directory does not exist.
    """
    study_files_path = os.path.join(studies_dir, study_id, "metadata")
    if not os.path.exists(study_files_path):
        raise FileNotFoundError(
            f"Path '{study_files_path}' not found. "
            "Run download_studies_data_dictionaries.ipynb first."
        )

    dict_files = [
        f
        for f in os.listdir(study_files_path)
        if f.lower().endswith(".xml") and "data_dict" in f.lower()
    ]

    var_report_by_pht = {}
    for f in os.listdir(study_files_path):
        if f.lower().endswith(".xml") and "var_report" in f.lower():
            pht_match = re.search(r"pht\d+", f)
            if pht_match:
                var_report_by_pht[pht_match.group()] = os.path.join(study_files_path, f)

    return study_files_path, dict_files, var_report_by_pht


def build_mapping_rows(suggestions, slot_values_lookup, study_id):
    """Convert one variable's harmonization suggestions into CSV row dicts.

    Args:
        suggestions (list[SingleHarmonizationSuggestion]): Suggestions for a
            single source variable, best first — one variable's worth of output
            from MultiPromptSimilaritySearch.
        slot_values_lookup (dict[str, str]): Maps target slot keys to
            comma-joined enum values.
        study_id (str): Study identifier to embed in every row.

    Returns:
        list[dict]: One dict per suggestion, keyed by CSV_HEADERS columns.
            ``rank`` is 1-based in descending similarity order.
    """
    rows = []
    for rank, suggestion in enumerate(suggestions, start=1):
        slot_key = f"{suggestion.target_node}.{suggestion.target_property}"
        value_labels = (suggestion.source_additional_metadata or {}).get(
            "value_labels", []
        )
        rows.append(
            {
                "Original Node.Property": f"{suggestion.source_node}.{suggestion.source_property}",
                "Suggested Target Node.Property": slot_key,
                "Similarity": round(suggestion.similarity, 4),
                "Target Description": suggestion.target_description,
                "Target Values": slot_values_lookup.get(slot_key, ""),
                "Original Description": suggestion.source_description,
                "Original Values": ", ".join(value_labels),
                "study_id": study_id,
                "source_table_id": suggestion.source_node,
                "source_variable_name": suggestion.source_property,
                "prompt_variant": (suggestion.target_additional_metadata or {}).get(
                    "prompt_variant", ""
                ),
                "rank": rank,
            }
        )
    return rows


def generate_variable_mappings(
    study_files_path,
    dict_files,
    var_report_by_pht,
    harmonization_approach,
    slot_values_lookup,
    study_id,
    k,
):
    """Generate mapping row dicts for every variable across all tables in a study.

    Yields one variable at a time so callers can write results incrementally
    rather than holding a whole study's mappings in memory.

    Args:
        study_files_path (str): Directory containing the XML files.
        dict_files (list[str]): Filenames of *_data_dict.xml files to process.
        var_report_by_pht (dict[str, str]): Maps pht accession to full var_report path.
        harmonization_approach (MultiPromptSimilaritySearch): Search index with
            the target schema already embedded.
        slot_values_lookup (dict[str, str]): Maps slot keys to comma-joined enum values.
        study_id (str): Study identifier embedded in every CSV row.
        k (int): Number of top suggestions to return per variable.

    Yields:
        list[dict]: CSV row dicts for one source variable (up to k rows).
    """
    for dict_filename in dict_files:
        full_dict_path = os.path.join(study_files_path, dict_filename)
        pht_match = re.search(r"pht\d+", dict_filename)
        full_report_path = (
            var_report_by_pht.get(pht_match.group()) if pht_match else None
        )

        try:
            _, source_model = parse_dbgap_table(full_dict_path, full_report_path)
        except Exception as e:
            print(f"Warning: could not parse {dict_filename}: {e}")
            continue

        for suggestions in harmonization_approach.iter_suggestions_by_property(
            source_model, k=k
        ):
            yield build_mapping_rows(suggestions, slot_values_lookup, study_id)


def summarize_rank1_similarity(rank1_group):
    """Compute mapping quality statistics for one study from rank-1 suggestions.

    Args:
        rank1_group (pd.DataFrame): Rows for a single study_id where rank == 1.

    Returns:
        pd.Series: Statistics including variable count, mean/median similarity,
            strong-match percentages, and the top bdchm target slot.
    """
    sim = rank1_group["Similarity"]
    strong, very_strong = STRONG_SIMILARITY, VERY_STRONG_SIMILARITY
    return pd.Series(
        {
            "Variables": len(rank1_group),
            "Mean sim (rank 1)": round(sim.mean(), 3),
            "Median sim": round(sim.median(), 3),
            f"≥{strong:.2f} (strong)": f"{(sim >= strong).mean():.0%}",
            f"≥{very_strong:.2f} (very strong)": f"{(sim >= very_strong).mean():.0%}",
            "Top bdchm target": rank1_group.nlargest(1, "Similarity")[
                "Suggested Target Node.Property"
            ].iloc[0],
        }
    )
