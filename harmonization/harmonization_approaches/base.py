import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel
from sssom.parsers import MappingSetDataFrame
from sssom.writers import write_table
from sssom_schema import Mapping, MappingSet

from harmonization.utils import nodeproperty_to_curie

DEFAULT_SSSOM_METADATA: Dict[str, str | dict] = {
    "mapping_set_id": "https://example.org/mappings/example.sssom.tsv",
    "license": "https://www.apache.org/licenses/LICENSE-2.0",
    "mapping_set_description": (
        "This mapping set aligns properties (columns/fields) between source and target data models (graph or relational). "
        "For each mapping, subject_id and object_id are composite identifiers in the form 'node.property', "
        "where 'node' is the original entity/table/class, and 'property' is the attribute/field/column. "
        "For example, 'Subject.name' refers to the 'name' property of the 'Subject' node in the source model."
    ),
    # "prefix_map": {"EX": "http://example.org/"},
}
DEFAULT_SSSOM_PREDICATE_ID: str = "skos:relatedMatch"
DEFAULT_SSSOM_MAPPING_JUSTIFICATION: str = "semapv:UnspecifiedMatching"


class SingleHarmonizationSuggestion(BaseModel):
    """
    Represents a harmonization suggestion for a pair of node.property attributes
    """

    source_node: str
    source_property: str
    source_description: str
    target_node: str
    target_property: str
    target_description: str
    similarity: Optional[float] = None


class HarmonizationSuggestions(BaseModel):
    """
    Container for a batch of harmonization suggestions.
    """

    suggestions: List[SingleHarmonizationSuggestion]


class HarmonizationApproach(ABC):
    @abstractmethod
    def get_harmonization_suggestions(self, **kwargs) -> HarmonizationSuggestions:
        """
        Returns HarmonizationSuggestions according to the implementing algorithm.
        """
        raise NotImplementedError()


class ExampleHarmonizationApproach(HarmonizationApproach):
    """
    Dummy example that returns a static harmonization suggestion.
    """

    def get_harmonization_suggestions(
        self, input_source_model, input_target_model, **kwargs
    ) -> HarmonizationSuggestions:
        """
        Example implementation for demonstration.
        """
        suggestions: List[SingleHarmonizationSuggestion] = [
            SingleHarmonizationSuggestion(
                source_node="source_node_1",
                source_property="source_property_1",
                source_description="source_description_1",
                target_node="target_node_1",
                target_property="target_property_1",
                target_description="target_description_1",
                similarity=0.9,
            )
        ]
        return HarmonizationSuggestions(suggestions=suggestions)


def harmonization_suggestions_to_sssom(
    suggestions: List[SingleHarmonizationSuggestion],
    filename: str,
    meta: Optional[dict] = None,
) -> None:
    """
    Convert a list of harmonization suggestions into a MappingSet and write to a SSSOM TSV file.

    Args:
        suggestions: List of SingleHarmonizationSuggestion.
        filename: Output .sssom.tsv file.
        meta: Optional SSSOM MappingSet metadata dictionary. If None, uses DEFAULT_SSSOM_METADATA.
    """
    meta = meta or DEFAULT_SSSOM_METADATA
    mappings = [
        suggestion_to_mapping(single_suggestion) for single_suggestion in suggestions
    ]
    mapping_set = MappingSet(**meta, mappings=mappings)
    write_mappingset_to_sssom(mapping_set, filename)


def suggestion_to_mapping(
    suggestion: SingleHarmonizationSuggestion,
    predicate_id: Optional[str] = None,
    mapping_justification: Optional[str] = None,
) -> Mapping:
    """
    Convert a SingleHarmonizationSuggestion to an SSSOM Mapping.

    Args:
        suggestion: The harmonization suggestion to convert.
        predicate_id: Which predicate to use; default is DEFAULT_SSSOM_PREDICATE_ID.
        mapping_justification: Which mapping justification to use; default is DEFAULT_SSSOM_MAPPING_JUSTIFICATION.

    Returns:
        Mapping: The equivalent SSSOM Mapping object.
    """
    predicate_id = predicate_id or DEFAULT_SSSOM_PREDICATE_ID
    mapping_justification = mapping_justification or DEFAULT_SSSOM_MAPPING_JUSTIFICATION

    # SSSOM requires CURIEs or URLs.
    subject_id = nodeproperty_to_curie(
        f"{suggestion.source_node}.{suggestion.source_property}"
    )
    object_id = nodeproperty_to_curie(
        f"{suggestion.target_node}.{suggestion.target_property}"
    )

    return Mapping(
        predicate_id=predicate_id,
        mapping_justification=mapping_justification,
        subject_id=subject_id,
        subject_label=suggestion.source_description,
        object_id=object_id,
        object_label=suggestion.target_description,
        object_source=None,  # TODO: data dictionary. note there's also a version
        similarity_score=suggestion.similarity,
    )


def write_mappingset_to_sssom(
    mapping_set: MappingSet,
    filename: str,
) -> None:
    """
    Write a MappingSet object to a SSSOM TSV file.

    Args:
        mapping_set: The SSSOM MappingSet to write.
        filename: The path to write the .sssom.tsv file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    mapping_set_df = MappingSetDataFrame.from_mapping_set(mapping_set)
    with open(filename, "w", encoding="utf-8") as output_file:
        write_table(mapping_set_df, output_file)
