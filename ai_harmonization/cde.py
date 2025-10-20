import json
import logging
import os
from typing import List, Union

import requests
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

from ai_harmonization.utils import TEMP_DIR

"""
Examples:

```json
{
  "nodes": [
    {
      "name": "read_group_qc_metadata",
      "description": "Metadata related to read groups and quality control metrics.",
      "links": ["foobar"],
      "properties": [
        {
          "description": "The unique identifier for the QC metadata of a read group.",
          "type": "identifier",
          "name": "read_group_qc_id"
        },
        ...
      ]
    },
    ...
  ]
}
```

```json

{
  "nodes": [
    {
      "name": "family history",
      "description": "A family history of cancer",
      "links": [
        "subjects"
      ],
      "properties": [
        {
          "description": "The subject's ID",
          "type": "string",
          "propertyName": "subjects_id"
        },
        ...
      ]
    },
    ...
```

"""


def get_cdes_as_langchain_documents() -> List[Document]:
    """
    Retrieves and converts CDEs to LangChain documents.

    Returns:
        List[Document]: A list of Document objects representing the converted LangChain documents.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    input_file_path = _get_cdes_and_write_to_file()
    filtered_file_path = _convert_cdes_to_jsonl_and_filter(input_file_path)
    documents = _convert_filtered_cdes_to_langchain_documents(filtered_file_path)
    return documents


def _get_cdes_and_write_to_file() -> str:
    """
    Bulk export CDEs from NIH server
    """
    url = "https://cde.nlm.nih.gov/server/de/searchExport"
    data = {
        "resultPerPage": 99999,
        "excludeAllOrgs": False,
        "excludeOrgs": [],
        "includeRetired": False,
        "selectedElements": [],
        "selectedElementsAlt": [],
        "page": 1,
        "includeAggregations": True,
        "selectedAdminStatuses": [],
        "selectedStatuses": [],
        "selectedDatatypes": [],
        "selectedCopyrightStatus": [],
        "searchToken": "",
        "nihEndorsed": False,
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
        stream=True,
    )

    filename = f"{TEMP_DIR}/SearchExport.json"
    with open(filename, "wb") as file:
        for chunk in response.iter_content(1024):
            file.write(chunk)

    logging.info(f"Wrote raw CDEs to {filename}")
    return filename


def _convert_cdes_to_jsonl_and_filter(input_file_path: str) -> str:
    """
    Converts a JSONL file to a filtered JSONL file by selecting only
    relevant metadata for the CDEs.

    Args:
        input_file_path: The path to the input JSONL file.

    Returns:
        str: The path to the filtered JSONL file.
    """
    output_file_path = os.path.splitext(input_file_path)[0] + ".jsonl"
    filtered_file_path = os.path.splitext(input_file_path)[0] + "_filtered.jsonl"

    with (
        open(input_file_path, "r") as input_file,
        open(output_file_path, "w") as output_file,
    ):
        raw_input = json.load(input_file)
        for item in raw_input:
            output_file.write(json.dumps(item) + "\n")

    # This extracts and filters metadata from the record
    with (
        open(output_file_path) as jsonl_file,
        open(filtered_file_path, "w") as filtered_output,
    ):
        for row in jsonl_file:
            item = json.loads(row)
            info = {
                "tinyID": item["tinyId"],
                "names": [item["designation"] for item in item["designations"]],
                "dataElementConcept": item["dataElementConcept"],
                "steward": item["steward"],
                "definitions": [item["definition"] for item in item["definitions"]],
                "properties": [item["value"] for item in item["properties"]],
                "elementType": item["elementType"],
                "ids": [item["id"] for item in item["ids"]],
                "nihEndorsed": item["nihEndorsed"],
                "valueDomain": item["valueDomain"],
            }
            filtered_output.write(json.dumps(info) + "\n")

    logging.info(f"Wrote filtered CDEs to {filtered_file_path}")
    return filtered_file_path


def _convert_filtered_cdes_to_langchain_documents(
    filtered_file_path: str, metadata_func: Union[callable, None] = None
) -> List[Document]:
    """
    Converts a filtered CDEs JSON file to a list of Langchain documents.

    Args:
        filtered_file_path: The path to the filtered CDEs JSON file.

    Returns:
        List[Document]: A list of Langchain Document objects.
    """
    metadata_func = metadata_func or _metadata_func
    loader = JSONLoader(
        file_path=filtered_file_path,
        jq_schema=".",
        text_content=False,
        json_lines=True,
        content_key="names",
        metadata_func=metadata_func,
    )

    documents = loader.load()

    return documents


def _metadata_func(record: dict, metadata: dict) -> dict:
    """This converts non-strings to string so all this can be tokenized"""
    metadata["names"] = str(record.get("names", []))
    metadata["dataElementConcept"] = str(record.get("dataElementConcept"))
    metadata["steward"] = str(record.get("steward"))
    metadata["definitions"] = str(record.get("definitions", []))
    metadata["properties"] = str(record.get("properties", []))
    metadata["ids"] = str(record.get("ids", []))
    metadata["nihEndorsed"] = record.get("nihEndorsed", False)
    metadata["valueDomain"] = str(record.get("valueDomain", []))

    metadata["source"] = f"https://cde.nlm.nih.gov/deView?tinyId={record['tinyID']}"

    return metadata


EXAMPLE_AI_MODEL_OUTPUT_JSON = json.loads(
    """
{
  "nodes": [
    {
      "name": "submitted_aligned_reads_file_manifest",
      "description": "Manifest of submitted aligned reads files",
      "links": [
        "projects_id"
      ]
    ,
      "properties": [
        {
          "description": "Unique identifier for the submitted aligned reads file",
          "type": "integer",
          "name": "submitted_aligned_reads_id"
        }
      ,
        {
          "description": "Name of the submitted aligned reads file",
          "type": "string",
          "name": "file_name"
        }
      ,
        {
          "description": "Format of the submitted aligned reads file",
          "type": "string",
          "name": "data_format"
        }
      ,
        {
          "description": "Category of the submitted aligned reads file",
          "type": "string",
          "name": "data_category"
        }
      ,
        {
          "description": "Type of the submitted aligned reads file",
          "type": "string",
          "name": "data_type"
        }
      ,
        {
          "description": "Experimental strategy used for the submitted aligned reads file",
          "type": "string",
          "name": "experimental_strategy"
        }
      ,
        {
          "description": "Consent codes associated with the submitted aligned reads file",
          "type": "string",
          "name": "consent_codes"
        }
      ,
        {
          "description": "Projects ID associated with the submitted aligned reads file",
          "type": "integer",
          "name": "projects_id"
        }
      ,
        {
          "description": "Cases ID associated with the submitted aligned reads file",
          "type": "integer",
          "name": "cases_id"
        }
      ,
        {
          "description": "Samples ID associated with the submitted aligned reads file",
          "type": "integer",
          "name": "samples_id"
        }
      ,
        {
          "description": "Aliquots ID associated with the submitted aligned reads file",
          "type": "integer",
          "name": "aliquots_id"
        }
      ,
        {
          "description": "Read groups ID associated with the submitted aligned reads file",
          "type": "integer",
          "name": "read_groups_id"
        }
      ]
    }
  ,
    {
      "name": "aligned_reads_index_file_manifest",
      "description": "Manifest of aligned reads index files",
      "links": [
        "submitted_aligned_reads_files_id"
      ]
    ,
      "properties": [
        {
          "description": "Unique identifier for the aligned reads index file",
          "type": "integer",
          "name": "aligned_reads_index_id"
        }
      ,
        {
          "description": "Category of the aligned reads index file",
          "type": "string",
          "name": "data_category"
        }
      ,
        {
          "description": "Type of the aligned reads index file",
          "type": "string",
          "name": "data_type"
        }
      ,
        {
          "description": "Format of the aligned reads index file",
          "type": "string",
          "name": "data_format"
        }
      ,
        {
          "description": "ID of the submitted aligned reads file associated with this index file",
          "type": "integer",
          "name": "submitted_aligned_reads_files_id"
        }
      ]
    }
  ,
    {
      "name": "read_group_metadata",
      "description": "Metadata of read groups",
      "links": [
        "projects_id",
        "cases_id",
        "samples_id",
        "aliquots_id"
      ]
    ,
      "properties": [
        {
          "description": "ID of the read group",
          "type": "integer",
          "name": "read_group_id"
        }
      ,
        {
          "description": "Age at diagnosis",
          "type": "integer",
          "name": "age_at_diagnosis"
        }
      ,
        {
          "description": "Age at index",
          "type": "integer",
          "name": "age_at_index"
        }
      ,
        {
          "description": "Aliquot description",
          "type": "string",
          "name": "aliquot_description"
        }
      ,
        {
          "description": "Amount of the sample",
          "type": "float",
          "name": "amount"
        }
      ,
        {
          "description": "Date of analyte isolation",
          "type": "date",
          "name": "analyte_isolation_date"
        }
      ,
        {
          "description": "Quantity of the analyte",
          "type": "float",
          "name": "analyte_quantity"
        }
      ,
        {
          "description": "Type of the analyte",
          "type": "string",
          "name": "analyte_type_id"
        }
      ,
        {
          "description": "Physical site of biospecimen",
          "type": "string",
          "name": "biospecimen_physical_site"
        }
      ,
        {
          "description": "Birth year",
          "type": "integer",
          "name": "birth_year"
        }
      ,
        {
          "description": "Other composition of the sample",
          "type": "string",
          "name": "composition_other"
        }
      ,
        {
          "description": "Method of contrivance",
          "type": "string",
          "name": "contrivance_method"
        }
      ,
        {
          "description": "County (long)",
          "type": "string",
          "name": "county_long"
        }
      ,
        {
          "description": "Cycles",
          "type": "integer",
          "name": "cycles"
        }
      ,
        {
          "description": "Date of collection",
          "type": "date",
          "name": "date_collected"
        }
      ,
        {
          "description": "Days to assay",
          "type": "integer",
          "name": "days_to_assay"
        }
      ,
        {
          "description": "Days to lost to followup",
          "type": "integer",
          "name": "days_to_lost_to_followup"
        }
      ,
        {
          "description": "Mean fragment length",
          "type": "float",
          "name": "fragment_mean_length"
        }
      ,
        {
          "description": "Minimum fragment length",
          "type": "float",
          "name": "fragment_minimum_length"
        }
      ,
        {
          "description": "ID of the GDC sample",
          "type": "string",
          "name": "gdc_sample_id"
        }
      ,
        {
          "description": "GenBank accession",
          "type": "string",
          "name": "genbank_accession"
        }
      ,
        {
          "description": "Host name or IP address of the submission tool",
          "type": "string",
          "name": "host_name"
        }
      ,
        {
          "description": "ID of the host",
          "type": "integer",
          "name": "id"
        }
      ,
        {
          "description": "Institution name",
          "type": "string",
          "name": "institution"
        }
      ,
        {
          "description": "International classification of diseases code",
          "type": "string",
          "name": "icd_code"
        }
      ,
        {
          "description": "ID of the project",
          "type": "integer",
          "name": "id"
        }
      ,
        {
          "description": "ID of the primary accession",
          "type": "string",
          "name": "id"
        }
      ,
        {
          "description": "Index (0-based)",
          "type": "integer",
          "name": "index"
        }
      ,
        {
          "description": "Illumina platform",
          "type": "string",
          "name": "illumina_platform"
        }
      ,
        {
          "description": "ID of the library",
          "type": "integer",
          "name": "id"
        }
      ,
        {
          "description": "Library strategy",
          "type": "string",
          "name": "library_strategy"
        }
      ,
        {
          "description": "Library type",
          "type": "string",
          "name": "library_type"
        }
      ,
        {
          "description": "Method of library preparation",
          "type": "string",
          "name": "method_library_preparation"
        }
      ,
        {
          "description": "Name of the file",
          "type": "string",
          "name": "name"
        }
      ,
        {
          "description": "ID of the patient",
          "type": "integer",
          "name": "id"
        }
      ,
        {
          "description": "Platform (e.g., Illumina)",
          "type": "string",
          "name": "platform"
        }
      ,
        {
          "description": "ID of the project",
          "type": "integer",
          "name": "project_id"
        }
      ,
        {
          "description": "Protocol name (e.g., Illumina TruSeq)",
          "type": "string",
          "name": "protocol_name"
        }
      ,
        {
          "description": "ID of the read group",
          "type": "integer",
          "name": "id"
        }
      ,
        {
          "description": "Read group status",
          "type": "string",
          "name": "read_group_status"
        }
      ,
        {
          "description": "ID of the sample",
          "type": "integer",
          "name": "sample_id"
        }
      ,
        {
          "description": "Scientific name of the organism",
          "type": "string",
          "name": "scientific_name"
        }
      ,
        {
          "description": "ID of the submission tool",
          "type": "integer",
          "name": "submission_tool_id"
        }
      ,
        {
          "description": "Submission timestamp (in seconds)",
          "type": "integer",
          "name": "submission_timestamp"
        }
      ,
        {
          "description": "ID of the study",
          "type": "integer",
          "name": "study_id"
        }
      ,
        {
          "description": "Study name",
          "type": "string",
          "name": "study_name"
        }
      ,
        {
          "description": "Submission tool (e.g., Illumina BCLConvert)",
          "type": "string",
          "name": "submission_tool"
        }
      ,
        {
          "description": "Submission timestamp (in seconds)",
          "type": "integer",
          "name": "submission_timestamp"
        }
      ,
        {
          "description": "Type of the sample",
          "type": "string",
          "name": "sample_type"
        }
      ,
        {
          "description": "ID of the submission tool",
          "type": "integer",
          "name": "id"
        }
      ]
    }
  ]}
"""
)
