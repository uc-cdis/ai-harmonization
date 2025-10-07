import json
import logging
import os
import pprint
import warnings
from typing import List, Union

import chromadb
import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import Optional, Tuple, List
from chromadb.api.types import (
    Documents,
    Embeddings,
    IDs,
    Metadatas,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.abspath(f"{CURRENT_DIR}/../output/temp")


class Property(BaseModel):
    description: str
    type: Union[str, List[str]]
    name: str


class Node(BaseModel):
    name: str
    description: str
    links: List[str]
    properties: List[Property]


class DataModel(BaseModel):
    nodes: List[Node]


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


def get_langchain_vectorstore_and_persistent_client(
    persist_directory_name: str = "cdes",
    embedding_function: Union[Embeddings, None] = None,
    chromadb_settings: Union[chromadb.config.Settings, None] = None,
) -> Chroma:
    """
    Creates and returns a Chroma vectorstore for the 'cdes' topic.

    Parameters:
        persist_directory_name (str): The name of the persistent directory.
        embedding_function (Embeddings | None): The language embedding function. Defaults to HuggingFaceEmbeddings with the "all-MiniLM-L6-v2" model.
        chromadb_settings (chromadb.config.Settings | None): ChromaDB settings. Defaults to chromadb Settings with migrations_hash_algorithm="sha256" and anonymized_telemetry=False.

    Returns:
        Chroma: A Chroma vectorstore object.
    """
    # We're using a general purpose language embedding algorithm
    embedding_function = embedding_function or HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    settings = chromadb_settings or chromadb.Settings(
        migrations_hash_algorithm="sha256",
        anonymized_telemetry=False,
    )

    persistent_client = chromadb.PersistentClient(
        path=f"{TEMP_DIR}/vectorstore/{persist_directory_name}", settings=settings
    )
    vectorstore = Chroma(
        client=persistent_client,
        collection_name=persist_directory_name,
        embedding_function=embedding_function,
        # https://docs.trychroma.com/usage-guide#changing-the-distance-function
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=f"{TEMP_DIR}/vectorstore/{persist_directory_name}",
        client_settings=settings,
    )

    return vectorstore, persistent_client


def create_batches(
    batch_size: int,
    ids: IDs,
    embeddings: Optional[Embeddings] = None,
    metadatas: Optional[Metadatas] = None,
    documents: Optional[Documents] = None,
) -> List[Tuple[IDs, Embeddings, Optional[Metadatas], Optional[Documents]]]:
    """
    Returns batches of provided batch_size from the lists of ids, embeddings, metadatas and documents.

    Mimics chromadb.utils.batch_utils.create_batches behaviour but instead of using api.get_max_batch_size() for the batch creation it uses provided batch_size parameter.

    Args:
      ids (IDs): list of document IDs
      embeddings ([Embeddings]): optional list of embeddings,
      metadatas ([Metadatas]): optional list of metadatas,
      documents ([Documents]): optional list of documents

    Returns:
        List[Tuple[IDs, Embeddings, Optional[Metadatas], Optional[Documents]]]: list of batches
    """
    _batches: List[Tuple[IDs, Embeddings, Optional[Metadatas], Optional[Documents]]] = (
        []
    )
    if len(ids) > batch_size:
        # create split batches
        for i in range(0, len(ids), batch_size):
            _batches.append(
                (
                    ids[i : i + batch_size],
                    embeddings[i : i + batch_size] if embeddings else None,
                    metadatas[i : i + batch_size] if metadatas else None,
                    documents[i : i + batch_size] if documents else None,
                )
            )
    else:
        _batches.append((ids, embeddings, metadatas, documents))
    return _batches


def add_documents_to_vectorstore(
    documents, vectorstore, persistent_client, batch_size=None
):
    batch_size = batch_size or persistent_client.get_max_batch_size()
    logging.info(
        "Number of documents that can be inserted at once:",
        batch_size,
    )
    ids = range(len(documents))
    batches = create_batches(
        batch_size=batch_size, ids=list(ids), documents=list(documents)
    )
    for batch in batches:
        logging.info(f"Adding batch of size {len(batch[0])}")
        vectorstore.add_documents(documents=batch[3])


def get_similar_documents(vectorstore, query, **kwargs) -> List[Document]:
    """
    Retrieves similar documents from a vector store using a given query.

    This function uses the `vectorstore.similarity_search_with_relevance_scores` method to find similar documents.
    The returned list of documents includes their relevance scores, which can be used for ranking or filtering purposes.

    Args:
        vectorstore (Chroma): The Chroma vector store to search in.
        query (str): The search query to use when retrieving similar documents.
        **kwargs: Optional keyword arguments to pass to the `vectorstore.similarity_search_with_relevance_scores` method.

    Returns:
        List[Document]: A list of Document objects representing the retrieved similar documents, along with their relevance scores.
    """
    # We want to ignore the printed warning when there's no documents.
    # We don't want that in stdout polluting it, we just will have an empty list
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vectorstore.similarity_search_with_relevance_scores(query, **kwargs)


def get_similar_documents_as_string(documents: tuple[Document, float]) -> str:
    """
    This function takes a tuple of Document objects and their corresponding similarity scores,
    and returns a formatted string representation of the input documents.

    The returned string includes the document's metadata, which is converted using pprint.pformat() for readability.

    Args:
        documents (tuple[Document, float]): A tuple containing a Document object and its corresponding similarity score.

    Returns:
        str: A formatted string representation of the input documents.
    """
    output = ""
    for doc, similarity_score in documents:
        output += f"-" * 80 + "\n"
        output += f"similarity_score: {similarity_score}" + "\n"
        output += f"             ids: {doc.metadata['ids']}" + "\n"
        output += f"           names: {doc.metadata['names']}" + "\n"
        output += f"          source: {doc.metadata['source']}" + "\n"
        output += f"     nihEndorsed: {doc.metadata['nihEndorsed']}" + "\n"
        output += f"     valueDomain: {doc.metadata['valueDomain']}" + "\n"
        output += f"\n   full metadata:\n"
        output += f"{pprint.pformat(doc.metadata)}" + "\n"
    return output


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


def get_data_model_as_langchain_documents(data_model: DataModel) -> List[Document]:
    """
    Retrieves and converts a data model output to LangChain documents.

    Returns:
        List[Document]: A list of Document objects representing the converted LangChain documents.
    """
    documents = []
    node_prop_descs = get_data_model_as_node_prop_descriptions(data_model)

    for value in node_prop_descs:
        document = Document(page_content=value, metadata={})
        documents.append(document)

    return documents


def get_data_model_as_node_prop_descriptions(data_model: DataModel) -> List[str]:
    """
    Retrieves and converts a data model output to a list of strings with the format:
        node_name.property_name: property_desc

    Returns:
        List[str]: A list of strings representing the property
    """
    if "nodes" not in data_model.keys():
        raise Exception(
            f"Input data model does not conform to schema. Must have nodes with properties and descriptions"
        )

    node_prop_descs = []
    for node in data_model.get("nodes", []):
        node_name = node.get("name", "")
        for property in node.get("properties", []):
            property_name = property.get("name", "")
            property_desc = (
                property.get("description", "").replace("\t", "    ").replace("\n", " ")
            )
            value = f"{node_name}.{property_name}: {property_desc}"
            node_prop_descs.append(value)
    return node_prop_descs


def get_gen3_data_model_as_langchain_documents(gen3_dd_schema) -> List[Document]:
    """
    Retrieves and converts a Gen3 data model output to LangChain documents.

    Returns:
        List[Document]: A list of Document objects representing the converted LangChain documents.
    """
    documents = []
    node_prop_descs = get_gen3_data_model_as_node_prop_descriptions(gen3_dd_schema)
    for value in node_prop_descs:
        document = Document(page_content=value, metadata={})
        documents.append(document)
    return documents


def get_gen3_data_model_as_node_prop_descriptions(gen3_dd_schema) -> List[str]:
    """
    Retrieves and converts a Gen3 data model output to a list of strings with the format:
        node_name.property_name: property_desc

    Returns:
        List[str]: A list of strings representing the property
    """
    node_prop_descs = []
    for node_name, node in gen3_dd_schema.items():
        if node_name.startswith("_"):
            continue

        for property_name, property in node.get("properties", {}).items():
            try:
                if type(property) == str:
                    property_desc = property
                else:
                    property_desc = (
                        property.get("description", "")
                        .replace("\t", "    ")
                        .replace("\n", " ")
                    )
                value = f"{node_name}.{property_name}: {property_desc}"
                node_prop_descs.append(value)
            except Exception as exc:
                logging.warning(
                    f"Skipping {property_name},{property} due to failed parsing, continuing..."
                )

    return node_prop_descs


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
