import bz2
import csv
import gzip
import json
import logging
import os
import re
import shutil
import tarfile
import zipfile
from csv import DictReader
from typing import List, Union

import pandas as pd
import requests
from pydantic import BaseModel
from tqdm import tqdm

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


def read_in_json(file_location):
    try:
        with open(file_location, "r") as f:
            data = json.load(f)
        return data, None
    except Exception as e:
        return None, e


def make_dir(dirname):
    logging.info(f"outputting to: {dirname}")
    try:
        os.makedirs(dirname, exist_ok=True)
    except Exception as e:
        logging.info("failed to make dir")
        logging.info(str(e))


def unzip_files(directory: str, remove_compressed_file: bool = True) -> None:
    """
    Unzip various types of compressed files (.gz, .tar, .tar.gz, .zip, .bz2)
    in the specified directory and its subdirectories.

    Parameters:
        directory (str): The path to the directory to search for compressed files.
        remove_compressed_file (bool): Whether to remove the compressed file after unzipping.
    """
    # Iterate through all files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            compressed_file_path = os.path.join(root, file)
            is_compressed_file = True

            if file.endswith(".gz") and not file.endswith(".tar.gz"):
                # Handle .gz files
                output_file_path = os.path.splitext(compressed_file_path)[
                    0
                ]  # Remove .gz extension
                with gzip.open(compressed_file_path, "rb") as f_in:
                    with open(output_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                logging.info(f"Unzipped: {compressed_file_path} to {output_file_path}")

            elif file.endswith(".tar"):
                # Handle .tar files
                with tarfile.open(compressed_file_path, "r") as tar:
                    tar.extractall(
                        path=root
                    )  # Extract all files into the same directory as the .tar

                logging.info(f"Extracted: {compressed_file_path} to {root}")

            elif file.endswith(".tar.gz") or file.endswith(".tgz"):
                # Handle .tar.gz files
                with tarfile.open(compressed_file_path, "r:gz") as tar:
                    tar.extractall(
                        path=root
                    )  # Extract all files into the same directory as the .tar.gz

                logging.info(f"Extracted: {compressed_file_path} to {root}")

            elif file.endswith(".zip"):
                # Handle .zip files
                with zipfile.ZipFile(compressed_file_path, "r") as zip_ref:
                    zip_ref.extractall(
                        root
                    )  # Extract all files into the same directory as the .zip

                logging.info(f"Extracted: {compressed_file_path} to {root}")

            elif file.endswith(".bz2"):
                # Handle .bz2 files
                output_file_path = os.path.splitext(compressed_file_path)[
                    0
                ]  # Remove .bz2 extension
                with bz2.open(compressed_file_path, "rb") as f_in:
                    with open(output_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                logging.info(f"Unzipped: {compressed_file_path} to {output_file_path}")

            else:
                is_compressed_file = False

            # Remove the compressed file if specified
            if is_compressed_file and remove_compressed_file:
                logging.info(f"Removing: {compressed_file_path}")
                os.remove(compressed_file_path)


def remove_unwanted_files(
    directory: str, inclusion_list: list[str], exclusion_list: list[str]
) -> None:
    """
    Walks through all folders and files in the given directory,
    and removes files that don't follow the inclusion and exclusion lists.

    Parameters:
        directory (str): The path to the root directory.
        inclusion_list (list[str]): List of allowed file extensions to keep.
        exclusion_list (list[str]): List of file extensions to explicitly remove.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Check file extension
            _, ext = os.path.splitext(file)
            if file.endswith(tuple(exclusion_list)):
                # Remove if it's in the exclusion list
                print(f"Removing excluded file: {file_path}")
                os.remove(file_path)
            elif ext not in inclusion_list:
                # Remove if it's not in the inclusion list
                print(f"Removing unwanted file: {file_path}")
                os.remove(file_path)


def generate_markdown_from_directory_with_custom_metrics(
    directory, prefix="custom_metrics_"
):
    """
    Generate markdown files from a directory of CSV files with Custom Metrics.

    Args:
        directory (str): The path to the directory containing the CSV files. Defaults to None.
        prefix (str, optional): The prefix for the CSV file names. Defaults to "custom_metrics_".

    Returns:
        None
    """

    if directory is not None and not os.path.exists(directory):
        logging.error(f"Directory '{directory}' does not exist.")
        return

    output_dir = os.path.abspath(os.path.join(os.getcwd(), "markdown"))
    if directory is not None:
        output_dir = os.path.abspath(os.path.join(directory, "markdown"))

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if (
                file.endswith(".csv")
                and "custom_metrics_" in file
                and ".ipynb_checkpoints" not in file
            ):
                file_path = os.path.join(root, file)
                # Read the CSV file into a DictReader object
                with open(file_path, "r") as input_file:
                    reader = DictReader(input_file)

                    # Iterate over each row in the reader
                    for index, row in enumerate(reader):
                        metrics_json = json.loads(
                            row["custom_metrics"].replace('""', '"').replace("'", '"')
                        )
                        metrics_json = {
                            key: value
                            for key, value in metrics_json["accuracy"].items()
                            if key.startswith("_")
                        }
                        info_to_write = {
                            "metrics": json.dumps(metrics_json, indent=2),
                            "input": row["input_text"].replace('""', '"'),
                            "output": row["generated_output_text"].replace('""', '"'),
                            "expected_output": row["output_text"].replace('""', '"'),
                        }

                        # Create the output file path
                        output_file_path = (
                            file_path.replace(".csv", "") + f"/line_{index}.md"
                        )

                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                        # Write the row to a new CSV file
                        with open(output_file_path, "w") as output_file:
                            output_file.write("Metrics:\n" + info_to_write["metrics"])
                            output_file.write("\n\nInput:\n" + info_to_write["input"])
                            output_file.write("\n\nAI Model Output:\n")
                            if True:  # wrap_model_output_in_json_codeblock
                                output_file.write("```json\n")
                            output_file.write(info_to_write["output"])
                            if True:  # wrap_model_output_in_json_codeblock
                                output_file.write("\n```")

                            output_file.write(
                                "\n\nExpected Output:\n```json\n"
                                + info_to_write["expected_output"]
                                + "\n```"
                            )


def write_to_csv(filepath: str, data, header=None):
    """Writes a CSV file to the given filepath, creating directories if needed.

    Args:
        filepath (str): The full path of the CSV file.
        data (list of lists): The data to write.
        header (list, optional): Column headers for the CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)
    return True


def nodeproperty_to_curie(s: str) -> str:
    """
    Converts a 'node.property' string to a CURIE with EX: prefix, e.g.
      'MedicalEncounterRecord.Molecular Evidence Present' ->
      'EX:MedicalEncounterRecord_Molecular_Evidence_Present'
    If the input does not contain a dot, treats the whole string as 'node'.
    """
    if "." in s:
        node, prop = s.split(".", 1)
    else:
        node, prop = s, ""

    def munge(t):
        t = t.strip()
        # Convert all non-alphanumeric characters to underscores
        t = re.sub(r"\W+", "_", t)
        # Remove leading/trailing underscores
        t = t.strip("_")
        return t

    node_str = munge(node)
    prop_str = munge(prop)
    if prop_str:
        return f"EX:{node_str}_{prop_str}"
    else:
        return f"EX:{node_str}"


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


def get_gen3_json_schemas_and_templates(
    url: str, output_dir: str, internal_folder_name: str = "Unmodified"
) -> None:
    """
    Fetches JSON schema files and templates from a given Gen3 API URL and saves them to the specified directory.

    Args:
        url (str): The base URL of the API endpoint.
        output_dir (str): The directory where the fetched data will be saved.
        internal_folder_name (str, optional): The name of the internal folder (which is within the {{url}} folder. Defaults to "Unmodified".
    Returns:
        None
    """
    # Construct full directory path
    output_dir = (
        output_dir.rstrip("/") + f"/{url.split('://')[-1]}" + f"/{internal_folder_name}"
    )
    try:
        all_schemas_url = url.rstrip("/") + "/api/v0/submission/_dictionary/_all"
        # Fetch data from the API
        all_schemas_response = requests.get(all_schemas_url)
        all_schemas_data = all_schemas_response.json()
        # Define a directory to save JSON schema files
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/nodes/", exist_ok=True)
        total_schemas = len(all_schemas_data)

        schema_filename = f"{output_dir}/schema.json"
        with open(schema_filename, "w") as file:
            json.dump(all_schemas_data, file, indent=4)

        for key, value in tqdm(
            all_schemas_data.items(),
            total=total_schemas,
            desc=f"Getting Gen3 JSON Schemas from {all_schemas_url}:",
        ):
            schema_filename = f"{output_dir}/nodes/{key}_schema.json"
            with open(schema_filename, "w") as file:
                json.dump(value, file, indent=4)
            # Get the template for this item and output in the same dir
            template_url = url.rstrip("/") + f"/api/v0/submission/template/{key}"
            template_response = requests.get(template_url)
            if template_response.status_code == 200:
                template_filename = f"{output_dir}/nodes/{key}_template.tsv"
                with open(template_filename, "w") as file:
                    file.write(template_response.text)
        logging.info(
            f"All JSON schemas and templates for {url} are saved here: {output_dir}."
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
