import re
from typing import List, Union
import json

from langchain_core.documents import Document
from pydantic import BaseModel


class Property(BaseModel):
    description: str
    type: Union[str, List[str]]
    name: str


class Node(BaseModel):
    name: str
    description: str
    links: List[str]
    properties: List[Property]


class SimpleDataModel(BaseModel):
    nodes: List[Node]

    @staticmethod
    def from_simple_json(input_json: str):
        """
        Converts a Simple Data Model json format to a standard format object
        """
        input_model = json.loads(input_json)

        nodes = []
        for node_data in input_model["nodes"]:
            properties = []
            for prop_data in node_data["properties"]:
                properties.append(
                    Property(
                        description=prop_data.get("description", ""),
                        type=prop_data.get("type", ""),
                        name=prop_data.get("name", ""),
                    )
                )

            nodes.append(
                Node(
                    name=node_data.get("name", ""),
                    description=node_data.get("description", ""),
                    links=node_data.get("links", []),
                    properties=properties,
                )
            )

        data_model = SimpleDataModel(nodes=nodes)
        return data_model

    @staticmethod
    def from_gen3_model(input_json):
        """
        Converts a Gen3 DD JSON model to a standard format
        """
        data_model = json.loads(input_json)

        # FIXME: Update to actually convert from Gen3 format

        nodes = []
        for node_data in data_model["nodes"]:
            properties = []
            for prop_data in node_data["properties"]:
                properties.append(
                    Property(
                        description=prop_data.get("description", ""),
                        type=prop_data.get("type", ""),
                        name=prop_data.get("name", ""),
                    )
                )

            nodes.append(
                Node(
                    name=node_data.get("name", ""),
                    description=node_data.get("description", ""),
                    links=node_data.get("links", []),
                    properties=properties,
                )
            )

        data_model = SimpleDataModel(nodes=nodes)
        return data_model

    @staticmethod
    def get_from_unknown_json_format(input_json: str):
        simple_data_model = None
        exception: BaseException = BaseException()

        try:
            simple_data_model = SimpleDataModel.from_simple_json(input_json)
        except BaseException as exc:
            print("Failed to convert using simple converter.")
            exception = exc
            pass

        try:
            simple_data_model = SimpleDataModel.from_gen3_model(input_json)
        except BaseException as exc:
            print("Failed to convert using Gen3 converter.")
            exception = exc
            pass

        if not simple_data_model:
            print(
                "Could not convert from unknown format to SimpleDataModel. Consider writing a custom converter."
            )
            raise exception

        return simple_data_model


def get_node_prop_type_desc_from_string(input_string: str) -> tuple[str, str, str, str]:
    """
    Parses a string of the format "node.property_name (type): desc" or "node.property_name: desc"
    and returns a tuple containing the node name, property name, property type, and property description.
    """
    match = re.match(r"^(.*?)\.(.*?)\s*(?:\((.*?)\):\s*(.*)|:\s*(.*))$", input_string)
    if match:
        node_name = match.group(1)
        prop_name = match.group(2)
        if match.group(3):
            prop_type = match.group(3)
            prop_desc = match.group(4)
        else:
            prop_type = ""
            prop_desc = match.group(5)
        return node_name, prop_name, prop_type, prop_desc
    return "", "", "", ""


def get_data_model_as_node_prop_type_descriptions(
    data_model: SimpleDataModel,
) -> List[str]:
    """
    Retrieves and converts a data model output to a list of strings with the format:
        node_name.property_name (type): property_desc

    Returns:
        List[str]: A list of strings representing the property
    """
    node_prop_descs = []
    for node in data_model.nodes:
        for property in node.properties:
            value = get_node_property_as_string(node, property)
            node_prop_descs.append(value)
    return node_prop_descs


def get_node_property_as_string(node: Node, node_property: Property) -> str:
    property_desc = node_property.description.replace("\t", "    ").replace("\n", " ")
    return f"{node.name}.{node_property.name} ({node_property.type}): {property_desc}"


def get_data_model_as_langchain_documents(
    data_model: SimpleDataModel,
) -> List[Document]:
    """
    Retrieves and converts a data model output to LangChain documents.

    Returns:
        List[Document]: A list of Document objects representing the converted LangChain documents.
    """
    documents = []
    node_prop_descs = get_data_model_as_node_prop_type_descriptions(data_model)

    for value in node_prop_descs:
        document = Document(page_content=value, metadata={})
        documents.append(document)

    return documents
