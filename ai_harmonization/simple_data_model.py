import json
import re
from typing import List, Optional, Union

import pandas as pd
from langchain_core.documents import Document
from pydantic import BaseModel


class Property(BaseModel):
    description: str
    type: Union[str, List[str]]
    name: str
    additional_metadata: Optional[dict] = None
    values: Optional[List] = None


class Node(BaseModel):
    name: str
    description: str
    links: List[str]
    properties: List[Property]
    additional_metadata: Optional[dict] = None


class SimpleDataModel(BaseModel):
    nodes: List[Node]
    additional_metadata: Optional[dict] = None

    def get_property_df(self) -> pd.DataFrame:
        """
        Get a pandas DataFrame with a row per property from the SimpleDataModel.
        """
        property_list = []
        for node in self.nodes:
            for property in node.properties:
                property_list.append(
                    {
                        "node_name": node.name,
                        "property_name": property.name,
                        "property_description": property.description,
                        "property_type": property.type,
                        "additional_metadata": property.additional_metadata,
                    }
                )
        return pd.DataFrame(property_list)

    @staticmethod
    def from_simple_json(input_json: str, ignore_properties_with_endings=None):
        """
        Converts a Simple Data Model json format to a standard format object
        """
        ignore_properties_with_endings = ignore_properties_with_endings or []

        input_model = json.loads(input_json)

        nodes = []
        for node_data in input_model["nodes"]:
            properties = []
            for prop_data in node_data["properties"]:
                property_name = prop_data.get("name", prop_data.get("name:", ""))

                skip = False
                for ending in ignore_properties_with_endings:
                    if property_name.strip().endswith(ending):
                        skip = True
                if skip:
                    continue

                properties.append(
                    Property(
                        description=prop_data.get("description", ""),
                        type=prop_data.get("type", ""),
                        name=property_name,
                    )
                )

            nodes.append(
                Node(
                    name=node_data.get("name", node_data.get("name:", "")),
                    description=node_data.get("description", ""),
                    links=node_data.get("links", []),
                    properties=properties,
                )
            )

        data_model = SimpleDataModel(nodes=nodes)
        return data_model

    @staticmethod
    def from_gen3_model(input_json: str, ignore_properties_with_endings=None):
        """
        Converts a Gen3 DD JSON model to a standard format
        """
        ignore_properties_with_endings = ignore_properties_with_endings or []

        gen3_model = json.loads(input_json)
        data_model = SimpleDataModel(nodes=[])

        for node_name, node_info in gen3_model.items():
            if node_name in [
                "_terms",
                "_settings",
                "_definitions",
                "metaschema",
                "root",
            ]:
                continue

            # Convert each property in the gen3_model to our Property model
            properties = []
            for property_name, property_info in node_info.get("properties", {}).items():
                skip = False
                for ending in ignore_properties_with_endings:
                    if property_name.strip().endswith(ending):
                        skip = True
                if skip:
                    continue

                # handle foreign key links
                if "anyOf" in property_info.keys():
                    for sub_properties in property_info["anyOf"]:
                        sub_node_name = property_name

                        if sub_node_name.endswith("s"):
                            sub_node_name = sub_node_name[:-1]

                        # TODO better handle non-plural

                        if sub_node_name.endswith("ies"):
                            sub_node_name = sub_node_name[:-3] + "y"

                        for sub_property_name, sub_property_info in (
                            sub_properties.get("items", {})
                            .get("properties", {})
                            .items()
                        ):
                            skip = False
                            for ending in ignore_properties_with_endings:
                                if sub_property_name.strip().endswith(ending):
                                    skip = True
                            if skip:
                                continue

                            node_property = Property(
                                description=sub_property_info.get("description", ""),
                                type=sub_property_info.get("type", ""),
                                name=f"{sub_node_name}.{sub_property_name}",
                            )
                            if not node_property.type:
                                if "enum" in sub_property_info:
                                    node_property.type = "enum"
                            if not node_property.description:
                                if "term" in sub_property_info:
                                    node_property.description = sub_property_info[
                                        "term"
                                    ].get("description", "")

                            if (
                                "term" in sub_property_info
                                and "termDef" in sub_property_info["term"]
                            ):
                                if not node_property.additional_metadata:
                                    node_property.additional_metadata = {}

                                node_property.additional_metadata.update(
                                    {"cde_info": sub_property_info["term"]["termDef"]}
                                )
                            properties.append(node_property)
                else:
                    node_property = Property(
                        description=property_info.get("description", ""),
                        type=property_info.get("type", ""),
                        name=property_name,
                    )
                    if not node_property.type:
                        if "enum" in property_info:
                            node_property.type = "enum"
                    if not node_property.description:
                        if "term" in property_info:
                            node_property.description = property_info["term"].get(
                                "description", ""
                            )

                    if "term" in property_info and "termDef" in property_info["term"]:
                        if not node_property.additional_metadata:
                            node_property.additional_metadata = {}

                        node_property.additional_metadata.update(
                            {"cde_info": property_info["term"]["termDef"]}
                        )

                    properties.append(node_property)

            node = Node(
                name=node_name,
                description=node_info["description"],
                properties=properties,
                links=[
                    link_info["name"]
                    for link_info in node_info.get("links", [])
                    if "name" in link_info
                ],
            )

            if not node.links and "subgroup" in node_info.get("links", {}):
                node.links = [
                    link_info["name"] for link_info in node_info["links"]["subgroup"]
                ]

            data_model.nodes.append(node)

        return data_model

    @staticmethod
    def from_gdc_model(input_json: str):
        """
        Converts a GDC model to a standard SimpleDataModel
        Accepts input as a json string or a parsed dict.
        """
        # If input is a string, parse it
        if isinstance(input_json, str):
            gdc_dict = json.loads(input_json)
        else:
            gdc_dict = input_json

        nodes = []
        for node_name, node_data in gdc_dict.items():
            # Skip metadata/_definitions nodes
            if node_name.startswith("_"):
                continue

            # Node might have description missing, so use empty string
            node_description = node_data.get("description", "")
            properties = []

            for prop_name, prop_data in node_data.get("properties", {}).items():
                # Exctract values
                values = None
                if "enum" in prop_data:
                    values = prop_data["enum"]
                elif "values" in prop_data:
                    values = prop_data["values"]

                # Build a Property object
                prop = Property(
                    name=prop_name,
                    description=prop_data.get("description", ""),
                    type=prop_data.get("type", ""),
                    values=values,
                )
                properties.append(prop)

            # Links: GDC rarely has links, but may add as empty list
            links = []
            if isinstance(node_data.get("links", []), list):
                links = [
                    link.get("name", "")
                    for link in node_data.get("links", [])
                    if "name" in link
                ]
            elif (
                isinstance(node_data.get("links", {}), dict)
                and "subgroup" in node_data["links"]
            ):
                links = [
                    link_info["name"] for link_info in node_data["links"]["subgroup"]
                ]()

            # Build Node
            node = Node(
                name=node_name,
                description=node_description,
                properties=properties,
                links=links,
            )

            nodes.append(node)

        data_model = SimpleDataModel(nodes=nodes)
        return data_model

    @staticmethod
    def from_linkml_jsonschema(input_json: str, ignore_properties_with_endings=None):
        """
        Converts a LinkML JSON model to a standard format
        """
        ignore_properties_with_endings = ignore_properties_with_endings or []

        input_model = json.loads(input_json)
        data_model = SimpleDataModel(nodes=[])

        for node_name, node_info in input_model["$defs"].items():
            node_description: str = node_info.get("description", "")
            properties: List[Property] = []
            for property_name, property_info in node_info.get("properties", {}).items():
                skip = False
                for ending in ignore_properties_with_endings:
                    if property_name.strip().endswith(ending):
                        skip = True
                if skip:
                    continue

                property_type = property_info.get("type", "")
                property_description = property_info.get("description", "")
                if "type" in property_info:
                    del property_info["type"]
                if "description" in property_info:
                    del property_info["description"]
                properties.append(
                    Property(
                        name=property_name,
                        description=property_description,
                        type=property_type,
                        additional_metadata=property_info,
                    )
                )
            if "properties" in node_info:
                del node_info["properties"]

            del node_info["description"]

            # TODO: populate links
            node_links = []

            data_model.nodes.append(
                Node(
                    name=node_name,
                    properties=properties,
                    links=node_links,
                    description=node_description,
                    additional_metadata=node_info,
                )
            )

        return data_model

    @staticmethod
    def get_from_unknown_json_format(input_json: str, *args, **kwargs):
        simple_data_model = None
        exception: BaseException = BaseException()

        try:
            simple_data_model = SimpleDataModel.from_simple_json(
                input_json, *args, **kwargs
            )
        except BaseException as exc:
            exception = exc
            pass

        try:
            simple_data_model = SimpleDataModel.from_gen3_model(
                input_json, *args, **kwargs
            )
        except BaseException as exc:
            exception = exc
            pass

        try:
            simple_data_model = SimpleDataModel.from_gdc_model(input_json)
        except BaseException as exc:
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
    match = re.match(r"^(.*?)\.(.*?)\s*(?:\(((?:(?!\)).)*?)\):|:)(.*)$", input_string)
    if match:
        node_name = match.group(1) or ""
        prop_name = match.group(2) or ""
        prop_type = match.group(3) or ""
        prop_desc = match.group(4) or ""
        return (
            node_name.strip(),
            prop_name.strip(),
            prop_type.strip(),
            prop_desc.strip(),
        )
    return ("", "", "", "")


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
