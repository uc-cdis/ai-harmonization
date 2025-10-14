import pandas as pd


def prefiltering_to_inference_prompt(df, prompt_format_file_path):

    # read in prompt format file #
    with open(prompt_format_file_path, "r", encoding="utf-8") as file:
        prompt_format = file.read()

    inference_prompts = []

    # for each unique source node.property returned from the prefiltering approach #
    for instance in df["Original Node.Property"].unique():

        # Get the source node.property and the property description (there should only be one) #
        source_node_property = df[df["Original Node.Property"] == instance][
            "Original Node.Property"
        ].iloc[0]
        source_description = df[df["Original Node.Property"] == instance][
            "Original Description"
        ].iloc[0]

        # Get a list of prefiltered target node.property(s) and property descriptions (there should be at least one) #
        target_node_properties = list(
            df[df["Original Node.Property"] == instance][
                "Suggested Target Node.Property"
            ]
        )
        target_descriptions = list(
            df[df["Original Node.Property"] == instance]["Target Description"]
        )

        # Parse the source node.property and description into a data model (data dict) format #
        source_format = {
            "node": source_node_property.split(".")[0],
            "property": source_node_property.split(".")[1],
            "description": source_description,
        }

        # Iterate through the target node.property(s) and property descriptions and parse to data model format #
        """
        {nodes: [

            {name: node_1,
            properties: [
                {name: property1, description: property1_description},
                {name: property2, description: property2_description},
                {name: property3, description: property3_description},
                ]
            },

            {name: node_2,
            properties: [
                {name: property1, description: property1_description},
                ]
            },

            {name: node_3,
            properties: [
                {name: property1, description: property1_description},
                {name: property2, description: property2_description},
                ]
            }
            ]
        }
        """
        target_format = {"nodes": []}
        for i in range(len(target_node_properties)):
            target_node_name = target_node_properties[i].split(".")[0]
            target_property_name = target_node_properties[i].split(".")[1]

            existing_nodes = [x["name"] for x in target_format["nodes"]]
            if target_node_name not in existing_nodes:
                target_format["nodes"].append(
                    {
                        "name": target_node_name,
                        "properties": [
                            {
                                "name": target_property_name,
                                "description": target_descriptions[i],
                            }
                        ],
                    }
                )
            else:
                [
                    x["properties"].append(
                        {
                            "name": target_property_name,
                            "description": target_descriptions[i],
                        }
                    )
                    for x in target_format["nodes"]
                    if target_node_name == x["name"]
                ]

        # Replace formatted source and target data model information with the prompt placeholder value #
        formated_prompt_instance = prompt_format.replace(
            "input_source_model", str(source_format)
        )
        formated_prompt_instance = formated_prompt_instance.replace(
            "input_target_model", str(target_format)
        )

        # Append the formated prompt to the list of inference prompts
        inference_prompts.append(formated_prompt_instance)

    # Create a dataframe of inference prompts to be fed to the model
    inference_data_dict = {
        "label": [f"prompt_{i}" for i in range(len(inference_prompts))],
        "input_text": inference_prompts,
        "expected_output_text": [""]
        * len(
            inference_prompts
        ),  # TODO - functionality for if an expected output exists
    }
    inference_df = pd.DataFrame(inference_data_dict)

    return inference_df
