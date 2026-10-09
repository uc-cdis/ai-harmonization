import pandas as pd
from IPython.display import display
from itables.widget import ITable


def get_interactive_table_for_suggestions(
    suggestions_df, column_for_filtering=1, table_description=None, **kwargs
):
    """
    Builds an interactive table of harmonization suggestions.

    The table has a search pane that filters on one column, allows selecting
    several rows, and prints ``table_description`` as usage notes.

    Args:
        suggestions_df (pd.DataFrame): The suggestions to show.
        column_for_filtering (int): Index of the column the search pane filters.
        table_description (str, optional): The usage notes to print; a default
            set of instructions is used when not given.
        **kwargs: Passed to ``itables.widget.ITable``.

    Returns:
        ITable: The table widget.
    """
    table_description = (
        table_description
        or """
    This table allows you to:
    1. Filter for each "Original Node.Property"
    2. Make a selection from the table below
        (which contains the suggestions based on CDEs for the selected "Original Node.Property")
    3. Multi-select (a ctrl / cmd click will add and remove items from the selection)
    """
    )
    table = ITable(
        suggestions_df,
        select=True,
        buttons=[
            {
                "extend": "colvis",
                "collectionLayout": "fixed columns",
                "popoverTitle": "Column visibility control",
            }
        ],  # "copyHtml5", "csvHtml5", "excelHtml5"
        layout={"top1": "searchPanes"},
        searchPanes={
            "layout": "columns-1",
            "cascadePanes": True,
            "columns": [column_for_filtering],
        },
        # showIndex=False,
        columnDefs=[{"className": "dt-left", "targets": "_all"}],
        # classes="display nowrap table_with_monospace_font",
        allow_html=True,
        **kwargs,
    )
    print(table_description)
    return table


def get_nodes_and_properties_df(
    data_model_json: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a dictionary containing data model information and returns two pandas DataFrames.
    The first DataFrame contains nodes with their names and descriptions, and the second DataFrame contains
    properties of these nodes.

    Parameters:
        data_model_json (dict): A dictionary containing data model information.

    Returns:
        tuple: Two pandas DataFrames - one for nodes and another for properties.
    """
    nodes = []
    properties = []
    for node in data_model_json["nodes"]:
        nodes.append({"Name": node["name"], "Description": node["description"]})
        for prop in node["properties"]:
            properties.append({"Node Name": node["name"], **prop})

    nodes_df = pd.DataFrame(nodes)
    properties_df = pd.DataFrame(properties)

    return nodes_df, properties_df
