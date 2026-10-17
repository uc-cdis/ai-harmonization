"""The GDC benchmark.

Variables from the supplementary tables of ten cancer papers, mapped by
experts to the GDC data dictionary: the ground truth of GDC-SM, the schema
matching benchmark published with Magneto. GDC-SM distributes the mappings
only, so the source tables are downloaded from their publishers and the
targets from the live GDC submission dictionary.
"""

import io
import json
import zipfile
from datetime import date
from pathlib import PurePosixPath

import pandas as pd

from ai_harmonization.formatters import VALUE_SEPARATOR
from ai_harmonization.simple_data_model import SimpleDataModel
from embedding_model.download import cached, fetch, sha256

# The manifest source that pins the ground truth, by its Zenodo record.
GROUND_TRUTH = "GDC-SM ground truth"
ZENODO_FILE = "https://zenodo.org/records/{record}/files/{name}?download=1"
GDC_DICTIONARY = "https://api.gdc.cancer.gov/v0/submission/_dictionary/_all"
GDC_STATUS = "https://api.gdc.cancer.gov/status"
# Entries the API serves beside the nodes: the JSON meta-schema and the root.
NOT_NODES = {"metaschema", "root"}

# The ground truth names a property, not a node. So a property that several
# nodes carry is an expected target in each of them, except where a node gives
# the name another meaning. A project's primary site and disease type describe
# the project, not a patient. A portion's weight is a sample's, in milligrams,
# not a patient's.
NOT_EXPECTED = {"project.primary_site", "project.disease_type", "portion.weight"}


def dictionary_targets(dictionary):
    """One target per data property in the GDC dictionary, as ``node.property``.

    The dictionary is read by ``SimpleDataModel.from_gdc_model``, as the
    harmonization approaches read it, so a target's text is the text they
    build. Of its properties, these are not targets: links (``anyOf``), which
    point to other nodes rather than hold data; the bookkeeping fields each
    node lists as ``systemProperties`` -- ``id``, ``created_datetime``,
    ``state`` -- and every node's ``type``, which holds the node's own name.
    Nor are the JSON meta-schema and root the API serves beside the nodes.

    Args:
        dictionary (dict): The dictionary as the GDC API serves it: nodes by
            name, each with its ``properties``; keys starting with ``_`` hold
            shared definitions, not nodes.

    Returns:
        list[dict]: The targets, keyed by TARGET_COLUMNS and sorted.
    """
    targets = []
    for node in SimpleDataModel.from_gdc_model(dictionary).nodes:
        if node.name in NOT_NODES:
            continue
        schema = dictionary[node.name]
        bookkeeping = {"type", *(schema.get("systemProperties") or [])}
        for prop in node.properties:
            if prop.name in bookkeeping or "anyOf" in schema["properties"][prop.name]:
                continue
            declared = prop.type
            if isinstance(declared, list):
                declared = "/".join(str(kind) for kind in declared if kind != "null")
            targets.append(
                {
                    "target_property": f"{node.name}.{prop.name}",
                    "target_type": str(declared),
                    "target_description": prop.description,
                    "target_values": VALUE_SEPARATOR.join(
                        str(value) for value in prop.values or []
                    ),
                }
            )
    targets.sort(key=lambda row: row["target_property"])
    return targets


def read_ground_truth(archive):
    """The expert mappings in the GDC-SM archive, one table per paper.

    Args:
        archive (bytes): ``gdc-sm-data.zip`` from the Zenodo record.

    Returns:
        dict[str, pd.DataFrame]: By paper, the columns
            ``original_paper_variable_names`` and ``GDC_format_variable_names``.
    """
    truth = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for name in zipped.namelist():
            if "/ground-truth/" in name and name.endswith(".csv"):
                truth[PurePosixPath(name).stem] = pd.read_csv(
                    io.BytesIO(zipped.read(name))
                )
    return truth


def read_source_table(spreadsheet, sheet, paper):
    """A paper's source table, as GDC-SM's own download script prepares it.

    The script (``gdc_download.py`` in the Zenodo record) reads the
    publisher's sheet, repairs four of the tables and saves each as CSV, and
    the same steps are taken here: Vasaikar's sheet loses its first row, of
    type labels (BIN, CON, ...), and its last, of counts; a tab is removed
    from a McDermott column name; a typo in Clark's ethnicity values is
    corrected; and Wang's secondhand-smoke answers are recoded to Yes and No.

    Args:
        spreadsheet (bytes): The publisher's ``.xlsx`` file.
        sheet (str): The sheet holding the table.
        paper (str): The paper's name in GDC-SM, such as ``Vasaikar``.

    Returns:
        pd.DataFrame: The table, with the column types its CSV is read back
            with.
    """
    table = pd.read_excel(io.BytesIO(spreadsheet), sheet_name=sheet)
    if paper == "Vasaikar":
        table = table.iloc[1:-1]
    if paper == "McDermott":
        table = table.rename(
            columns={
                "Age in Months at Time of Tissue Procurement \t": (
                    "Age in Months at Time of Tissue Procurement"
                )
            }
        )
    if paper == "Clark":
        table["Ethnicity_Self_Identify"] = table["Ethnicity_Self_Identify"].replace(
            "Whte", "White"
        )
    if paper == "Wang":
        table["secondhand_smoke_exposure"] = table["secondhand_smoke_exposure"].replace(
            {
                "No or minimal exposure to secondhand smoke": "No",
                "Exposed in childhood houshold": "Yes",
                "Exposed in current household": "Yes",
            }
        )
    # GDC-SM's tables are these sheets saved as CSV. Reading one back gives its
    # column types: Vasaikar's ages, text while the label row was above them,
    # are numbers again.
    return pd.read_csv(io.StringIO(table.to_csv(index=False)))


def mapping_rows(truth, tables, targets):
    """One row per source variable, with every target the experts mapped it to.

    A variable keeps every distinct value its column holds; the prompt
    formatters cap how many they show. A published mapping whose property the
    dictionary does not carry is left out and named, since a new dictionary
    can drop or rename a property.

    Args:
        truth (dict[str, pd.DataFrame]): The ground truth, from
            read_ground_truth.
        tables (dict[str, pd.DataFrame]): The source tables, by paper.
        targets (list[dict]): The dictionary's targets, from
            dictionary_targets.

    Returns:
        list[dict]: The rows, keyed by MAPPING_COLUMNS and sorted by paper and
            variable. ``expected_target`` is a sorted list of target keys.

    Raises:
        ValueError: If a mapped variable is not a column of its paper's table.
    """
    nodes_by_property = {}
    for target in targets:
        key = target["target_property"]
        if key not in NOT_EXPECTED:
            nodes_by_property.setdefault(key.split(".", 1)[1], []).append(key)

    variables, missing, dropped = {}, [], []
    for paper, published in sorted(truth.items()):
        table = tables.get(paper, pd.DataFrame())
        for source, target in zip(
            published["original_paper_variable_names"],
            published["GDC_format_variable_names"],
        ):
            if source not in table.columns:
                missing.append(f"{paper}: {source}")
                continue
            keys = nodes_by_property.get(target, [])
            if not keys:
                dropped.append(f"{paper}: {source} -> {target}")
                continue
            if (paper, source) not in variables:
                column = table[source]
                variables[(paper, source)] = {
                    "type": (
                        "number" if pd.api.types.is_numeric_dtype(column) else "string"
                    ),
                    # convert_dtypes keeps an integer column with blanks integer,
                    # so its values read 64, not 64.0.
                    "values": [
                        str(value)
                        for value in column.convert_dtypes().dropna().unique()
                    ],
                    "keys": set(),
                }
            variables[(paper, source)]["keys"].update(keys)
    if missing:
        raise ValueError(
            f"{len(missing)} mapped variables are not columns of their paper's "
            "table: " + "; ".join(missing[:12])
        )
    if dropped:
        print(f"  {len(dropped)} published mappings have no target in this dictionary:")
        for mapping in dropped:
            print(f"    {mapping}")

    return [
        {
            "source_study_id": "GDC",
            "source_table_id": paper,
            "source_variable_id": "",
            "source_variable_name": source,
            "source_type": entry["type"],
            # GDC-SM describes no variable, so the description is left empty.
            "source_description": "",
            "source_values": VALUE_SEPARATOR.join(entry["values"]),
            "expected_target": sorted(entry["keys"]),
        }
        for (paper, source), entry in sorted(variables.items())
    ]


def build(downloads, manifest):
    """Build the GDC benchmark from the inputs its manifest pins.

    The ground truth and the publishers' spreadsheets are kept between builds.
    The dictionary is fetched each time, since the GDC API serves only the
    current one; the manifest records its SHA-256, so a build on a changed
    dictionary shows as a different benchmark.

    Args:
        downloads (Path): Where downloaded files are kept between builds.
        manifest (dict): The committed manifest; its ground-truth source gives
            the Zenodo record.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: The targets, the mappings
            and the manifest's sources.
    """
    ground_truth = next(
        (s for s in manifest["sources"] if s["name"] == GROUND_TRUTH), None
    )
    if ground_truth is None:
        raise KeyError(f"the manifest has no source named {GROUND_TRUTH!r}")
    record = ground_truth["version"]
    archive = cached(
        ZENODO_FILE.format(record=record, name="gdc-sm-data.zip"),
        downloads / record / "gdc-sm-data.zip",
    )
    papers = json.loads(
        cached(
            ZENODO_FILE.format(record=record, name="papers_info.json"),
            downloads / record / "papers_info.json",
        )
    )
    truth = read_ground_truth(archive)

    tables, sources = {}, []
    for name, info in sorted(papers.items()):
        paper = name.removesuffix(".csv")
        spreadsheet = cached(
            info["Dataset URL"], downloads / PurePosixPath(info["Dataset URL"]).name
        )
        tables[paper] = read_source_table(spreadsheet, info["Sheet Name"], paper)
        sources.append(
            {
                "role": "source text",
                "name": f"{paper}, supplementary table",
                "location": info["Dataset URL"],
                "sha256": sha256(spreadsheet),
                "citation": info["Paper URL"],
            }
        )

    status = json.loads(fetch(GDC_STATUS))
    dictionary = json.loads(fetch(GDC_DICTIONARY))
    targets = dictionary_targets(dictionary)
    mappings = mapping_rows(truth, tables, targets)
    print(
        f"  {len(truth)} papers, {len(mappings)} source variables, "
        f"{len(targets):,} targets ({status['data_release']})"
    )
    sources = [
        {**ground_truth, "sha256": sha256(archive)},
        *sources,
        {
            "role": "targets",
            "name": "GDC submission dictionary",
            "location": GDC_DICTIONARY,
            "version": (
                f"{status['data_release']}; API {status['tag']}, "
                f"commit {status['commit']}"
            ),
            "fetched": date.today().isoformat(),
            # Of the content, keys sorted, so a reordering by the API is not a
            # new dictionary.
            "sha256": sha256(json.dumps(dictionary, sort_keys=True).encode()),
            "license": "Apache-2.0",
            "citation": "embedding_model/docs/benchmarks.md#gdc",
        },
    ]
    return targets, mappings, sources
