"""The GDC benchmark.

Variables from the supplementary tables of ten cancer papers, mapped by
experts to the GDC data dictionary: the ground truth of GDC-SM, the schema
matching benchmark published with Magneto. GDC-SM distributes the mappings
only, so the source tables are downloaded from their publishers and the
targets from the GDC submission dictionary that the GDC API serves. Each
variable's study is the dbGaP study of its paper's cohort.
"""

import io
import json
import zipfile
from pathlib import PurePosixPath

import pandas as pd

from ai_harmonization.formatters import VALUE_SEPARATOR
from ai_harmonization.simple_data_model import SimpleDataModel
from embedding_model import manifests
from embedding_model.download import cached, fetch, keep, modified
from embedding_model.manifests import sha256

# The manifest source that pins the ground truth, by its Zenodo record.
GROUND_TRUTH = "GDC-SM ground truth"
ZENODO_FILE = "https://zenodo.org/records/{record}/files/{name}?download=1"
GDC_DICTIONARY = "https://api.gdc.cancer.gov/v0/submission/_dictionary/_all"
GDC_STATUS = "https://api.gdc.cancer.gov/status"
# The status's fields that the manifest records as the dictionary's version.
STATUS_FIELDS = ("data_release", "tag", "commit")
# Entries the API serves beside the nodes: the JSON meta-schema and the root.
NOT_NODES = {"metaschema", "root"}

# The ground truth names a property, not a node. So a property that several
# nodes carry is an expected target in each of them, except where a node gives
# the name another meaning. A project's primary site and disease type describe
# the project, not a patient. A portion's weight is a sample portion's weight
# in milligrams, not a patient's weight.
NOT_EXPECTED = {"project.primary_site", "project.disease_type", "portion.weight"}

# Each paper's dbGaP study, DOI and license, by its name in GDC-SM. A paper's
# patients are the cases of one GDC project, and each project is one dbGaP
# study. CPTAC-2 (breast, colon, ovary) is phs000892. CPTAC-3 (brain, head and
# neck, kidney, lung, pancreas, uterus) is phs001287. This was checked through
# the GDC API on 2026-10-22. Its /cases endpoint gave each case's project, from
# the case ids in each paper's table, which GDC calls submitter_id. All of a
# paper's cases belong to the project given here. Its /projects endpoint gave
# each project's dbGaP study. Crossref gave each paper's DOI from its publisher
# id, and the titles match. Each paper's Crossref record gives its license,
# written here as an SPDX license expression.
PAPERS = {
    "Cao": {
        "study": "phs001287",
        "doi": "10.1016/j.cell.2021.08.023",
        "license": "CC-BY-4.0",
    },
    "Clark": {
        "study": "phs001287",
        "doi": "10.1016/j.cell.2019.10.007",
        "license": "CC-BY-4.0",
    },
    "Dou": {
        "study": "phs001287",
        "doi": "10.1016/j.cell.2020.01.026",
        "license": "CC-BY-4.0",
    },
    "Gillette": {
        "study": "phs001287",
        "doi": "10.1016/j.cell.2020.06.013",
        "license": "CC-BY-NC-ND-4.0",
    },
    "Huang": {
        "study": "phs001287",
        "doi": "10.1016/j.ccell.2020.12.007",
        "license": "CC-BY-NC-ND-4.0",
    },
    "Krug": {
        "study": "phs000892",
        "doi": "10.1016/j.cell.2020.10.036",
        "license": "CC-BY-NC-ND-4.0",
    },
    "McDermott": {
        "study": "phs000892",
        "doi": "10.1016/j.xcrm.2020.100004",
        "license": "CC-BY-NC-ND-4.0",
    },
    "Satpathy": {
        "study": "phs001287",
        "doi": "10.1016/j.cell.2021.07.016",
        "license": "CC-BY-NC-ND-4.0",
    },
    "Vasaikar": {
        "study": "phs000892",
        "doi": "10.1016/j.cell.2019.03.030",
        "license": "CC-BY-NC-ND-4.0",
    },
    "Wang": {
        "study": "phs001287",
        "doi": "10.1016/j.ccell.2021.01.006",
        "license": "CC-BY-NC-ND-4.0",
    },
}


def dictionary_targets(dictionary):
    """One target per data property in the GDC dictionary, as ``node.property``.

    The dictionary is read by ``SimpleDataModel.from_gdc_model``, as the
    harmonization approaches read it, so a target's text is the text that they
    build. Of its properties, these are not targets: links (``anyOf``), which
    point to other nodes rather than hold data; the bookkeeping fields that
    each node lists as ``systemProperties`` -- ``id``, ``created_datetime``,
    ``state`` -- and every node's ``type``, which holds the node's own name.
    The JSON meta-schema and the root, which the API serves beside the nodes,
    are not targets either.

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


def require_papers(listed, mapped, papers):
    """Stop when a paper lacks its source table or its study and DOI.

    Args:
        listed (Iterable[str]): The papers with a table in GDC-SM's
            ``papers_info.json``, by name, such as ``Clark``.
        mapped (Iterable[str]): The papers that the ground truth maps.
        papers (dict): Each paper's dbGaP study and DOI, as in PAPERS.

    Raises:
        ValueError: Naming each mapped paper whose table is not listed, and
            each paper without an entry in ``papers``.
    """
    listed, mapped = set(listed), set(mapped)
    problems = []
    no_table = sorted(mapped - listed)
    if no_table:
        problems.append(
            "the ground truth maps papers with no listed table: " + ", ".join(no_table)
        )
    no_entry = sorted((listed | mapped) - set(papers))
    if no_entry:
        problems.append(
            "these papers have no dbGaP study and DOI: " + ", ".join(no_entry)
        )
    if problems:
        raise ValueError("; ".join(problems))


def paper_source(paper, info, spreadsheet, papers):
    """The manifest entry for a paper's supplementary table, cited by its DOI.

    Args:
        paper (str): The paper's name in GDC-SM, such as ``Clark``.
        info (dict): The paper's entry in ``papers_info.json``.
        spreadsheet (bytes): The table's file, as downloaded.
        papers (dict): Each paper's dbGaP study, DOI and license, as in PAPERS.

    Returns:
        dict: The source, keyed like the manifest's other sources.
    """
    return {
        "role": "source text",
        "name": f"{paper}, supplementary table",
        "location": info["Dataset URL"],
        "sha256": sha256(spreadsheet),
        "license": papers[paper]["license"],
        "citation": f"https://doi.org/{papers[paper]['doi']}",
    }


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
    publisher's sheet, repairs four of the tables and saves each as CSV. The
    same steps are taken here: Vasaikar's sheet loses its first row, of type
    labels (BIN, CON, ...), and its last, of counts; a tab is removed from a
    McDermott column name; a typo in Clark's ethnicity values is corrected; and
    Wang's secondhand-smoke answers are recoded to Yes and No.

    Args:
        spreadsheet (bytes): The publisher's ``.xlsx`` file.
        sheet (str): The sheet holding the table.
        paper (str): The paper's name in GDC-SM, such as ``Vasaikar``.

    Returns:
        pd.DataFrame: The table, with the column types that its CSV is read
            back with.
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


def mapping_rows(truth, tables, targets, papers):
    """One row per source variable, with every target that the experts mapped it to.

    A variable keeps every distinct value that its column holds. The prompt
    formatters cap how many they show. A published mapping is left out and
    named when the dictionary does not carry its property, since a new
    dictionary can drop or rename a property.

    Args:
        truth (dict[str, pd.DataFrame]): The ground truth, from
            read_ground_truth.
        tables (dict[str, pd.DataFrame]): The source tables, by paper.
        targets (list[dict]): The dictionary's targets, from
            dictionary_targets.
        papers (dict): Each paper's dbGaP study and DOI, as in PAPERS.

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
        table = tables[paper]
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
            "source_study_id": papers[paper]["study"],
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


def kept_dictionary(downloads):
    """The GDC API's status and dictionary, fetched once and kept together.

    Both are fetched again when either one is missing, so the kept status
    always names the release of the kept dictionary. Both responses are read
    before either is kept. A response that is not JSON, or a status without
    STATUS_FIELDS, is never kept, so the next build tries again.

    Args:
        downloads (Path): Where the GDC benchmark's downloads are kept.

    Returns:
        tuple[dict, dict, Path]: The status, the dictionary, and the kept
            dictionary's file.

    Raises:
        ValueError: If the status lacks one of STATUS_FIELDS.
    """
    files = {
        GDC_STATUS: downloads / "dictionary" / "status.json",
        GDC_DICTIONARY: downloads / "dictionary" / "dictionary.json",
    }
    if not all(path.exists() for path in files.values()):
        served = {url: fetch(url) for url in files}
        status = json.loads(served[GDC_STATUS])
        json.loads(served[GDC_DICTIONARY])
        missing = [field for field in STATUS_FIELDS if field not in status]
        if missing:
            raise ValueError(
                f"{GDC_STATUS} answered without {', '.join(missing)}. Nothing "
                "was kept; try again later."
            )
        for url, path in files.items():
            keep(path, served[url])
    status, dictionary = (json.loads(path.read_bytes()) for path in files.values())
    return status, dictionary, files[GDC_DICTIONARY]


def build(downloads, manifest):
    """Build the GDC benchmark from the inputs that its manifest pins.

    The ground truth, the publishers' spreadsheets and the dictionary are kept
    between builds. The GDC API serves only the current dictionary, so the
    kept copy is the only copy of the dictionary that the manifest pins. The
    API's status, which names the data release, is kept with it, as
    ``kept_dictionary`` says. The manifest records the dictionary's SHA-256,
    so a build on a changed dictionary shows as a different benchmark.

    Args:
        downloads (Path): Where downloaded files are kept between builds.
        manifest (dict): The committed manifest; its ground-truth source gives
            the Zenodo record.

    Returns:
        tuple[list[dict], list[dict], list[dict], dict]: The targets, the
            mappings, the manifest's sources, and the counts that the build
            prints.
    """
    ground_truth = manifests.pinned(manifest, GROUND_TRUTH)
    record = ground_truth["version"]
    archive = cached(
        ZENODO_FILE.format(record=record, name="gdc-sm-data.zip"),
        downloads / record / "gdc-sm-data.zip",
    )
    papers_info = json.loads(
        cached(
            ZENODO_FILE.format(record=record, name="papers_info.json"),
            downloads / record / "papers_info.json",
        )
    )
    truth = read_ground_truth(archive)
    require_papers((name.removesuffix(".csv") for name in papers_info), truth, PAPERS)

    tables, sources = {}, []
    for name, info in sorted(papers_info.items()):
        paper = name.removesuffix(".csv")
        spreadsheet = cached(
            info["Dataset URL"], downloads / PurePosixPath(info["Dataset URL"]).name
        )
        tables[paper] = read_source_table(spreadsheet, info["Sheet Name"], paper)
        sources.append(paper_source(paper, info, spreadsheet, PAPERS))

    status, dictionary, dictionary_file = kept_dictionary(downloads)
    targets = dictionary_targets(dictionary)
    mappings = mapping_rows(truth, tables, targets, PAPERS)
    counts = {
        "papers": len(truth),
        "source_variables": len(mappings),
        "targets": len(targets),
    }
    print(
        f"  {counts['papers']} papers, {counts['source_variables']} source "
        f"variables, {counts['targets']:,} targets ({status['data_release']})"
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
            "fetched": modified(dictionary_file),
            # The SHA-256 is of the content, with keys sorted, so a reordering
            # by the API is not a new dictionary.
            "sha256": sha256(json.dumps(dictionary, sort_keys=True).encode()),
            "license": "Apache-2.0",
            "citation": "embedding_model/docs/licenses.md#gdc",
        },
    ]
    return targets, mappings, sources, counts
