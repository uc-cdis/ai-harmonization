"""The GDC benchmark.

Variables from the supplementary tables of ten cancer papers, mapped by
experts to the GDC data dictionary: the ground truth of GDC-SM, the schema
matching benchmark published with Magneto. GDC-SM distributes the mappings
only, so the source tables are downloaded from their publishers. The targets
come from a release of gdcdictionary, GDC's data dictionary on GitHub,
pinned by its commit. Each variable's study is the dbGaP study of its paper's
cohort.
"""

import io
import json
import tarfile
import zipfile
from pathlib import PurePosixPath

import pandas as pd
import yaml
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

from ai_harmonization.dbgap import resolve_variable_type
from ai_harmonization.formatters import VALUE_SEPARATOR
from ai_harmonization.simple_data_model import SimpleDataModel
from embedding_model import manifests
from embedding_model.download import cached, github_archive
from embedding_model.manifests import sha256

# The manifest source that pins the ground truth, by its Zenodo record.
GROUND_TRUTH = "GDC-SM ground truth"
ZENODO_FILE = "https://zenodo.org/records/{record}/files/{name}?download=1"
# The manifest source that pins the dictionary: gdcdictionary's repository, and
# the commit of a release.
DICTIONARY = "GDC data dictionary"
# The schemas that the nodes refer to by $ref, and the meta-schema. As in
# gdcdictionary's own loader, they are not nodes, and only what a node refers
# to in them is read.
SHARED_SCHEMAS = {
    "metaschema.yaml",
    "_definitions.yaml",
    "_terms.yaml",
    "_terms_enum.yaml",
}
# The root of GDC's graph, whose one property beside its id and its type is
# the dictionary's schema version, not data.
NOT_NODES = {"root"}

# The ground truth names a property, not a node. So a property that several
# nodes carry is an expected target in each of them, except where a node gives
# the name another meaning. A project's primary site and disease type describe
# the project, not a patient. A portion's weight is a sample portion's weight
# in milligrams, not a patient's weight.
NOT_EXPECTED = {"project.primary_site", "project.disease_type", "portion.weight"}
# The ground truth's names that the dictionary also records under another name.
# GDC-SM maps the papers' sex or gender columns, which hold male or female, to
# gender. The dictionary records a patient's sex at birth in the demographic
# node, with the same values.
ALSO_EXPECTED = {"gender": ["demographic.sex_at_birth"]}

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
    The nodes of NOT_NODES give no targets.

    Args:
        dictionary (dict): The dictionary's nodes by name, each with its
            ``properties``, as ``read_dictionary`` gives them.

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


def paper_source(paper, paper_info, spreadsheet, papers):
    """The manifest entry for a paper's supplementary table, cited by its DOI.

    Args:
        paper (str): The paper's name in GDC-SM, such as ``Clark``.
        paper_info (dict): The paper's entry in ``papers_info.json``.
        spreadsheet (bytes): The table's file, as downloaded.
        papers (dict): Each paper's dbGaP study, DOI and license, as in PAPERS.

    Returns:
        dict: The source, keyed like the manifest's other sources.
    """
    return {
        "role": "source text",
        "name": f"{paper}, supplementary table",
        "location": paper_info["Dataset URL"],
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


def column_type_and_values(column):
    """A source column's type and values, typed as a dbGaP study variable is.

    GDC-SM declares no type, so the type is calculated from the column and
    given to ``resolve_variable_type``, as dbGaP's calculated type is. A
    column of numbers is an ``integer`` or a ``float``, without values: its
    values are measurements, not categories. A column counts as numbers when
    more than half of its values are numbers, so a few values such as
    ``unknown`` do not make it a column of categories. A column of True and
    False is two categories, though pandas counts it as numbers. Any other
    column is ``string/encoded``, as a variable with value codes is. Its
    values are listed most frequent first, so a formatter that shows only
    some shows the most common ones. Values that are equally frequent keep
    their order in the table.

    Args:
        column (pd.Series): The column, as read_source_table gives it.

    Returns:
        tuple[str, list[str]]: The type and the values.
    """
    non_missing = column.dropna()
    numbers = pd.to_numeric(non_missing, errors="coerce").dropna()
    is_numbers = (
        pd.api.types.is_numeric_dtype(column) or len(numbers) > len(non_missing) / 2
    )
    if is_numbers and not pd.api.types.is_bool_dtype(column):
        calculated = "integer" if (numbers % 1 == 0).all() else "decimal"
        return resolve_variable_type(calculated, None, has_codes=False), []
    text = non_missing.astype(str)
    counts = text.groupby(text, sort=False).size()
    values = list(counts.sort_values(ascending=False, kind="stable").index)
    return resolve_variable_type("string", None, has_codes=True), values


def mapping_rows(truth, tables, targets, papers):
    """One row per source variable, with every target that the experts mapped it to.

    A variable's type and values come from its column, as
    column_type_and_values gives them. The expected targets are the property
    that the experts named, in each node that carries it, and the properties
    of ALSO_EXPECTED. A published mapping is left out and named when the
    dictionary does not carry its property, since a new dictionary can drop or
    rename a property.

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
    properties = {target["target_property"] for target in targets}
    nodes_by_property = {}
    for key in sorted(properties - NOT_EXPECTED):
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
            keys = nodes_by_property.get(target, []) + [
                key for key in ALSO_EXPECTED.get(target, []) if key in properties
            ]
            if not keys:
                dropped.append(f"{paper}: {source} -> {target}")
                continue
            if (paper, source) not in variables:
                kind, values = column_type_and_values(table[source])
                variables[(paper, source)] = {
                    "type": kind,
                    "values": values,
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
            "source_type": variable["type"],
            # GDC-SM describes no variable, so the description is left empty.
            "source_description": "",
            "source_values": VALUE_SEPARATOR.join(variable["values"]),
            "expected_target": sorted(variable["keys"]),
        }
        for (paper, source), variable in sorted(variables.items())
    ]


def resolved(schema, resolver):
    """A schema with its ``$ref`` resolved, as gdcdictionary's loader resolves it.

    A ``$ref`` is one reference or a list of them. Each one's definition is
    merged over the schema's own keys, so where both hold a key, the
    definition's wins. A definition's own references are resolved from its
    own file.

    Args:
        schema (dict | list | object): A schema, or a part of one.
        resolver (referencing.Resolver): What looks a reference up, from the
            file that the schema is in.

    Returns:
        dict | list | object: The schema, with no ``$ref`` left.
    """
    if isinstance(schema, list):
        return [resolved(part, resolver) for part in schema]
    if not isinstance(schema, dict):
        return schema
    merged = {key: part for key, part in schema.items() if key != "$ref"}
    references = schema.get("$ref", [])
    for reference in references if isinstance(references, list) else [references]:
        referenced = resolver.lookup(reference)
        merged.update(resolved(referenced.contents, referenced.resolver))
    return {key: resolved(part, resolver) for key, part in merged.items()}


def read_dictionary(archive):
    """GDC's dictionary from a release of gdcdictionary, with its references resolved.

    The release keeps each node as a YAML file under ``schemas/``. The nodes
    refer by ``$ref`` to SHARED_SCHEMAS and to each other, and the
    ``referencing`` library looks each reference up. The result is what
    gdcdictionary's own loader gives, ``GDCDictionary`` in its ``core.py``.

    Args:
        archive (bytes): The release's ``.tar.gz``, as GitHub serves it.

    Returns:
        dict: The nodes by their ``id``, each with its references resolved.
    """
    schemas = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if (
                member.isfile()
                and path.parent.name == "schemas"
                and path.suffix == ".yaml"
            ):
                schemas[path.name] = yaml.safe_load(tar.extractfile(member))
    registry = Registry().with_resources(
        (name, Resource.from_contents(schema, default_specification=DRAFT4))
        for name, schema in schemas.items()
    )
    nodes = {}
    for name, schema in sorted(schemas.items()):
        if name not in SHARED_SCHEMAS:
            nodes[schema["id"]] = resolved(schema, registry.resolver(base_uri=name))
    return nodes


def build(downloads, manifest):
    """Build the GDC benchmark from the inputs that its manifest pins.

    The ground truth, the publishers' spreadsheets and the dictionary's
    release are kept between builds.

    Args:
        downloads (Path): Where downloaded files are kept between builds.
        manifest (dict): The committed manifest; its ground-truth source gives
            the Zenodo record, and its dictionary source gives gdcdictionary's
            repository and commit.

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
    for name, paper_info in sorted(papers_info.items()):
        paper = name.removesuffix(".csv")
        spreadsheet = cached(
            paper_info["Dataset URL"],
            downloads / PurePosixPath(paper_info["Dataset URL"]).name,
        )
        tables[paper] = read_source_table(spreadsheet, paper_info["Sheet Name"], paper)
        sources.append(paper_source(paper, paper_info, spreadsheet, PAPERS))

    release = manifests.pinned(manifest, DICTIONARY)
    commit = release["version"]
    dictionary = read_dictionary(github_archive(release["location"], commit, downloads))
    targets = dictionary_targets(dictionary)
    mappings = mapping_rows(truth, tables, targets, PAPERS)
    counts = {
        "papers": len(truth),
        "source_variables": len(mappings),
        "targets": len(targets),
    }
    print(
        f"  {counts['papers']} papers, {counts['source_variables']} source "
        f"variables, {counts['targets']:,} targets (gdcdictionary {commit[:7]})"
    )
    sources = [{**ground_truth, "sha256": sha256(archive)}, *sources, release]
    return targets, mappings, sources, counts
