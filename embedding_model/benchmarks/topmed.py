"""The TOPMed benchmark.

The benchmark holds dbGaP study variables. The experts of the TOPMed Data
Coordinating Center mapped them to the Center's harmonized variables. For each
harmonized variable, the Center's documentation lists its harmonization units.
A unit is a study or a subcohort of one.

For each unit, the documentation lists the dbGaP variables that the unit's
harmonization function reads. In the documentation's words, there is "at least
one representing the phenotype itself, and at least one for the age at
measurement". The documentation cites each variable by accession, release
included. So the variable's text comes from the data dictionaries of that
dbGaP release, which need no controlled access.

The targets are the harmonized schema, written ``<node>.<property>``. The node
is the dataset whose folder holds the variable's documentation
(``atherosclerosis``). The property is the phenotype (``cac_score_1``), or the
age at measurement (``age_at_cac_score_1``) where the variable has one. A
component variable maps to the age when its dbGaP description says "age", and
to the phenotype otherwise. A reviewed list, ``topmed_corrections.csv``, then
corrects these mappings. It removes the mappings whose variable does not carry
the target's value. It also maps a variable to the age that its unit computes
from the variable.
"""

import csv
import io
import json
import re
import tarfile
from pathlib import Path

from ai_harmonization.dbgap import map_dbgap_type, parse_dbgap_table
from ai_harmonization.formatters import VALUE_SEPARATOR
from embedding_model import manifests
from embedding_model.download import cached, downloaded, github_archive
from embedding_model.manifests import sha256

# The manifest source that pins the documentation, by its commit.
DOCUMENTATION = "TOPMed DCC harmonized phenotype documentation"
# The reviewed corrections to the mappings that the age rule gives, committed
# beside the manifest. Each row excludes a mapping or adds one to an age.
CORRECTIONS = Path(__file__).resolve().parent / "manifests" / "topmed_corrections.csv"
CORRECTION_ACTIONS = ("exclude", "add")
# How the unit uses the variable, as each correction says.
CORRECTION_USES = (
    "selects records",
    "times the event",
    "checks another value",
    "records another concept",
    "not used",
    "computes the age",
)
DOCUMENTATION_FOLDER = "harmonized-variable-documentation"
DBGAP = "https://ftp.ncbi.nlm.nih.gov/dbgap/studies"
REFERENCE = re.compile(r"(phs\d+)\.v(\d+)\.(pht\d+)\.v(\d+)\.(phv\d+)")
# "age" as a word, so "Age at exam 1" counts and "Tumor stage" does not.
AGE_IN_DESCRIPTION = re.compile(r"(?i)\bage\b")


def read_documentation(archive):
    """The harmonized variables in the documentation.

    Args:
        archive (bytes): The documentation repository at its pinned commit, as
            a ``.tar.gz``.

    Returns:
        list[dict]: One dict per harmonized variable, sorted by dataset and
            name, with these keys:

            - ``dataset``, ``name``, ``type``, ``description``, ``values`` and
              ``has_age``
            - ``components``: the dbGaP references that the variable's units
              cite, each a tuple (phs, release, pht, table version, phv)

    Raises:
        ValueError: If a unit cites a component that is not a phs.v.pht.v.phv
            accession. The error names the file.
    """
    variables = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Paths in the archive start with a folder named for the commit,
            # then <documentation folder>/<dataset>/<variable>.json.
            parts = member.name.split("/")[1:]
            path = "/".join(parts)
            if not (
                member.isfile()
                and len(parts) == 3
                and parts[0] == DOCUMENTATION_FOLDER
                and parts[2].endswith(".json")
            ):
                continue
            document = json.load(tar.extractfile(member))
            references = []
            for unit in document["harmonization_units"]:
                for component in unit["component_study_variables"]:
                    match = REFERENCE.search(component)
                    if match is None:
                        raise ValueError(
                            f"{path}: unit {unit['name']} cites {component!r}, "
                            "not a phs.v.pht.v.phv accession"
                        )
                    references.append(match.groups())
            variables.append(
                {
                    "dataset": parts[1],
                    "name": document["name"],
                    "type": document.get("data_type") or "",
                    "description": (document.get("description") or "").strip(),
                    "values": [
                        str(encoded_value["value"])
                        for encoded_value in document.get("encoded_values") or []
                    ],
                    "has_age": bool(document.get("has_age_variable")),
                    "components": references,
                }
            )
    return sorted(variables, key=lambda v: (v["dataset"], v["name"]))


def target_rows(variables):
    """The targets: each harmonized variable's phenotype and, if it has one, age.

    target_rows describes the age target as the BioData Catalyst data model
    describes the same properties: "age at measurement of <name> (years)". It
    spells each type as ``ai_harmonization.dbgap`` spells the sources' types,
    so the two sides of a mapping name a type alike.

    Args:
        variables (list[dict]): From read_documentation.

    Returns:
        list[dict]: The targets, keyed by TARGET_COLUMNS and sorted.
    """
    rows = []
    for variable in variables:
        node, name = variable["dataset"], variable["name"]
        rows.append(
            {
                "target_property": f"{node}.{name}",
                "target_type": map_dbgap_type(variable["type"]) or "string",
                "target_description": variable["description"],
                "target_values": VALUE_SEPARATOR.join(variable["values"]),
            }
        )
        if variable["has_age"]:
            rows.append(
                {
                    "target_property": f"{node}.age_at_{name}",
                    "target_type": map_dbgap_type("decimal"),
                    "target_description": f"age at measurement of {name} (years)",
                    "target_values": "",
                }
            )
    return sorted(rows, key=lambda row: row["target_property"])


def is_age(description):
    """Whether a dbGaP variable's description says it is an age."""
    return bool(AGE_IN_DESCRIPTION.search(description))


def cited_variables(dictionary, report, study, wanted):
    """The cited variables of one dbGaP table, as mapping rows.

    The documentation names each component variable only by accession, so the
    variable's text comes from the table that holds it. ``parse_dbgap_table``
    reads the table, as it reads any study's table. cited_variables keeps only
    the cited variables, each as a mapping row still without its target.
    resolve_components calls this once per cited table.

    Args:
        dictionary (Path): The table's ``data_dict.xml``.
        report (Path | None): The table's ``var_report.xml``, which gives the
            types.
        study (str): The study accession, for example ``phs999999``.
        wanted (set[str]): The phv accessions cited from this table, without
            their version, for example ``phv99999901``.

    Returns:
        dict[str, dict]: Each cited variable that the table has, by phv
            accession, keyed by MAPPING_COLUMNS without ``expected_target``.
    """
    table_id, model = parse_dbgap_table(
        str(dictionary), str(report) if report else None
    )
    variables = {}
    for variable in model.nodes[0].properties:
        accession = (variable.additional_metadata or {}).get("variable_id") or ""
        phv = accession.split(".")[0]
        if phv in wanted:
            variables[phv] = {
                "source_study_id": study,
                "source_table_id": table_id,
                "source_variable_id": accession,
                "source_variable_name": variable.name,
                "source_type": variable.type or "",
                "source_description": variable.description or "",
                "source_values": VALUE_SEPARATOR.join(variable.values or []),
            }
    return variables


def resolve_components(references, downloads):
    """Find every component variable in the dbGaP release that its reference cites.

    resolve_components downloads only the tables that the references name,
    each with its data dictionary and variable report. To find them, it reads
    two of dbGaP's folder listings:

    - the study's listing, for the folder of the release cited. A reference
      gives ``phs999999.v1``, and the folder is ``phs999999.v1.p1``.
    - that release's listing, for the names of its files

    A published release never changes, so resolve_components downloads its
    listing once and keeps it like any file. The study's listing gains a
    folder with each new release. So resolve_components downloads the study's
    listing again when the release cited is missing from the kept copy.

    Args:
        references (set[tuple]): (phs, release, pht, table version, phv).
        downloads (Path): The folder that keeps the dbGaP files.

    Returns:
        tuple[dict, list[dict]]: Two values:

            - each variable found, by (phs, release, phv)
            - one manifest source per dbGaP release used

    Raises:
        LookupError: If a cited release, table or variable is not on dbGaP.
    """
    tables = {}
    for phs, release, pht, table_version, phv in references:
        tables.setdefault((phs, release), {}).setdefault(
            (pht, table_version), set()
        ).add(phv)

    variables, sources = {}, []
    for (phs, release), cited in sorted(tables.items()):
        listing_path = downloads / phs / "listing.html"
        folders = []
        for attempt in ("kept", "fetched"):
            if attempt == "fetched":
                listing_path.unlink(missing_ok=True)
            listing = cached(f"{DBGAP}/{phs}/", listing_path).decode("utf-8")
            folders = sorted(set(re.findall(rf"{phs}\.v{release}\.p\d+", listing)))
            if folders:
                break
        if not folders:
            raise LookupError(f"{phs} release v{release} is not on dbGaP")
        folder = folders[-1]
        url = f"{DBGAP}/{phs}/{folder}/pheno_variable_summaries/"
        names = re.findall(
            r'href="([^"]+\.xml)"',
            cached(url, downloads / folder / "listing.html").decode("utf-8"),
        )
        for (pht, table_version), phvs in sorted(cited.items()):
            # Older releases date their files: `...data_dict_2011_02_07.xml`.
            prefix = f"{phs}.v{release}.{pht}.v{table_version}."
            files = {
                kind: next(
                    (n for n in names if n.startswith(prefix) and kind in n), None
                )
                for kind in ("data_dict", "var_report")
            }
            if files["data_dict"] is None:
                raise LookupError(f"{folder} has no data dictionary for {pht}")
            paths = {}
            for kind, name in files.items():
                if name:
                    paths[kind] = downloads / folder / name
                    downloaded(url + name, paths[kind])
            table = cited_variables(
                paths["data_dict"], paths.get("var_report"), phs, phvs
            )
            missing = phvs - set(table)
            if missing:
                raise LookupError(f"{folder} {pht} lacks {sorted(missing)}")
            variables.update({(phs, release, phv): row for phv, row in table.items()})
        sources.append(
            {
                "role": "source text",
                "name": f"dbGaP variable summaries, {folder}",
                "location": url,
                "version": folder,
                "license": "NONE",
                "citation": "embedding_model/docs/licenses.md#topmed",
            }
        )
    return variables, sources


def mapping_rows(variables, resolved):
    """One row per component variable, with every target it maps to.

    A variable that several units cite is one row with all its targets. For
    example, three cohorts of one study can cite the same age at exam. If the
    documentation cites a variable from more than one release, mapping_rows
    uses the text of the latest release cited.

    Args:
        variables (list[dict]): From read_documentation.
        resolved (dict): From resolve_components.

    Returns:
        list[dict]: The rows, keyed by MAPPING_COLUMNS and sorted by study,
            variable name and accession. ``expected_target`` is a sorted list
            of target keys.
    """
    rows = {}
    for variable in variables:
        for phs, release, _, _, phv in variable["components"]:
            record = resolved[(phs, release, phv)]
            age = variable["has_age"] and is_age(record["source_description"])
            name = f"age_at_{variable['name']}" if age else variable["name"]
            row = rows.setdefault(phv, {"release": -1, "targets": set()})
            if int(release) > row["release"]:
                row.update(record, release=int(release))
            row["targets"].add(f"{variable['dataset']}.{name}")
    return sorted(
        (
            {
                **{k: v for k, v in row.items() if k not in ("release", "targets")},
                "expected_target": sorted(row["targets"]),
            }
            for row in rows.values()
        ),
        key=lambda row: (
            row["source_study_id"],
            row["source_variable_name"],
            row["source_variable_id"],
        ),
    )


def read_corrections(path):
    """The reviewed corrections to the benchmark's mappings.

    Args:
        path (Path): The corrections file, a CSV with ``target``, ``unit``,
            ``source_variable_id``, ``source_variable_name``, ``action``,
            ``use`` and ``reason``.

    Returns:
        list[dict]: The rows, each with every column.

    Raises:
        ValueError: If a row's action is not one of CORRECTION_ACTIONS, or its
            use is not one of CORRECTION_USES.
    """
    with open(path, newline="", encoding="utf-8") as file:
        corrections = list(csv.DictReader(file))
    unknown = [
        f"{row['source_variable_id']} -> {row['target']}: {row['action']!r}, {row['use']!r}"
        for row in corrections
        if row["action"] not in CORRECTION_ACTIONS or row["use"] not in CORRECTION_USES
    ]
    if unknown:
        raise ValueError(
            f"{path.name}: rows with an unknown action or use: " + "; ".join(unknown)
        )
    return corrections


def apply_corrections(rows, corrections, targets):
    """The mapping rows with the reviewed corrections applied.

    An ``exclude`` removes a variable's mapping to a target, and an ``add``
    gives it one. apply_corrections drops a row whose targets are all
    excluded.

    Args:
        rows (list[dict]): From mapping_rows.
        corrections (list[dict]): From read_corrections.
        targets (list[dict]): From target_rows.

    Returns:
        list[dict]: The rows that keep a target, in their order.

    Raises:
        ValueError: If corrections do not match the documentation, in each of
            these cases:

            - an excluded mapping that the documentation does not give
            - an added mapping to a variable or a target that the
              documentation does not have
            - an added mapping that the documentation gives already

            The error gives the number of such corrections, and names the
            first 12.
    """
    expected = {row["source_variable_id"]: set(row["expected_target"]) for row in rows}
    properties = {target["target_property"] for target in targets}
    stale = []
    for correction in corrections:
        mapped = expected.get(correction["source_variable_id"])
        target = correction["target"]
        if correction["action"] == "exclude" and mapped and target in mapped:
            mapped.remove(target)
        elif (
            correction["action"] == "add"
            and mapped is not None
            and target in properties
            and target not in mapped
        ):
            mapped.add(target)
        else:
            stale.append(
                f"{correction['action']} {correction['source_variable_id']} -> {target}"
            )
    if stale:
        raise ValueError(
            f"{len(stale)} corrections do not match the documentation. Review "
            f"{CORRECTIONS}: " + "; ".join(stale[:12])
        )
    return [
        {**row, "expected_target": sorted(expected[row["source_variable_id"]])}
        for row in rows
        if expected[row["source_variable_id"]]
    ]


def build(downloads, manifest):
    """Build the TOPMed benchmark from the inputs that its manifest pins.

    Args:
        downloads (Path): The folder that keeps the downloaded files between
            builds.
        manifest (dict): The committed manifest. Its documentation source
            gives the repository and the commit.

    Returns:
        tuple[list[dict], list[dict], list[dict], dict]: Four values:

            - the targets
            - the mappings
            - the manifest's sources
            - the counts that the build prints
    """
    documentation = manifests.pinned(manifest, DOCUMENTATION)
    variables = read_documentation(
        github_archive(documentation["location"], documentation["version"], downloads)
    )
    references = {ref for variable in variables for ref in variable["components"]}
    resolved, dbgap_sources = resolve_components(references, downloads / "dbgap")
    targets = target_rows(variables)
    corrections = read_corrections(CORRECTIONS)
    mappings = apply_corrections(
        mapping_rows(variables, resolved), corrections, targets
    )
    counts = {
        "harmonized_variables": len(variables),
        "targets": len(targets),
        "component_references": len(references),
        "dbgap_releases": len(dbgap_sources),
        "source_variables": len(mappings),
        "mapped_to_an_age": sum(
            any(".age_at_" in key for key in row["expected_target"]) for row in mappings
        ),
    }
    print(
        f"  {counts['harmonized_variables']} harmonized variables, "
        f"{counts['targets']} targets. {counts['component_references']:,} "
        f"component references in {counts['dbgap_releases']} dbGaP releases. "
        f"{counts['source_variables']:,} source variables, "
        f"{counts['mapped_to_an_age']:,} of them mapped to an age, after "
        f"{len(corrections)} corrections"
    )
    correction_source = {
        "role": "corrections",
        "name": "TOPMed mapping corrections",
        "location": f"embedding_model/benchmarks/manifests/{CORRECTIONS.name}",
        "sha256": sha256(CORRECTIONS.read_bytes()),
        "license": "Apache-2.0",
    }
    return targets, mappings, [documentation, *dbgap_sources, correction_source], counts
