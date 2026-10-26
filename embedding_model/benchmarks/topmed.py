"""The TOPMed benchmark.

dbGaP study variables mapped by the experts of the TOPMed Data Coordinating
Center to its harmonized variables. For each harmonized variable, the
documentation lists its harmonization units. A unit is a study or a subcohort
of one. For each unit, it lists the dbGaP variables that the unit's
harmonization function reads: "at least one representing the phenotype itself,
and at least one for the age at measurement", in the documentation's words. It
cites each variable by accession, release included, so the variable's text
comes from the data dictionaries of that dbGaP release, which need no
controlled access.

The targets are the harmonized schema, written ``<node>.<property>``. The node
is the dataset whose folder holds the variable's documentation
(``atherosclerosis``). The property is the phenotype (``cac_score_1``), or the
age at measurement (``age_at_cac_score_1``) where the variable has one. A
component variable maps to the age when its dbGaP description says "age", and
to the phenotype otherwise.
"""

import io
import json
import re
import tarfile

from ai_harmonization.dbgap import map_dbgap_type, parse_dbgap_table
from ai_harmonization.formatters import VALUE_SEPARATOR
from embedding_model.download import cached, downloaded

# The manifest source that pins the documentation, by its commit.
DOCUMENTATION = "TOPMed DCC harmonized phenotype documentation"
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
        list[dict]: One per harmonized variable, sorted by dataset and name:
            its ``dataset``, ``name``, ``type``, ``description``, ``values``,
            ``has_age`` and ``components`` -- the dbGaP references that its
            units cite, each a tuple (phs, release, pht, table version, phv).
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
                        str(entry["value"])
                        for entry in document.get("encoded_values") or []
                    ],
                    "has_age": bool(document.get("has_age_variable")),
                    "components": references,
                }
            )
    return sorted(variables, key=lambda v: (v["dataset"], v["name"]))


def target_rows(variables):
    """The targets: each harmonized variable's phenotype and, if it has one, age.

    The age target is described the way the BioData Catalyst data model
    describes the same properties: "age at measurement of <name> (years)".
    Types are spelled as ``ai_harmonization.dbgap`` spells the sources' types,
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

    The documentation names each component variable only by accession, so its
    text comes from the table that holds it. The table is read with
    ``parse_dbgap_table``, as any study's table is. Only the cited variables
    are kept, each as a mapping row still without its target.
    resolve_components calls this once per cited table.

    Args:
        dictionary (Path): The table's ``data_dict.xml``.
        report (Path | None): Its ``var_report.xml``, which gives the types.
        study (str): The study accession, such as ``phs999999``.
        wanted (set[str]): The phv accessions cited from this table, without
            their version, such as ``phv99999901``.

    Returns:
        dict[str, dict]: Each cited variable that the table has, by phv
            accession, keyed by MAPPING_COLUMNS without ``expected_target``.
    """
    table_id, model = parse_dbgap_table(
        str(dictionary), str(report) if report else None
    )
    found = {}
    for variable in model.nodes[0].properties:
        accession = (variable.additional_metadata or {}).get("variable_id") or ""
        phv = accession.split(".")[0]
        if phv in wanted:
            found[phv] = {
                "source_study_id": study,
                "source_table_id": table_id,
                "source_variable_id": accession,
                "source_variable_name": variable.name,
                "source_type": variable.type or "",
                "source_description": variable.description or "",
                "source_values": VALUE_SEPARATOR.join(variable.values or []),
            }
    return found


def resolve_components(references, downloads):
    """Look up every component variable in the dbGaP release that its reference cites.

    Only the tables that the references name are downloaded, each with its data
    dictionary and variable report. Two of dbGaP's folder listings are read to
    find them: the study's, for the folder of the release cited (a reference
    gives ``phs999999.v1``, the folder is ``phs999999.v1.p1``), and that
    release's, for the names of its files. A published release never changes,
    so its listing is downloaded once and kept like any file. The study's
    listing gains a folder with each new release, so it is fetched again when
    the release cited is missing from the copy kept.

    Args:
        references (set[tuple]): (phs, release, pht, table version, phv).
        downloads (Path): Where the dbGaP files are kept.

    Returns:
        tuple[dict, list[dict]]: Each variable found, by (phs, release, phv);
            and one manifest source per dbGaP release used.

    Raises:
        LookupError: If a cited release, table or variable is not on dbGaP.
    """
    tables = {}
    for phs, release, pht, table_version, phv in references:
        tables.setdefault((phs, release), {}).setdefault(
            (pht, table_version), set()
        ).add(phv)

    found, sources = {}, []
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
            found.update({(phs, release, phv): row for phv, row in table.items()})
        sources.append(
            {
                "role": "source text",
                "name": f"dbGaP variable summaries, {folder}",
                "location": url,
                "version": folder,
            }
        )
    return found, sources


def mapping_rows(variables, resolved):
    """One row per component variable, with every target it maps to.

    A variable that several units cite is one row with all its targets. For
    example, three cohorts of one study can cite the same age at exam. Where a
    variable is cited from more than one release, the text of the latest
    release cited is used.

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


def build(downloads, manifest):
    """Build the TOPMed benchmark from the inputs that its manifest pins.

    Args:
        downloads (Path): Where downloaded files are kept between builds.
        manifest (dict): The committed manifest; its documentation source
            gives the repository and the commit.

    Returns:
        tuple[list[dict], list[dict], list[dict], dict]: The targets, the
            mappings, the manifest's sources, and the counts that the build
            prints.
    """
    documentation = next(
        (s for s in manifest["sources"] if s["name"] == DOCUMENTATION), None
    )
    if documentation is None:
        raise KeyError(f"the manifest has no source named {DOCUMENTATION!r}")
    repository = documentation["location"].removeprefix("https://github.com/")
    commit = documentation["version"]
    archive = cached(
        f"https://codeload.github.com/{repository}/tar.gz/{commit}",
        downloads / f"{commit}.tar.gz",
    )
    variables = read_documentation(archive)
    references = {ref for variable in variables for ref in variable["components"]}
    resolved, dbgap_sources = resolve_components(references, downloads / "dbgap")
    targets = target_rows(variables)
    mappings = mapping_rows(variables, resolved)
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
        f"{counts['targets']} targets; {counts['component_references']:,} "
        f"component references in {counts['dbgap_releases']} dbGaP releases; "
        f"{counts['source_variables']:,} source variables, "
        f"{counts['mapped_to_an_age']:,} of them mapped to an age"
    )
    return targets, mappings, [documentation, *dbgap_sources], counts
