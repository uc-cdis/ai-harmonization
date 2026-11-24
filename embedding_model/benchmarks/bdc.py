"""The BDC benchmark: study variables mapped to the BDC Gen3 data dictionary.

The BDC benchmark is in preparation for publication. The builder reads the
release archive that the committed manifest pins.
"""

import io
import json
import sys
import tarfile
from pathlib import PurePosixPath

from ai_harmonization.simple_data_model import SimpleDataModel
from embedding_model import manifests
from embedding_model.benchmarks.common import target_row
from embedding_model.manifests import sha256

RELEASE = "BDC Gen3 harmonization benchmark"
# The start of the name of the archive's file of studies. The release's version
# follows it, as in ``harmonization_benchmark_bdc_gen3_v2.0_44studies_<date>.jsonl``.
STUDIES_FILE_PREFIX = "harmonization_benchmark_bdc_gen3_v"


def read_studies(archive, release_version):
    """The studies of the release, one dict per line of its file of studies.

    Args:
        archive (bytes): The release archive, a ``.tar.gz``.
        release_version (str): The release's version, for example ``2.0``.

    Returns:
        list[dict]: The studies, in the file's order.

    Raises:
        ValueError: If the archive does not hold exactly one file of studies of
            that version.
    """
    prefix = f"{STUDIES_FILE_PREFIX}{release_version}_"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).name.startswith(prefix)
            and member.name.endswith(".jsonl")
        ]
        if len(members) != 1:
            raise ValueError(
                f"the archive holds {len(members)} files named {prefix}*.jsonl,"
                " not one"
            )
        text = tar.extractfile(members[0]).read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def study_accession(study):
    """A study's dbGaP accession, from its custom_metrics, or None if it has none."""
    return study.get("custom_metrics", {}).get("study_accession")


def study_id(study):
    """A study's id: its dbGaP accession, or else the release's own id for the study."""
    return study_accession(study) or study["study_id"]


def dictionary_targets(studies):
    """One target per property of the dictionary that every study shares, as from_gen3_model reads it.

    Args:
        studies (list[dict]): The studies, from read_studies.

    Returns:
        tuple[list[dict], str]: The targets, keyed by
            ``benchmarks.TARGET_COLUMNS`` and sorted, and the dictionary's
            version.

    Raises:
        ValueError: If the studies do not all share one dictionary.
    """
    dictionary = studies[0]["input_target_model"]
    if any(study["input_target_model"] != dictionary for study in studies):
        raise ValueError("the studies do not all share one target dictionary")
    targets = [
        target_row(node.name, prop)
        for node in SimpleDataModel.from_gen3_model(json.dumps(dictionary)).nodes
        for prop in node.properties
    ]
    targets.sort(key=lambda row: row["target_property"])
    dictionary_version = dictionary.get("_settings", {}).get("_dict_version", "")
    return targets, dictionary_version


def mapping_rows(studies, targets):
    """One row per source variable, with every target that the experts mapped it to.

    mapping_rows reads each study's mappings with
    ``get_harmonization_suggestions_from_harmonized_mapping``, and its source
    model with ``SimpleDataModel.from_simple_json``, as the harmonization
    evaluation reads them.

    Args:
        studies (list[dict]): The studies, from read_studies.
        targets (list[dict]): The targets, from dictionary_targets.

    Returns:
        tuple[list[dict], list[str]]: Two values:

            - the rows, keyed by ``benchmarks.MAPPING_COLUMNS`` and sorted by
              study, table and variable, with ``expected_target`` as a sorted
              list of target keys
            - each mapping left out because its target is not among the
              targets, as ``study: variable -> target``

    Raises:
        ValueError: If a mapping names a source variable that its study's
            source model does not hold.
    """
    from ai_harmonization.harmonization_benchmark import (
        get_harmonization_suggestions_from_harmonized_mapping,
    )

    properties = {target["target_property"] for target in targets}
    variables, left_out = {}, []
    for study in studies:
        source_study_id = study_id(study)
        source_model = SimpleDataModel.from_simple_json(
            json.dumps(study["input_source_model"])
        )
        source_properties = {
            f"{node.name}.{prop.name}": (node.name, prop)
            for node in source_model.nodes
            for prop in node.properties
        }
        suggestions = get_harmonization_suggestions_from_harmonized_mapping(
            study["harmonized_mapping"]
        ).suggestions
        for mapping in suggestions:
            source = f"{mapping.source_node}.{mapping.source_property}"
            target = f"{mapping.target_node}.{mapping.target_property}"
            if source not in source_properties:
                raise ValueError(
                    f"{source_study_id}: the mapping names {source}, which its"
                    " source model does not hold"
                )
            if target not in properties:
                left_out.append(f"{source_study_id}: {source} -> {target}")
                continue
            node_name, prop = source_properties[source]
            variable = variables.setdefault(
                (source_study_id, node_name, prop.name),
                {"type": prop.type, "description": prop.description, "keys": set()},
            )
            variable["keys"].add(target)
    rows = [
        {
            "source_study_id": source_study_id,
            "source_table_id": node_name,
            "source_variable_id": "",
            "source_variable_name": variable_name,
            "source_type": variable["type"],
            "source_description": variable["description"],
            "source_values": "",
            "expected_target": sorted(variable["keys"]),
        }
        for (source_study_id, node_name, variable_name), variable in sorted(
            variables.items()
        )
    ]
    return rows, left_out


def build(downloads, manifest):
    """Build the BDC benchmark from the release archive that its manifest pins.

    The build warns about each study that has no dbGaP accession, since the
    corpus's study rule cannot match such a study.

    Args:
        downloads (Path): The folder that holds the release archive.
        manifest (dict): The committed manifest. Its release source gives the
            archive's file name, version and SHA-256.

    Returns:
        tuple[list[dict], list[dict], list[dict], dict]: Four values:

            - the targets
            - the mappings
            - the manifest's sources
            - the counts that the build prints

    Raises:
        FileNotFoundError: If the archive is not in the downloads folder.
        ValueError: If the archive's SHA-256 is not the pinned one.
    """
    release = manifests.pinned(manifest, RELEASE)
    path = downloads / release["location"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The BDC benchmark is in preparation for"
            " publication, and has no download yet"
        )
    archive = path.read_bytes()
    archive_sha256 = sha256(archive)
    if release.get("sha256") and archive_sha256 != release["sha256"]:
        raise ValueError(
            f"{path.name} has the SHA-256 {archive_sha256}, not the pinned"
            f" {release['sha256']}"
        )
    studies = read_studies(archive, release["version"])
    targets, dictionary_version = dictionary_targets(studies)
    mappings, left_out = mapping_rows(studies, targets)
    ids_without_accession = [
        study["study_id"] for study in studies if not study_accession(study)
    ]
    counts = {
        "studies": len(studies),
        "studies_without_accession": len(ids_without_accession),
        "source_variables": len(mappings),
        "targets": len(targets),
        "mappings_left_out": len(left_out),
    }
    print(
        f"  {counts['studies']} studies, {counts['source_variables']} source"
        f" variables, {counts['targets']} targets (BDC Gen3 dictionary"
        f" {dictionary_version})"
    )
    if ids_without_accession:
        print(
            "warning: studies without a dbGaP accession, which the corpus's"
            f" study rule cannot match: {', '.join(ids_without_accession)}",
            file=sys.stderr,
        )
    if left_out:
        print("  mappings left out, since their target is not among the targets:")
        for mapping_shown in left_out:
            print(f"    {mapping_shown}")
    return targets, mappings, [{**release, "sha256": archive_sha256}], counts
