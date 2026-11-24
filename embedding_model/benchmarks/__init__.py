"""Benchmarks: expert mappings from study variables to data-model properties.

Each benchmark is two CSV files under ``<inputs>/benchmarks/<benchmark>/``,
beside the manifest of its build:

- ``<benchmark>_mappings.csv``: one row per source variable, with the keys of
  its expected targets, separated by ``;`` where there are several
- ``<benchmark>_targets.csv``: every target property that ``score`` ranks for
  a source variable, the expected ones and all the others

The files carry third-party text, so ``benchmarks build`` builds them from
their published inputs, and the repository does not redistribute them. A
manifest committed under ``manifests/`` pins those inputs, for example a
commit, a record or a release. It also holds the SHA-256 of every file built
from them. So a rebuild either matches it or names what differs.
"""

import csv
from collections import Counter
import io
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from embedding_model import manifests
from embedding_model.benchmarks import gdc, topmed
from embedding_model.manifests import sha256

MAPPING_COLUMNS = [
    "source_study_id",
    "source_table_id",
    "source_variable_id",
    "source_variable_name",
    "source_type",
    "source_description",
    "source_values",
    "expected_target",
]
TARGET_COLUMNS = [
    "target_property",
    "target_type",
    "target_description",
    "target_values",
]
TARGET_SEPARATOR = ";"
MANIFESTS = Path(__file__).resolve().parent / "manifests"
BUILDERS = {"topmed": topmed, "gdc": gdc}
# Each benchmark's name as its publishers write it, for figures and reports.
TITLES = {"topmed": "TOPMed", "gdc": "GDC"}


def write_benchmark(directory, name, version, targets, mappings, sources, counts):
    """Write a benchmark's two files and the manifest of this build.

    Args:
        directory (Path): Where the files go.
        name (str): The benchmark's name, the stem of its files.
        version (str): The benchmark's version.
        targets (list[dict]): One row per target, keyed by TARGET_COLUMNS.
        mappings (list[dict]): One row per source variable, keyed by
            MAPPING_COLUMNS. Each row's ``expected_target`` is a list of target
            keys. write_benchmark sorts the keys and joins them by
            TARGET_SEPARATOR.
        sources (list[dict]): The benchmark's inputs.
            ``embedding_model/docs/benchmarks.md`` describes their fields.
        counts (dict): What the build counted, as the benchmark's module prints
            the counts.

    Returns:
        dict: The manifest, also written as ``manifest.json`` beside the
            files.

    Raises:
        ValueError: If a property is in the targets twice, or an expected
            target of the mappings is not in the targets.
    """
    listed = Counter(row["target_property"] for row in targets)
    repeated = sorted(key for key, count in listed.items() if count > 1)
    if repeated:
        raise ValueError(f"{name}'s targets list a property twice: {repeated}")
    missing = sorted(
        {key for row in mappings for key in row["expected_target"]} - set(listed)
    )
    if missing:
        raise ValueError(
            f"{name}'s expected targets are not among its targets: {missing}"
        )
    mappings = [
        {
            **row,
            "expected_target": TARGET_SEPARATOR.join(sorted(row["expected_target"])),
        }
        for row in mappings
    ]
    files = {}
    for kind, columns, rows in (
        ("mappings", MAPPING_COLUMNS, mappings),
        ("targets", TARGET_COLUMNS, targets),
    ):
        path = directory / f"{name}_{kind}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        files[path.name] = {"rows": len(rows), "sha256": sha256(path.read_bytes())}
    manifest = {
        "benchmark": name,
        "version": version,
        "built": date.today().isoformat(),
        "files": files,
        "sources": sources,
        "counts": counts,
    }
    manifests.write(directory / "manifest.json", manifest)
    return manifest


def read_benchmark(inputs, name):
    """A built benchmark: its targets, its mappings and its manifest.

    Each file must have the SHA-256 that the build's manifest holds. So scores
    made from the files can name exactly what was scored. read_benchmark reads
    each file once, and parses the same bytes that it checks. It prints a
    warning when the build's version is not the version committed under
    ``manifests/``.

    Args:
        inputs (Path): The inputs folder that holds the built benchmark.
        name (str): The benchmark, for example ``topmed``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: The targets, the mappings and
            the build's manifest. Every cell is a string, and an empty cell is
            ``""``.

    Raises:
        FileNotFoundError: If the benchmark is not built there.
        ValueError: In each of these cases:

            - the manifest cannot be read, is not a JSON object, or names
              another benchmark
            - the manifest does not record the benchmark's version and files
            - a file differs from the manifest
            - the files do not have the columns that ``benchmarks build``
              writes
    """
    directory = Path(inputs) / "benchmarks" / name
    paths = [directory / f"{name}_{kind}.csv" for kind in ("targets", "mappings")]
    manifest_path = directory / "manifest.json"
    command = f"python -m embedding_model benchmarks build --{name}"
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: {command}")
    manifest = manifests.read(manifest_path, command, "benchmark", name)
    try:
        version = manifest["version"]
        recorded = {path.name: manifest["files"][path.name]["sha256"] for path in paths}
    except Exception as error:
        raise ValueError(
            f"{manifest_path} does not record the benchmark's version and files"
            f" ({type(error).__name__}: {error}). Run: {command}"
        ) from error
    manifests.warn_if_not_committed(
        f"{name} in {directory}", version, MANIFESTS / f"{name}.json", command
    )
    file_bytes = {path: path.read_bytes() for path in paths}
    changed = [
        path.name for path in paths if sha256(file_bytes[path]) != recorded[path.name]
    ]
    if changed:
        raise ValueError(
            f"{manifest_path.name} holds another SHA-256 for"
            f" {', '.join(changed)}. Run: {command}"
        )
    targets, mappings = (
        pd.read_csv(io.BytesIO(file_bytes[path]), dtype=str, keep_default_na=False)
        for path in paths
    )
    columns = (list(targets.columns), list(mappings.columns))
    if columns != (TARGET_COLUMNS, MAPPING_COLUMNS):
        raise ValueError(
            f"the files in {directory} do not have the columns that"
            f" `benchmarks build` writes. Run: {command}"
        )
    return targets, mappings, manifest


def build(names, inputs, update_manifest=False):
    """Build benchmarks from the inputs that their committed manifests pin.

    Each build goes to a staging folder first. If the build matches the
    committed manifest, or ``update_manifest`` accepts it, its folder replaces
    the working one. If not, ``manifests.decide`` prints the differences,
    and the working folder stays as it was. Then the exit status
    is 1.

    A version committed in git never changes. So a differing build is accepted
    only under a version not committed yet, and not at all when git cannot
    answer. If a benchmark cannot be built, for example because its download
    fails, build prints the error and still builds the other benchmarks.

    Args:
        names (list[str]): Benchmarks to build, from BUILDERS.
        inputs (Path): The inputs folder. The files go to its ``benchmarks/``.
        update_manifest (bool): Accept a build that differs from its
            committed manifest, under a version not committed in git, and
            record the build in that manifest.

    Returns:
        int: 0 if every build matches or was accepted, 1 otherwise.
    """
    directory = Path(inputs) / "benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    status = 0
    for name in names:
        print(f"{name}:")
        try:
            status |= build_one(name, directory, update_manifest)
        except Exception as error:
            print(
                f"  {name} not built: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            status = 1
    return status


def build_one(name, directory, update_manifest):
    """Build one benchmark, and replace its working folder if the build may be kept.

    ``manifests.write_checked`` writes the files, and keeps them only if the
    committed manifest allows.

    Args:
        name (str): The benchmark, from BUILDERS.
        directory (Path): The inputs folder's ``benchmarks/``.
        update_manifest (bool): As in ``build``.

    Returns:
        int: 0 if the build matches or was accepted, 1 otherwise.
    """
    committed_path = MANIFESTS / f"{name}.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    targets, mappings, sources, counts = BUILDERS[name].build(
        directory / "downloads" / name, committed
    )
    return manifests.write_checked(
        directory / name,
        lambda built: write_benchmark(
            built, name, committed["version"], targets, mappings, sources, counts
        ),
        committed_path,
        update_manifest,
        f"python -m embedding_model benchmarks build --{name}",
    )
