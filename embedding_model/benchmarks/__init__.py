"""Benchmarks: expert mappings from study variables to data-model properties.

Each benchmark is two CSV files under ``<inputs>/benchmarks/``:

- ``<benchmark>_mappings.csv``: one row per source variable, with the keys of
  its expected targets, separated by ``;`` where there are several;
- ``<benchmark>_targets.csv``: every target property that a source is ranked
  against, the expected ones and all the others.

The files carry third-party text, so they are built from their published
inputs rather than redistributed. A manifest committed under ``manifests/``
pins those inputs, such as a commit, a record or a release. It also records the
SHA-256 of every file built from them. So a rebuild either matches it or names
what differs.
"""

import csv
import io
import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from embedding_model import manifests
from embedding_model.benchmarks import gdc, topmed
from embedding_model.download import sha256

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


def write_benchmark(directory, name, version, targets, mappings, sources, counts):
    """Write a benchmark's two files and the manifest of this build.

    Args:
        directory (Path): Where the files go.
        name (str): The benchmark's name, the stem of its files.
        version (str): The benchmark's version.
        targets (list[dict]): One row per target, keyed by TARGET_COLUMNS.
        mappings (list[dict]): One row per source variable, keyed by
            MAPPING_COLUMNS. Its ``expected_target`` is a list of target keys,
            written sorted and joined by TARGET_SEPARATOR.
        sources (list[dict]): What the benchmark was built from; see
            ``embedding_model/docs/benchmarks.md`` for the fields.
        counts (dict): What the build counted, as its benchmark's module prints
            it.

    Returns:
        dict: The manifest, also written as ``<name>_manifest.json`` beside
            the files.

    Raises:
        TypeError: If a mapping's ``expected_target`` is a string, not a list.
    """
    if any(isinstance(row["expected_target"], str) for row in mappings):
        raise TypeError("a mapping's expected_target must be a list of target keys")
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
    (directory / f"{name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def read_benchmark(inputs, name):
    """A built benchmark: its targets, its mappings and its manifest.

    Each file must have the SHA-256 that its manifest records. So scores made
    from the files can name exactly what was scored. Each file is read once,
    and the bytes checked are the bytes parsed. A warning is printed when the
    build's version is not the version committed under ``manifests/``.

    Args:
        inputs (Path): The inputs folder that the benchmark was built into.
        name (str): The benchmark, such as ``topmed``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: The targets, the mappings and
            the build's manifest. Every cell is a string, and an empty cell is
            ``""``.

    Raises:
        FileNotFoundError: If the benchmark has not been built there.
        ValueError: If the manifest cannot be read or names another
            benchmark, a file differs from its manifest, or the files do not
            have the columns that ``benchmarks build`` writes.
    """
    directory = Path(inputs) / "benchmarks"
    paths = [directory / f"{name}_{kind}.csv" for kind in ("targets", "mappings")]
    manifest_path = directory / f"{name}_manifest.json"
    command = f"python -m embedding_model benchmarks build --{name}"
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run: {command}")
    manifest = manifests.read(manifest_path, command, "benchmark", name)
    try:
        version = manifest["version"]
        recorded = {path.name: manifest["files"][path.name]["sha256"] for path in paths}
    except Exception as error:
        raise ValueError(
            f"{manifest_path} does not record the benchmark's version and files"
            f" ({type(error).__name__}: {error}); run: {command}"
        ) from error
    manifests.warn_if_not_committed(
        f"{name} in {directory}", version, MANIFESTS / f"{name}.json", command
    )
    data = {path: path.read_bytes() for path in paths}
    changed = [path.name for path in paths if sha256(data[path]) != recorded[path.name]]
    if changed:
        raise ValueError(
            f"{manifest_path.name} records another SHA-256 for"
            f" {', '.join(changed)}; run: {command}"
        )
    targets, mappings = (
        pd.read_csv(io.BytesIO(data[path]), dtype=str, keep_default_na=False)
        for path in paths
    )
    columns = (list(targets.columns), list(mappings.columns))
    if columns != (TARGET_COLUMNS, MAPPING_COLUMNS):
        raise ValueError(
            f"{directory}'s {name} files do not have the columns that"
            f" `benchmarks build` writes; run: {command}"
        )
    return targets, mappings, manifest


def build(names, inputs, update_manifest=False):
    """Build benchmarks from the inputs that their committed manifests pin.

    Each build is written to a staging folder first. If it matches the
    committed manifest, or ``update_manifest`` accepts it, its files replace
    the working ones. If not, the differences are printed, the working files
    are left as they were, and the exit status is 1. A version committed in git
    never changes: a differing build is accepted only under a version not
    committed yet, and not at all when git cannot be asked.

    Args:
        names (list[str]): Benchmarks to build, from BUILDERS.
        inputs (Path): The inputs folder; files go to its ``benchmarks/``.
        update_manifest (bool): Accept a build that differs from its
            committed manifest, under a version not committed in git, and
            record it there.

    Returns:
        int: 0 if every build matches or was accepted, 1 otherwise.
    """
    directory = Path(inputs) / "benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    status = 0
    for name in names:
        committed_path = MANIFESTS / f"{name}.json"
        committed = json.loads(committed_path.read_text(encoding="utf-8"))
        print(f"{name}:")
        targets, mappings, sources, counts = BUILDERS[name].build(
            directory / "downloads" / name, committed
        )
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            built = write_benchmark(
                Path(staging),
                name,
                committed["version"],
                targets,
                mappings,
                sources,
                counts,
            )
            status |= manifests.settle(
                committed_path,
                built,
                Path(staging),
                directory,
                [*built["files"], f"{name}_manifest.json"],
                update_manifest,
            )
    return status
