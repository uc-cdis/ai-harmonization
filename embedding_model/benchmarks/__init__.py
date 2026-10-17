"""Benchmarks: expert mappings from study variables to data-model properties.

Each benchmark is two CSV files under ``<inputs>/benchmarks/``:

- ``<benchmark>_mappings.csv``: one row per source variable, with the keys of
  its expected targets, separated by ``;`` where there are several;
- ``<benchmark>_targets.csv``: every target property a source is ranked
  against, the expected ones and all the others.

The files carry third-party text, so they are built from their published
inputs rather than redistributed. A manifest committed under ``manifests/``
pins those inputs -- a commit, a record, a release -- and records the SHA-256
of every file built from them: a rebuild either matches it or names what
differs.
"""

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

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


def write_benchmark(directory, name, version, targets, mappings, sources):
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
    }
    (directory / f"{name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def read_benchmark(inputs, name):
    """A built benchmark: its targets, its mappings and its manifest.

    Each file must have the SHA-256 its manifest records. So scores made from
    the files can name exactly what was scored. Each file is read once, and
    the bytes checked are the bytes parsed. A warning is printed when the
    build's version is not the version committed under ``manifests/``.

    Args:
        inputs (Path): The inputs folder the benchmark was built into.
        name (str): The benchmark, such as ``topmed``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: The targets, the mappings and
            the build's manifest. Every cell is a string, and an empty cell is
            ``""``.

    Raises:
        FileNotFoundError: If the benchmark has not been built there.
        ValueError: If a file differs from its manifest, or the manifest
            cannot be read.
    """
    directory = Path(inputs) / "benchmarks"
    paths = [directory / f"{name}_{kind}.csv" for kind in ("targets", "mappings")]
    manifest_path = directory / f"{name}_manifest.json"
    command = f"`python -m embedding_model benchmarks build --{name}`"
    missing = [path for path in [*paths, manifest_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing[0]} is missing: build the benchmark first with {command}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(
            f"{manifest_path} cannot be read ({error}): rebuild with {command}"
        ) from error
    committed_path = MANIFESTS / f"{name}.json"
    if committed_path.exists():
        committed = json.loads(committed_path.read_text(encoding="utf-8"))["version"]
        if committed != manifest["version"]:
            print(
                f"warning: {name} in {directory} is version {manifest['version']}, "
                f"and the committed manifest is version {committed}. Rebuild "
                f"with {command} to score version {committed}.",
                file=sys.stderr,
            )
    data = {path: path.read_bytes() for path in paths}
    changed = [
        path.name
        for path in paths
        if sha256(data[path]) != manifest["files"].get(path.name, {}).get("sha256")
    ]
    if changed:
        raise ValueError(
            f"{manifest_path.name} records another SHA-256 for "
            f"{', '.join(changed)}: rebuild with {command}"
        )
    targets, mappings = (
        pd.read_csv(io.BytesIO(data[path]), dtype=str, keep_default_na=False)
        for path in paths
    )
    return targets, mappings, manifest


def identities(manifest):
    """What a build is compared by: its files, then its sources.

    A file is identified by its rows and SHA-256, and a source by its SHA-256,
    or by its version where it has none.

    Args:
        manifest (dict): A benchmark manifest.

    Returns:
        dict: One entry per file, by name, and per source, by role and name.
    """
    return {
        **manifest["files"],
        **{
            f"{source['role']} {source['name']!r}": source.get("sha256")
            or source.get("version")
            for source in manifest["sources"]
        },
    }


def differences(committed, built):
    """What differs between a committed manifest and a new build's.

    A source is identified by its SHA-256 where it has one, and otherwise by
    its version, so a dictionary served under a new release label but with the
    same content is the same source. Dates are not compared.

    Args:
        committed (dict): The manifest under ``manifests/``.
        built (dict): The manifest of the build just made.

    Returns:
        list[str]: One line per difference; empty when the two match.
    """
    before, after = identities(committed), identities(built)
    keys = [*before, *(key for key in after if key not in before)]
    return [
        f"{key}: {before.get(key)} -> {after.get(key)}"
        for key in keys
        if before.get(key) != after.get(key)
    ]


def committed_version(name):
    """The version of a benchmark's manifest as committed in git, at ``HEAD``.

    Args:
        name (str): The benchmark, as in BUILDERS.

    Returns:
        str | None: The version, or None when the manifest is not committed
            yet.

    Raises:
        RuntimeError: If git cannot be asked: outside a checkout, with git
            missing, or when git refuses the folder.
    """
    git = ["git", "-C", str(MANIFESTS)]
    try:
        inside = subprocess.run(
            [*git, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        )
        shown = subprocess.run(
            [*git, "show", f"HEAD:./{name}.json"], capture_output=True, text=True
        )
    except Exception as error:
        raise RuntimeError(f"git could not run: {error}") from error
    if inside.returncode != 0:
        raise RuntimeError(inside.stderr.strip() or "not a git checkout")
    if shown.returncode != 0:
        return None
    return json.loads(shown.stdout)["version"]


def build(names, inputs, update_manifest=False):
    """Build benchmarks from the inputs their committed manifests pin.

    Each build is written to a staging folder first. If it matches the
    committed manifest, or ``update_manifest`` accepts it, its files replace
    the working ones; if not, the differences are printed, the working files
    are left as they were, and the exit status is 1. A version committed in
    git never changes: a differing build is accepted only under a version not
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
        targets, mappings, sources = BUILDERS[name].build(
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
            )
            found = differences(committed, built)
            refusal = None
            if found and not update_manifest:
                refusal = "pass --update-manifest to accept this build"
            elif found:
                try:
                    at_head = committed_version(name)
                except RuntimeError as error:
                    refusal = (
                        f"git cannot say whether version {committed['version']} "
                        f"is committed ({error}); fix that to accept this build"
                    )
                else:
                    if at_head == committed["version"]:
                        refusal = (
                            f"version {committed['version']} is committed; raise "
                            f'"version" in {committed_path} to accept this '
                            "build as a new version"
                        )
            if refusal:
                print(f"  differs from {committed_path.name}; working files kept:")
                for line in found:
                    print(f"    {line}")
                print(f"  {refusal}")
                status = 1
                continue
            for file in [*built["files"], f"{name}_manifest.json"]:
                shutil.move(Path(staging) / file, directory / file)
        for file, entry in built["files"].items():
            print(f"  {directory / file}: {entry['rows']:,} rows")
        if found:
            committed_path.write_text(
                json.dumps(built, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  accepted, and recorded in {committed_path}")
        else:
            print(f"  matches {committed_path.name}")
    return status
