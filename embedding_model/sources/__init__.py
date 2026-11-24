"""Sources: the terminologies and data dictionaries that the training pairs come from.

``fetch`` downloads each source from its publisher into
``<inputs>/sources/downloads/<source>/`` and keeps it there, so reading it
again needs no network. It writes the source's tables as parquet files under
``<inputs>/sources/<source>/``, beside the manifest of the fetch.

A manifest committed under ``manifests/`` pins the release that each source is
read from, as the benchmarks' manifests do. It also holds the SHA-256 of
every table's content. A new fetch either matches it or names what differs.
See ``embedding_model.manifests``.

A fetch keeps every item of a source, including retired, draft and closed
items. The next step, ``pairs prepare``, decides which of them become training
pairs. Each item carries an ``open`` column, set from its source's restriction
field.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from embedding_model import manifests
from embedding_model.manifests import file_sha256, table_sha256
from embedding_model.sources import cadsr, cde, cdisc, loinc, ncit, nhanes, obo, phenx

MANIFESTS = Path(__file__).resolve().parent / "manifests"
FETCHERS = {
    "ncit": ncit,
    "obo": obo,
    "loinc": loinc,
    "cdisc": cdisc,
    "cde": cde,
    "cadsr": cadsr,
    "nhanes": nhanes,
    "phenx": phenx,
}


def write_source(directory, name, version, tables, sources, counts):
    """Write a source's tables as parquet, and the manifest of this fetch.

    Args:
        directory (Path): Where the files go.
        name (str): The source, as in FETCHERS.
        version (str): The source's version.
        tables (dict[str, pd.DataFrame]): The tables, by name.
        sources (list[dict]): What the tables were read from. See
            ``embedding_model/docs/sources.md`` for the fields.
        counts (dict): What the fetch counted, as its source's module prints it.

    Returns:
        dict: The manifest, also written as ``manifest.json``.
    """
    files = {}
    for table, frame in tables.items():
        path = directory / f"{table}.parquet"
        frame.to_parquet(path, index=False)
        files[path.name] = {
            "rows": len(frame),
            "sha256": file_sha256(path),
            "content_sha256": table_sha256(frame),
        }
    manifest = {
        "source": name,
        "version": version,
        "fetched": date.today().isoformat(),
        "files": files,
        "sources": sources,
        "counts": counts,
    }
    manifests.write(directory / "manifest.json", manifest)
    return manifest


def read_source(inputs, name):
    """A fetched source's tables and the manifest of its fetch.

    Each table's file must have the ``sha256`` that its manifest holds, so
    it is unchanged since its fetch wrote it. read_source prints a warning when
    the fetch's version is not the version committed under ``manifests/``.

    Args:
        inputs (Path): The inputs folder that the source was fetched into.
        name (str): The source, as in FETCHERS.

    Returns:
        tuple[dict[str, pd.DataFrame], dict]: The tables, by name, and the
            fetch's manifest.

    Raises:
        FileNotFoundError: If the source, or one of its tables, is not fetched
            there.
        ValueError: In each of these cases:

            - the manifest cannot be read, is not a JSON object, or names
              another source
            - the manifest does not record the fetch's version and tables
            - the manifest holds no SHA-256 for a table
            - a table cannot be read, or differs from its manifest
    """
    directory = Path(inputs) / "sources" / name
    manifest_path = directory / "manifest.json"
    command = f"python -m embedding_model sources fetch --source {name}"
    manifest = manifests.read(manifest_path, command, "source", name)
    try:
        version = manifest["version"]
        files = list(manifest["files"])
    except Exception as error:
        raise ValueError(
            f"{manifest_path} does not record the fetch's version and tables"
            f" ({type(error).__name__}: {error}). Run: {command}"
        ) from error
    manifests.warn_if_not_committed(
        f"{name} in {directory}", version, MANIFESTS / f"{name}.json", command
    )
    tables = {}
    for file in files:
        path = directory / file
        manifests.check_file(path, manifest, manifest_path, command)
        try:
            frame = pd.read_parquet(path)
        except Exception as error:
            raise ValueError(
                f"{path} cannot be read ({type(error).__name__}: {error})."
                f" Run: {command}"
            ) from error
        tables[file.removesuffix(".parquet")] = frame
    return tables, manifest


def fetch(names, inputs, update_manifest=False):
    """Fetch sources at the releases that their committed manifests pin.

    ``manifests.write_checked`` writes each fetch to a staging folder first.
    It keeps the fetch only if the fetch matches the committed manifest or
    ``update_manifest`` accepts it. See ``manifests.decide``.

    If a source cannot be fetched, for example because its download needs
    credentials that are not set, ``fetch`` prints the error. It still fetches
    the other sources.

    Args:
        names (list[str]): Sources to fetch, from FETCHERS.
        inputs (Path): The inputs folder. The files go to its ``sources/``.
        update_manifest (bool): Accept a fetch that differs from its
            committed manifest, under a version not committed in git, and
            record it there.

    Returns:
        int: 0 if every fetch matches or was accepted, 1 otherwise.
    """
    root = Path(inputs) / "sources"
    root.mkdir(parents=True, exist_ok=True)
    status = 0
    for name in names:
        committed_path = MANIFESTS / f"{name}.json"
        print(f"{name}:")
        try:
            status |= fetch_one(name, committed_path, root, update_manifest)
        except Exception as error:
            print(
                f"  {name} not fetched: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            status = 1
    return status


def fetch_one(name, committed_path, root, update_manifest):
    """Fetch one source, and replace its working folder if the fetch may be kept.

    ``manifests.write_checked`` writes the tables, and keeps them only if the
    committed manifest allows.

    Args:
        name (str): The source, from FETCHERS.
        committed_path (Path): The source's committed manifest.
        root (Path): The inputs folder's ``sources/``.
        update_manifest (bool): As in ``fetch``.

    Returns:
        int: 0 if the fetch matches or was accepted, 1 otherwise.
    """
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    tables, sources, counts = FETCHERS[name].fetch_source(
        root / "downloads" / name, committed
    )
    return manifests.write_checked(
        root / name,
        lambda built: write_source(
            built, name, committed["version"], tables, sources, counts
        ),
        committed_path,
        update_manifest,
        f"python -m embedding_model sources fetch --source {name}",
    )
