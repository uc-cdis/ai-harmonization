"""Pairs: each source's items, made into pairs of texts that mean the same thing.

``pairs prepare`` reads the tables that ``sources fetch`` wrote. It writes each
source's pairs to ``<inputs>/pairs/<source>/pairs.parquet``, beside their
manifest, ``manifest.json``. The folder is written whole, through
``folders.staged``. Each source has its own module, which reads the
source into items. ``common.star`` then makes the pairs the same way for every
source. ``read_pairs`` reads them back, checked against their manifest. The
rules are in ``embedding_model/docs/pairs.md``.

A pair keeps both rendered texts, ``anchor`` and ``positive``, and the fields
that they were rendered from. So a later step can compare fields rather than
strings.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from embedding_model.pairs import (
    cadsr,
    cde,
    cdisc,
    common,
    loinc,
    ncit,
    nhanes,
    obo,
    phenx,
)
from embedding_model import manifests
from embedding_model.folders import staged
from embedding_model.manifests import file_sha256, table_sha256
from embedding_model.normalization import normalised
from embedding_model.sources import read_source

# The sources, in order. When a corpus combines them, a pair found in two
# sources is kept in the first. CDISC is published as NCIt subsets, so NCIt
# comes before it.
READERS = {
    "ncit": ncit,
    "obo": obo,
    "loinc": loinc,
    "cdisc": cdisc,
    "cde": cde,
    "cadsr": cadsr,
    "nhanes": nhanes,
    "phenx": phenx,
}


def pairs_sha256(pairs):
    """The SHA-256 of a pairs table's content, computed by ``table_sha256``.

    A list read back from parquet is an array, and CSV writes an array
    differently from a list. So each value list is written as JSON first.

    Args:
        pairs (pd.DataFrame): A source's pairs.

    Returns:
        str: The SHA-256, as hex.
    """
    lists = {
        column: pairs[column].map(lambda values: json.dumps(list(values)))
        for column in ("anchor_values", "positive_values")
    }
    return table_sha256(pairs.assign(**lists))


def new_pairs(pairs, seen):
    """Which of a source's pairs no earlier source has, by the hash of their texts.

    The texts are hashed after the tokenizer's normalisation.
    ``common.distinct_pairs`` compares them the same way.

    Args:
        pairs (pd.DataFrame): A source's pairs.
        seen (set[int]): The hashes of the earlier sources' pairs. This
            source's hashes are added to it.

    Returns:
        pd.Series: True where a pair is the first copy.
    """
    texts = zip(normalised(pairs.anchor), normalised(pairs.positive))
    hashes = [hash(pair_texts) for pair_texts in texts]
    new = pd.Series([pair_hash not in seen for pair_hash in hashes], index=pairs.index)
    seen.update(hashes)
    return new.astype(bool)


def ambiguous_texts(pairs):
    """How many source-side texts point to two or more canonicals.

    Args:
        pairs (pd.DataFrame): A source's pairs.

    Returns:
        int: The number of such texts, compared by ``common.key``.
    """
    names = pairs[pairs.kind == common.NAME]
    targets = names.positive_name.map(common.key).groupby(
        names.anchor_name.map(common.key)
    )
    return int((targets.nunique() > 1).sum())


def manifest(name, pairs, pairs_file_sha256, counts, fetch):
    """The manifest of a source's pairs: what they were made from, and what they hold.

    It has the fields of every manifest, and ``counts``. Its one input is the
    source's fetch, with the fetch's own record of its tables. So pairs made
    from another fetch can be told apart.

    Args:
        name (str): The source, as in READERS.
        pairs (pd.DataFrame): Its pairs.
        pairs_file_sha256 (str): The SHA-256 of the pairs' written file.
        counts (dict[str, int]): The counts ``common.star`` returns.
        fetch (dict): The manifest of the source's fetch.

    Returns:
        dict: The manifest.
    """
    return {
        "source": name,
        "built": date.today().isoformat(),
        "files": {
            "pairs.parquet": {
                "rows": len(pairs),
                "sha256": pairs_file_sha256,
                "csv_sha256": pairs_sha256(pairs),
            }
        },
        "sources": [
            {
                "role": "fetch",
                "name": name,
                "version": fetch["version"],
                "fetched": fetch["fetched"],
                "files": fetch["files"],
            }
        ],
        "counts": {
            **counts,
            "ambiguous_texts": ambiguous_texts(pairs),
            "kind": pairs.kind.value_counts().to_dict(),
            "term_group": pairs.term_group.value_counts().to_dict(),
            "status": pairs.status.value_counts().to_dict(),
        },
    }


def read_pairs_file(table, manifest, field, name, columns, command):
    """A table of pairs that a step wrote, and its manifest, checked against each other.

    ``pairs prepare`` and ``corpus build`` each write a table of pairs, and a
    manifest that names it and records its checksums. Five things are checked,
    in this order:

    1. the pairs and the manifest exist;
    2. the manifest can be read;
    3. the manifest names the build asked for;
    4. the file's ``sha256`` matches the manifest, so the file is unchanged
       since its step wrote it;
    5. the pairs can be read, and have the columns given.

    ``csv_sha256`` is not checked here: it compares the tables of two builds.

    Args:
        table (Path): The parquet file of the pairs.
        manifest (Path): The manifest written with it.
        field (str): The manifest's field that names what was built, such as
            ``source`` or ``corpus``.
        name (str): The name that the field must hold.
        columns (list[str]): The columns that the pairs must have, in order.
        command (str): The command that writes the pairs again, for the
            messages.

    Returns:
        tuple[pd.DataFrame, dict]: The pairs and their manifest.

    Raises:
        FileNotFoundError: If the pairs or their manifest were not written.
        ValueError: If the pairs or the manifest cannot be read, the manifest
            names another build, or the pairs lack a column or differ from the
            manifest.
    """
    if not table.exists():
        raise FileNotFoundError(f"{table} was not written; run: {command}")
    record = manifests.read(manifest, command, field, name)
    try:
        checksum = record["files"][table.name]["sha256"]
    except Exception as error:
        raise ValueError(
            f"{manifest} records no SHA-256 for {table.name}"
            f" ({type(error).__name__}: {error}); run: {command}"
        ) from error
    if file_sha256(table) != checksum:
        raise ValueError(f"{table} differs from its manifest; run: {command}")
    try:
        pairs = pd.read_parquet(table)
    except Exception as error:
        raise ValueError(
            f"{table} cannot be read ({type(error).__name__}: {error});"
            f" run: {command}"
        ) from error
    if list(pairs.columns) != columns:
        raise ValueError(
            f"{table} does not have the columns that its step writes; run: {command}"
        )
    return pairs, record


def read_pairs(inputs, name):
    """A source's pairs and their manifest, checked against each other.

    The manifest records the pairs' file and the fetch that they come from.
    When that fetch is in the inputs folder, the pairs must come from it.

    Args:
        inputs (Path): The inputs folder that the pairs were prepared in.
        name (str): The source, as in READERS.

    Returns:
        tuple[pd.DataFrame, dict]: The pairs and their manifest.

    Raises:
        FileNotFoundError: If the source's pairs were not prepared there.
        ValueError: If the pairs, their manifest or the fetch's manifest cannot
            be read, the manifest is another source's, the pairs lack a column
            or differ from their manifest, or they come from another fetch.
    """
    folder = Path(inputs) / "pairs" / name
    manifest_path = folder / "manifest.json"
    fetch_path = Path(inputs) / "sources" / name / "manifest.json"
    command = f"python -m embedding_model pairs prepare --source {name}"
    pairs, record = read_pairs_file(
        folder / "pairs.parquet", manifest_path, "source", name, common.COLUMNS, command
    )
    try:
        fetch_entry = record["sources"][0]
        fetched = fetch_entry["files"]
        if "version" not in fetch_entry:
            raise KeyError("version")
    except Exception as error:
        raise ValueError(
            f"{manifest_path} does not record the fetch of {name}"
            f" ({type(error).__name__}: {error}); run: {command}"
        ) from error
    if not fetch_path.exists():
        return pairs, record
    fetch_command = f"python -m embedding_model sources fetch --source {name}"
    fetch_manifest = manifests.read(fetch_path, fetch_command, "source", name)
    if fetch_manifest.get("files") != fetched:
        raise ValueError(
            f"{name}'s pairs come from another fetch than {fetch_path.parent};"
            f" run: {command}"
        )
    return pairs, record


def prepare_one(name, inputs):
    """Make one source's pairs, and write them whole with their manifest.

    Args:
        name (str): The source, as in READERS.
        inputs (Path): The inputs folder that the source was fetched into.

    Returns:
        pd.DataFrame: The pairs, also written as
            ``pairs/<source>/pairs.parquet``, beside their manifest,
            ``manifest.json``.
    """
    tables, fetch_manifest = read_source(inputs, name)
    pairs, counts = common.star(READERS[name].items(tables), name)
    destination = Path(inputs) / "pairs" / name
    with staged(destination) as built:
        pairs.to_parquet(built / "pairs.parquet", index=False)
        pairs_manifest = manifest(
            name, pairs, file_sha256(built / "pairs.parquet"), counts, fetch_manifest
        )
        manifests.write(built / "manifest.json", pairs_manifest)
    print(
        f"  {len(pairs):,} pairs from {counts['items']:,} items; "
        f"{counts['items_not_open']:,} items not open, "
        f"{counts['duplicates_removed']:,} duplicate pairs removed -> {destination}"
    )
    return pairs


def prepare(names, inputs):
    """Make the pairs of fetched sources, in READERS' order.

    A source that cannot be read is reported, and the others are still
    prepared. Then, for each source, ``new_pairs`` counts the pairs that an
    earlier source in the same run also has.

    Args:
        names (list[str]): Sources to prepare, from READERS.
        inputs (Path): The inputs folder that the sources were fetched into.
            Each source's pairs go to its ``pairs/<source>/`` folder.

    Returns:
        int: 0 if every source was prepared, 1 otherwise.
    """
    status = 0
    seen, shared = set(), {}
    for name in [name for name in READERS if name in names]:
        print(f"{name}:")
        try:
            pairs = prepare_one(name, inputs)
        except Exception as error:
            print(
                f"  {name} not prepared: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            status = 1
            continue
        shared[name] = int((~new_pairs(pairs, seen)).sum())
    if len(shared) > 1:
        copy_counts = ", ".join(
            f"{name} {count:,}" for name, count in shared.items() if count
        )
        print(
            f"pairs also in an earlier source, kept there once: {copy_counts or 'none'}"
        )
    return status
