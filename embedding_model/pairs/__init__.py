"""Pairs: each source's items, made into pairs of texts that mean the same thing.

``pairs prepare`` reads the tables that ``sources fetch`` wrote. It writes each
source's pairs to ``<inputs>/pairs/<source>.parquet``, beside their manifest,
``<source>_manifest.json``. Each source has its own module, which reads the
source into items. ``common.star`` then makes the pairs the same way for every
source. The rules are in ``embedding_model/docs/pairs.md``.

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
from embedding_model.sources import read_source, table_sha256

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

    Args:
        pairs (pd.DataFrame): A source's pairs.
        seen (set[int]): The hashes of the earlier sources' pairs. This
            source's hashes are added to it.

    Returns:
        pd.Series: True where a pair is the first copy.
    """
    hashes = [hash(texts) for texts in zip(pairs.anchor, pairs.positive)]
    new = pd.Series([found not in seen for found in hashes], index=pairs.index)
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


def manifest(name, pairs, counts, fetch):
    """The manifest of a source's pairs: what they were made from, and what they hold.

    It has the fields of every manifest, and ``counts``. Its one input is the
    source's fetch, with the fetch's own record of its tables. So pairs made
    from another fetch can be told apart.

    Args:
        name (str): The source, as in READERS.
        pairs (pd.DataFrame): Its pairs.
        counts (dict[str, int]): The counts ``common.star`` returns.
        fetch (dict): The manifest of the source's fetch.

    Returns:
        dict: The manifest.
    """
    return {
        "source": name,
        "built": date.today().isoformat(),
        "files": {
            f"{name}.parquet": {"rows": len(pairs), "csv_sha256": pairs_sha256(pairs)}
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


def prepare_one(name, inputs, root):
    """Make one source's pairs, and write them with their manifest.

    Args:
        name (str): The source, as in READERS.
        inputs (Path): The inputs folder that the source was fetched into.
        root (Path): The folder for the pairs.

    Returns:
        pd.DataFrame: The pairs, also written as ``<source>.parquet``, beside
            their manifest, ``<source>_manifest.json``.
    """
    tables, fetch = read_source(inputs, name)
    pairs, counts = common.star(READERS[name].items(tables), name)
    found = manifest(name, pairs, counts, fetch)
    pairs.to_parquet(root / f"{name}.parquet", index=False)
    (root / f"{name}_manifest.json").write_text(
        json.dumps(found, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"  {len(pairs):,} pairs from {counts['items']:,} items; "
        f"{counts['items_not_open']:,} items not open, "
        f"{counts['duplicates_removed']:,} duplicate pairs removed"
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
            The pairs go to its ``pairs/`` folder.

    Returns:
        int: 0 if every source was prepared, 1 otherwise.
    """
    root = Path(inputs) / "pairs"
    root.mkdir(parents=True, exist_ok=True)
    status = 0
    seen, shared = set(), {}
    for name in [name for name in READERS if name in names]:
        print(f"{name}:")
        try:
            pairs = prepare_one(name, inputs, root)
        except Exception as error:
            print(
                f"  {name} not prepared: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            status = 1
            continue
        shared[name] = int((~new_pairs(pairs, seen)).sum())
    if len(shared) > 1:
        found = ", ".join(
            f"{name} {count:,}" for name, count in shared.items() if count
        )
        print(f"pairs also in an earlier source, kept there once: {found or 'none'}")
    return status
