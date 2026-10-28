"""Corpora: the sources' pairs, combined, checked for leakage, and split.

``corpus build`` reads the pairs that ``pairs prepare`` wrote, for the sources
named, and writes one corpus under ``<inputs>/corpora/<name>/``:

1. The sources are combined in READERS' order. A pair whose two texts
   normalise to the same two texts as an earlier source's pair is kept once,
   in the earlier source, with the studies of every copy. The texts are
   normalised as ``normalization.TOKENIZER_NORMALIZER`` normalises them.
2. Leakage prevention drops the pairs that would leak a benchmark built under
   ``<inputs>/benchmarks/`` (see ``embedding_model.leakage``).
3. The kept pairs are split into training and validation pairs. All pairs of
   one group go to the same side: an item, or an NCIt concept in both NCIt and
   CDISC. A hash of the group decides, so the split is the same on every
   build, and a group keeps its side as the corpus grows.

The corpus holds ``pairs.parquet``, the kept pairs with their ``split``,
``dropped.parquet``, every dropped pair with its findings, and
``manifest.json``. It is written to a staging folder first, and replaces the
last corpus of that name only once it is complete.
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from embedding_model import benchmarks, leakage
from embedding_model.download import file_sha256
from embedding_model.pairs import READERS, pairs_sha256, read_pairs, read_pairs_file
from embedding_model.folders import FOLDER_NAME, FOLDER_NAME_RULE, staged
from embedding_model.pairs.common import COLUMNS, distinct_pairs

# The share of items held out for validation, and the hash buckets it is cut in.
VALIDATION_SHARE = 0.02
BUCKETS = 1000
# Sources whose item ids are NCIt concept codes.
CONCEPT_CODES = {"ncit", "cdisc"}
TRAIN = "train"
VALIDATION = "validation"


def read_corpus(inputs, name):
    """A corpus's kept pairs and its manifest, checked against each other.

    Args:
        inputs (Path): The inputs folder that the corpus was built in.
        name (str): The corpus's name.

    Returns:
        tuple[pd.DataFrame, dict]: The kept pairs, with ``split``, and the
            corpus's manifest.

    Raises:
        FileNotFoundError: If no corpus of that name was built there.
        ValueError: If the name is not a corpus name, the pairs or the manifest
            cannot be read, the manifest is another corpus's, or the pairs
            lack a column or differ from the manifest.
    """
    if not FOLDER_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a corpus name: use {FOLDER_NAME_RULE}")
    folder = Path(inputs) / "corpora" / name
    return read_pairs_file(
        folder / "pairs.parquet",
        folder / "manifest.json",
        "corpus",
        name,
        [*COLUMNS, "split"],
        f"python -m embedding_model corpus build --name {name}",
    )


def built_benchmarks(inputs):
    """Every benchmark built in the inputs folder, read and checked.

    A benchmark counts as built when its manifest is there. A built benchmark
    that cannot be read, such as one with a file missing or changed, raises.

    Args:
        inputs (Path): The inputs folder.

    Returns:
        dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]]: Each benchmark's
            targets, mappings and manifest, in BUILDERS' order.
    """
    folder = Path(inputs) / "benchmarks"
    return {
        name: benchmarks.read_benchmark(inputs, name)
        for name in benchmarks.BUILDERS
        if (folder / f"{name}_manifest.json").exists()
    }


def combine(frames):
    """The sources' pairs, each pair once, with the studies of every copy.

    A pair whose two texts normalise to the same two texts as an earlier
    pair's is dropped, as ``pairs.common.distinct_pairs`` compares them. The
    pair kept takes the studies of all its copies, so the study rule still
    sees a later copy's study.

    Args:
        frames (list[pd.DataFrame]): The sources' pairs, in READERS' order.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The pairs kept, and the source of
            each copy that was dropped.
    """
    combined = pd.concat(frames, ignore_index=True)
    distinct = distinct_pairs(combined)
    return distinct.reset_index(drop=True), combined.source.drop(index=distinct.index)


def group_keys(pairs):
    """The group of each pair: its source and item, or the NCIt concept code."""
    ncit_codes = pairs.source.isin(CONCEPT_CODES)
    return ("ncit:" + pairs["item"]).where(
        ncit_codes, pairs.source + ":" + pairs["item"]
    )


def bucket(key):
    """A stable hash bucket of a text. Python's own hash() differs per process."""
    return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) % BUCKETS


def splits(pairs):
    """Each pair's split, by a hash of its group.

    Args:
        pairs (pd.DataFrame): The pairs, with ``source`` and ``item``.

    Returns:
        pd.Series: TRAIN or VALIDATION, per pair.
    """
    groups = group_keys(pairs)
    cut = VALIDATION_SHARE * BUCKETS
    held_out = {group for group in groups.unique() if bucket(group) < cut}
    return groups.isin(held_out).map({True: VALIDATION, False: TRAIN})


def manifest(
    *,
    name,
    distinct,
    copy_sources,
    dropped,
    kept,
    files_sha256,
    pairs_manifests,
    benchmark_manifests,
    shared_vocabulary,
):
    """The corpus's manifest, for ``manifest.json``.

    It has the fields of every manifest, and ``settings`` and ``counts``. Its
    inputs are each source's pairs and each benchmark.

    Args:
        name (str): The corpus's name.
        distinct (pd.DataFrame): The pairs, each once, from ``combine``.
        copy_sources (pd.Series): The source of each copy dropped, from
            ``combine``.
        dropped (pd.DataFrame): The dropped pairs, one row per pair and
            benchmark that it leaks.
        kept (pd.DataFrame): The kept pairs, with ``split``.
        files_sha256 (dict[str, str]): The SHA-256 of each written file, by
            its name.
        pairs_manifests (list[dict]): The manifest of each source's pairs, as
            ``read_pairs`` returns it.
        benchmark_manifests (dict[str, dict]): Each benchmark's manifest, by
            name, as ``built_benchmarks`` reads it.
        shared_vocabulary (dict): The shared vocabulary, from ``leakage.audit``.

    Returns:
        dict: The manifest.
    """
    return {
        "corpus": name,
        "built": date.today().isoformat(),
        "files": {
            "pairs.parquet": {
                "rows": len(kept),
                "sha256": files_sha256["pairs.parquet"],
                "csv_sha256": pairs_sha256(kept),
            },
            "dropped.parquet": {
                "rows": len(dropped),
                "sha256": files_sha256["dropped.parquet"],
                "csv_sha256": pairs_sha256(dropped),
            },
        },
        "sources": [
            *(
                {
                    "role": "pairs",
                    "name": record["source"],
                    "version": record["sources"][0]["version"],
                    "files": record["files"],
                }
                for record in pairs_manifests
            ),
            *(
                {
                    "role": "benchmark",
                    "name": benchmark,
                    "version": record["version"],
                    "files": record["files"],
                }
                for benchmark, record in benchmark_manifests.items()
            ),
        ],
        "settings": {
            "near_duplicate_words": leakage.NEAR_DUPLICATE_WORDS,
            "validation_share": VALIDATION_SHARE,
            "buckets": BUCKETS,
        },
        "counts": {
            "rows": {
                "combined": len(distinct) + len(copy_sources),
                "duplicates_removed": len(copy_sources),
                "leakage_removed": len(distinct) - len(kept),
                "kept": len(kept),
                **{
                    split: int((kept.split == split).sum())
                    for split in (TRAIN, VALIDATION)
                },
            },
            "duplicates_by_source": copy_sources.value_counts().to_dict(),
            "leakage_findings": {
                "by_finding": dropped.finding.value_counts().to_dict(),
                "by_benchmark": dropped.benchmark.value_counts().to_dict(),
                "by_source": dropped.source.value_counts().to_dict(),
                "by_fields": dropped.fields[dropped.fields != ""]
                .value_counts()
                .to_dict(),
            },
            "shared_vocabulary": shared_vocabulary,
        },
    }


def build(name, sources, inputs):
    """Combine the sources' pairs, drop what leaks a benchmark, and split.

    Args:
        name (str): The corpus's name, used as its folder's name.
        sources (list[str]): The sources to combine, from READERS.
        inputs (Path): The inputs folder. The corpus goes to its
            ``corpora/<name>/``.

    Returns:
        int: 0 if the corpus was written, 1 if a name, a source or a benchmark
            stops the build, or no pair is left to write.

    Raises:
        OSError: If the corpus cannot be written or moved into place. The
            previous corpus of that name is then kept.
    """
    if not FOLDER_NAME.fullmatch(name):
        print(
            f"{name!r} is not a corpus name: use {FOLDER_NAME_RULE}",
            file=sys.stderr,
        )
        return 1
    unknown = sorted(set(sources) - set(READERS))
    if unknown or not sources:
        print(
            f"choose sources among {', '.join(READERS)}; unknown: {unknown}",
            file=sys.stderr,
        )
        return 1
    try:
        benchmark_builds = built_benchmarks(inputs)
    except Exception as error:
        print(f"a benchmark cannot be read: {error}", file=sys.stderr)
        return 1
    if not benchmark_builds:
        print(
            "no benchmark is built, so the corpus cannot be checked for leakage;"
            " run: python -m embedding_model benchmarks build --topmed",
            file=sys.stderr,
        )
        return 1
    print(
        "  checked against: "
        + ", ".join(
            f"{benchmark} v{record['version']}"
            for benchmark, (_, _, record) in benchmark_builds.items()
        )
    )
    frames, pairs_manifests = [], []
    for source in [source for source in READERS if source in sources]:
        try:
            pairs, pairs_manifest = read_pairs(inputs, source)
        except Exception as error:
            print(error, file=sys.stderr)
            return 1
        frames.append(pairs)
        pairs_manifests.append(pairs_manifest)
        print(f"  {source:<7}{len(pairs):>10,} pairs")
    distinct, copy_sources = combine(frames)
    if distinct.empty:
        print(
            f"no pairs in {', '.join(sources)}, so there is no corpus to build",
            file=sys.stderr,
        )
        return 1
    # The sources' own frames, and the last one read, are not needed past this
    # point. Freeing them lowers the peak memory.
    del frames, pairs
    print(f"  {len(copy_sources):,} pairs already in an earlier source, kept once")

    try:
        findings, shared_vocabulary = leakage.audit(
            distinct,
            {benchmark: tables[:2] for benchmark, tables in benchmark_builds.items()},
        )
    except ValueError as error:
        print(f"leakage prevention cannot run: {error}", file=sys.stderr)
        return 1
    detail = findings.drop(columns="pair").rename(
        columns={
            column: f"benchmark_{column}"
            for column in leakage.COLUMNS
            if column not in ("pair", "finding", "benchmark", "fields")
        }
    )
    dropped = pd.concat(
        [distinct.loc[findings.pair].reset_index(drop=True), detail], axis=1
    )
    kept = distinct.drop(index=findings.pair.unique()).reset_index(drop=True)
    if kept.empty:
        print(
            f"leakage prevention dropped all {len(distinct):,} pairs, so there is"
            " no corpus to build",
            file=sys.stderr,
        )
        return 1
    kept["split"] = splits(kept)

    destination = Path(inputs) / "corpora" / name
    with staged(destination) as built:
        kept.to_parquet(built / "pairs.parquet", index=False)
        dropped.to_parquet(built / "dropped.parquet", index=False)
        written = manifest(
            name=name,
            distinct=distinct,
            copy_sources=copy_sources,
            dropped=dropped,
            kept=kept,
            files_sha256={
                file: file_sha256(built / file)
                for file in ("pairs.parquet", "dropped.parquet")
            },
            pairs_manifests=pairs_manifests,
            benchmark_manifests={
                benchmark: record
                for benchmark, (_, _, record) in benchmark_builds.items()
            },
            shared_vocabulary=shared_vocabulary,
        )
        (built / "manifest.json").write_text(
            json.dumps(written, indent=2) + "\n", encoding="utf-8"
        )
    rows = written["counts"]["rows"]
    counted = written["counts"]["leakage_findings"]["by_finding"]
    print(
        f"  {rows['leakage_removed']:,} pairs dropped for leakage"
        + (
            " (findings: "
            + ", ".join(f"{key} {value:,}" for key, value in counted.items())
            + ")"
            if counted
            else ""
        )
    )
    print(
        f"  {rows['kept']:,} pairs kept: {rows[TRAIN]:,} for training, "
        f"{rows[VALIDATION]:,} for validation -> {destination}"
    )
    return 0
