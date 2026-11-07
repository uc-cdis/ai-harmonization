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
   one group go to the same side: items that share an NCIt concept code or a
   canonical text, as ``group_ids`` links them. A hash of the group decides,
   so the same sources and benchmarks always give the same split.

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

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from embedding_model import benchmarks, leakage, manifests
from embedding_model.pairs import READERS, pairs_sha256, read_pairs, read_pairs_file
from embedding_model.folders import FOLDER_NAME, FOLDER_NAME_RULE, staged
from embedding_model.manifests import file_sha256
from embedding_model.pairs.common import COLUMNS, NAME_TO_DESCRIPTION, distinct_pairs

# The share of groups held out for validation by default, and the hash buckets
# that the share is cut in. So a share is a whole number of buckets: a multiple
# of 0.001.
VALIDATION_SHARE = 0.02
BUCKETS = 1000
# Sources whose item ids are NCIt concept codes.
CONCEPT_CODES = {"ncit", "cdisc"}
TRAIN = "train"
VALIDATION = "validation"


def build_command(folder):
    """The command that builds a corpus again, with the sources and share it had.

    The corpus's manifest gives its sources and validation share, so the
    command builds the same corpus. When the manifest cannot be read, the
    command names the corpus alone, with the defaults.

    Args:
        folder (Path): The corpus's folder, named after the corpus.

    Returns:
        str: The command.
    """
    command = f"python -m embedding_model corpus build --name {folder.name}"
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        sources = [
            source["name"]
            for source in manifest["sources"]
            if source["role"] == "pairs"
        ]
        share = manifest["settings"]["validation_share"]
    except Exception:
        return command
    return f"{command} --source {' '.join(sources)} --validation-share {share}"


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
        build_command(folder),
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
        if (folder / name / "manifest.json").exists()
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


def item_ids(pairs):
    """The ``item_id`` of each pair: its source and item, or the NCIt concept code."""
    ncit_codes = pairs.source.isin(CONCEPT_CODES)
    return ("ncit:" + pairs["item"]).where(
        ncit_codes, pairs.source + ":" + pairs["item"]
    )


def canonical_texts(pairs):
    """Each pair's canonical: the positive's name, or the anchor's name for a
    ``name_to_description`` pair, whose positive is a description alone."""
    return pairs.positive_name.where(
        pairs.kind != NAME_TO_DESCRIPTION, pairs.anchor_name
    )


def group_ids(pairs):
    """The ``group_id`` of each pair: the ``item_id`` of its group's first item.

    Two items are in one group when they share an ``item_id``, as an NCIt
    concept and its CDISC terms do, or a canonical text's ``text_key``, as
    ``pairs.common.text_key`` writes it. So two canonicals that differ only in
    case or punctuation link their items. An empty ``text_key`` links nothing.
    The links carry over, so the groups are the connected components of a
    graph: each pair links its item to its canonical text. A group's
    ``group_id`` is the first ``item_id`` of its items in sorted order, so the
    same pairs always give the same ``group_id``.

    Args:
        pairs (pd.DataFrame): The pairs, with ``source``, ``item``, ``kind``,
            ``anchor_name`` and ``positive_name``.

    Returns:
        pd.Series: The ``group_id`` of each pair.
    """
    items = item_ids(pairs)
    canonical_keys = leakage.text_keys(canonical_texts(pairs))
    texts = ("text:" + canonical_keys).where(canonical_keys != "", items)
    nodes = pd.Index(np.concatenate([items, texts])).unique()
    item_nodes, text_nodes = nodes.get_indexer(items), nodes.get_indexer(texts)
    links = coo_matrix(
        (np.ones(len(pairs)), (item_nodes, text_nodes)), shape=(len(nodes),) * 2
    )
    _, component = connected_components(links, directed=False)
    first_item = items.groupby(component[item_nodes]).min()
    return pd.Series(
        first_item.loc[component[item_nodes]].to_numpy(), index=pairs.index
    )


def bucket(group_id):
    """A stable hash bucket of a ``group_id``.

    Python's own hash() differs per process.
    """
    return (
        int(hashlib.blake2b(group_id.encode(), digest_size=8).hexdigest(), 16) % BUCKETS
    )


def validation_buckets(validation_share):
    """The number of buckets that a validation share holds.

    Args:
        validation_share (float): The share of groups held out for
            validation, such as 0.02.

    Returns:
        int: The buckets, from 1 to BUCKETS - 1.

    Raises:
        ValueError: If the share is not a multiple of 1 / BUCKETS between 0
            and 1, both left out.
    """
    held = round(validation_share * BUCKETS)
    if not 0 < held < BUCKETS or abs(validation_share * BUCKETS - held) > 1e-9:
        raise ValueError(
            f"a validation share must be between {1 / BUCKETS} and"
            f" {1 - 1 / BUCKETS}, in steps of {1 / BUCKETS}: {validation_share}"
        )
    return held


def splits(pairs, validation_share=VALIDATION_SHARE):
    """Each pair's split, by a hash of its group.

    A group goes to validation when its bucket is below the share's buckets.
    So a larger share keeps the groups that a smaller share holds out, and
    adds others.

    Args:
        pairs (pd.DataFrame): The pairs, with the columns that ``group_ids``
            reads.
        validation_share (float): The share of groups held out for
            validation, a multiple of 1 / BUCKETS.

    Returns:
        pd.Series: TRAIN or VALIDATION, per pair.
    """
    groups = group_ids(pairs)
    cut = validation_buckets(validation_share)
    held_out = {group for group in groups.unique() if bucket(group) < cut}
    return groups.isin(held_out).map({True: VALIDATION, False: TRAIN})


def manifest(
    *,
    name,
    distinct,
    copy_sources,
    dropped,
    kept_pairs,
    files_sha256,
    pairs_manifests,
    benchmark_manifests,
    shared_vocabulary,
    validation_share,
):
    """The corpus's manifest, for ``manifest.json``.

    It has the fields of every manifest, and ``settings`` and ``counts``. Its
    inputs are each source's pairs and each benchmark. ``settings`` holds the
    leakage and split settings, and ``format``, the text format of the pairs,
    which every source's pairs share.

    Args:
        name (str): The corpus's name.
        distinct (pd.DataFrame): The pairs, each once, from ``combine``.
        copy_sources (pd.Series): The source of each copy dropped, from
            ``combine``.
        dropped (pd.DataFrame): The dropped pairs, one row per pair and
            benchmark that it leaks.
        kept_pairs (pd.DataFrame): The kept pairs, with ``split``.
        files_sha256 (dict[str, str]): The SHA-256 of each written file, by
            its name.
        pairs_manifests (list[dict]): The manifest of each source's pairs, as
            ``read_pairs`` returns it.
        benchmark_manifests (dict[str, dict]): Each benchmark's manifest, by
            name, as ``built_benchmarks`` reads it.
        shared_vocabulary (dict): The shared vocabulary, from ``leakage.audit``.
        validation_share (float): The share of groups held out for validation.

    Returns:
        dict: The manifest.
    """
    return {
        "corpus": name,
        "built": date.today().isoformat(),
        "files": {
            "pairs.parquet": {
                "rows": len(kept_pairs),
                "sha256": files_sha256["pairs.parquet"],
                "content_sha256": pairs_sha256(kept_pairs),
            },
            "dropped.parquet": {
                "rows": len(dropped),
                "sha256": files_sha256["dropped.parquet"],
                "content_sha256": pairs_sha256(dropped),
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
            "format": pairs_manifests[0]["settings"]["format"],
            "min_overlap_words": leakage.MIN_OVERLAP_WORDS,
            "min_name_words": leakage.MIN_NAME_WORDS,
            "validation_share": validation_share,
            "buckets": BUCKETS,
        },
        "counts": {
            "rows": {
                "combined": len(distinct) + len(copy_sources),
                "duplicates_removed": len(copy_sources),
                "leakage_removed": len(distinct) - len(kept_pairs),
                "kept": len(kept_pairs),
                **{
                    split: int((kept_pairs.split == split).sum())
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


def checked_sources(name, sources, validation_share):
    """Check the corpus's name, sources and validation share, and order the sources.

    Args:
        name (str): The corpus's name, used as its folder's name.
        sources (list[str]): The sources to combine, from READERS.
        validation_share (float): The share of groups held out for
            validation, a multiple of 1 / BUCKETS.

    Returns:
        list[str]: The sources, each once, in READERS' order: the order in
            which they are read and combined.

    Raises:
        ValueError: If the name is not a FOLDER_NAME, no source is given or
            one is not in READERS, or the validation share is not a multiple
            of 1 / BUCKETS between 0 and 1.
    """
    if not FOLDER_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a corpus name: use {FOLDER_NAME_RULE}")
    unknown = sorted(set(sources) - set(READERS))
    if unknown or not sources:
        raise ValueError(
            f"choose sources among {', '.join(READERS)}; unknown: {unknown}"
        )
    try:
        validation_buckets(validation_share)
    except ValueError as error:
        raise ValueError(f"--validation-share: {error}") from error
    return [source for source in READERS if source in sources]


def read_benchmarks(inputs):
    """The built benchmarks, which the corpus is checked against for leakage.

    The benchmarks are printed with their versions. A benchmark that cannot be
    read, or no benchmark built, is reported on stderr.

    Args:
        inputs (Path): The inputs folder that the benchmarks were built into.

    Returns:
        dict | None: Each benchmark's targets, mappings and record, from
            built_benchmarks, or None when the corpus cannot be checked.
    """
    try:
        benchmark_builds = built_benchmarks(inputs)
    except Exception as error:
        print(f"a benchmark cannot be read: {error}", file=sys.stderr)
        return None
    if not benchmark_builds:
        print(
            "no benchmark is built, so the corpus cannot be checked for leakage;"
            " run: python -m embedding_model benchmarks build --topmed",
            file=sys.stderr,
        )
        return None
    print(
        "  checked against: "
        + ", ".join(
            f"{benchmark} v{record['version']}"
            for benchmark, (_, _, record) in benchmark_builds.items()
        )
    )
    return benchmark_builds


def read_sources(inputs, sources):
    """Each source's pairs and their manifest, in the order given.

    Each source read prints its number of pairs. A source that cannot be read
    is reported on stderr, and so are sources whose pairs are in different
    text formats.

    Args:
        inputs (Path): The inputs folder that the sources' pairs are in.
        sources (list[str]): The sources to combine, from checked_sources.

    Returns:
        tuple[list[pd.DataFrame] | None, list[dict] | None]: The sources'
            pairs and their manifests, or None for both when a source cannot
            be used.
    """
    frames, pairs_manifests, failed = [], [], []
    for source in sources:
        try:
            pairs, pairs_manifest = read_pairs(inputs, source)
        except Exception as error:
            print(f"  {source}: {type(error).__name__}: {error}", file=sys.stderr)
            failed.append(source)
            continue
        frames.append(pairs)
        pairs_manifests.append(pairs_manifest)
        print(f"  {source:<7}{len(pairs):>10,} pairs")
    if failed:
        print(
            f"the pairs of {', '.join(failed)} cannot be used, as said above",
            file=sys.stderr,
        )
        return None, None
    formats = {
        record["source"]: record["settings"]["format"] for record in pairs_manifests
    }
    if len(set(formats.values())) > 1:
        print(
            "the sources' pairs are in different text formats: "
            + ", ".join(f"{source} {formats[source]}" for source in formats)
            + "; prepare them again: python -m embedding_model pairs prepare"
            f" --source {' '.join(formats)}",
            file=sys.stderr,
        )
        return None, None
    return frames, pairs_manifests


def dropped_and_kept(distinct, benchmark_builds):
    """The pairs that leak a benchmark, with their findings, and the pairs kept.

    Args:
        distinct (pd.DataFrame): The sources' pairs, combined, each once.
        benchmark_builds (dict): Each benchmark's targets, mappings and
            record, from built_benchmarks.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: The dropped pairs, each with
            its finding's columns; the pairs kept; and each benchmark's
            vocabulary shared with the pairs, from leakage.audit.
    """
    findings, shared_vocabulary = leakage.audit(
        distinct,
        {benchmark: tables[:2] for benchmark, tables in benchmark_builds.items()},
    )
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
    kept_pairs = distinct.drop(index=findings.pair.unique()).reset_index(drop=True)
    return dropped, kept_pairs, shared_vocabulary


def print_summary(corpus_manifest, destination):
    """Print the pairs dropped for leakage, the pairs kept, and where they went.

    Args:
        corpus_manifest (dict): The corpus's manifest, with its counts.
        destination (Path): The corpus's folder.
    """
    rows = corpus_manifest["counts"]["rows"]
    counted = corpus_manifest["counts"]["leakage_findings"]["by_finding"]
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


def build(name, sources, inputs, validation_share=VALIDATION_SHARE):
    """Combine the sources' pairs, drop what leaks a benchmark, and split.

    Args:
        name (str): The corpus's name, used as its folder's name.
        sources (list[str]): The sources to combine, from READERS.
        inputs (Path): The inputs folder. The corpus goes to its
            ``corpora/<name>/``.
        validation_share (float): The share of groups held out for
            validation, a multiple of 1 / BUCKETS.

    Returns:
        int: 0 if the corpus was written, 1 if a name, a source, the
            validation share or a benchmark stops the build, or no pair is
            left to write.

    Raises:
        OSError: If the corpus cannot be written or moved into place. The
            previous corpus of that name is then kept.
    """
    try:
        sources = checked_sources(name, sources, validation_share)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    benchmark_builds = read_benchmarks(inputs)
    if benchmark_builds is None:
        return 1
    frames, pairs_manifests = read_sources(inputs, sources)
    if frames is None:
        return 1
    distinct, copy_sources = combine(frames)
    if distinct.empty:
        print(
            f"no pairs in {', '.join(sources)}, so there is no corpus to build",
            file=sys.stderr,
        )
        return 1
    # The sources' own frames are not needed past this point. Freeing them
    # lowers the peak memory.
    del frames
    print(f"  {len(copy_sources):,} pairs already in an earlier source, kept once")
    dropped, kept_pairs, shared_vocabulary = dropped_and_kept(
        distinct, benchmark_builds
    )
    if kept_pairs.empty:
        print(
            f"leakage prevention dropped all {len(distinct):,} pairs, so there is"
            " no corpus to build",
            file=sys.stderr,
        )
        return 1
    kept_pairs["split"] = splits(kept_pairs, validation_share)
    destination = Path(inputs) / "corpora" / name
    with staged(destination) as built:
        kept_pairs.to_parquet(built / "pairs.parquet", index=False)
        dropped.to_parquet(built / "dropped.parquet", index=False)
        corpus_manifest = manifest(
            name=name,
            distinct=distinct,
            copy_sources=copy_sources,
            dropped=dropped,
            kept_pairs=kept_pairs,
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
            validation_share=validation_share,
        )
        manifests.write(built / "manifest.json", corpus_manifest)
    print_summary(corpus_manifest, destination)
    return 0
