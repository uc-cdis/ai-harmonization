"""Leakage prevention: find the training pairs that would leak a benchmark.

Each side of a pair is a text formatted from its fields: a name, a type, a
description and values. For example, the anchor ``ABCD_SC (integer)`` is
formatted from the name ``ABCD_SC`` and the type ``integer``. The step compares
the name and description fields rather than the formatted texts. The same name
can come with another type or other values, and its formatted text then
differs. The fields compared are:

- a pair's side texts: the name and the description of its anchor and of its
  positive;
- a benchmark row's texts: the name and the description of its source
  variable and of each expected target. A target's name is its property name,
  without its node.

The benchmark texts are cleaned first with ``pairs.common.clean``, the
cleaning that ``pairs prepare`` gives every text. Two texts match when their
keys are equal, as ``pairs.common.key`` writes them: normalised as the
tokenizer normalises them, case-folded, with punctuation as spaces. A side text also matches a benchmark description when
one contains the other, and the shorter has at least NEAR_DUPLICATE_WORDS
words. A pair can hold a description in a name field, such as a
``name_to_description`` pair's positive, so every side text is compared,
whatever its field.

A finding is a reason to drop a pair, found against one benchmark. There are
two kinds:

- ``answer_key``: one side's name or description matches the source
  variable's name or description, and the other side's name or description
  matches one of that variable's expected targets, by its property name or
  description. Either side can match the source.
  For example, a benchmark maps the variable ``body_weight`` to
  ``vitals.weight``. Then the pair ``Weight`` -> ``Body Weight`` is an answer
  key: its positive matches the source variable, and its anchor matches the
  expected target.
- ``benchmark_study``, the study rule: the pair's ``study`` holds the source
  study of a benchmark row. For example, a benchmark row comes from the study
  ``phs999901``. Then every pair made from a dbGaP variable of ``phs999901``
  is dropped, whatever its texts.

Each benchmark is checked on its own, and a pair is dropped if any benchmark
finds it. Against one benchmark, a pair has at most one finding: an answer key
comes before a benchmark study. A pair that leaks two benchmarks has a finding
for each. Values are never compared.

A benchmark text can also occur in a pair without a finding. For example,
``age`` is one side of a pair whose other side matches no expected target of
that benchmark row. That is not leakage: the pair is kept, and the text is
reported as shared vocabulary.
"""

from collections import defaultdict

import pandas as pd

from embedding_model.benchmarks import TARGET_SEPARATOR
from embedding_model.pairs.common import clean, key

# The words that the shorter of two texts needs, for one containing the other
# to count as a match.
NEAR_DUPLICATE_WORDS = 10
# The near-duplicate search indexes texts by their sequences of SHINGLE
# words, called shingles. It changes the search's speed, not what it finds.
SHINGLE = 3
# The two findings, as the findings table and the corpus's manifest name them.
ANSWER_KEY = "answer_key"
BENCHMARK_STUDY = "benchmark_study"
SIDES = ("anchor", "positive")
FIELDS = ("name", "description")
# How many shared texts the corpus's manifest lists per benchmark.
SHARED_SHOWN = 20
COLUMNS = [
    "pair",
    "finding",
    "benchmark",
    "row",
    "source_variable_name",
    "source_variable_description",
    "target_property",
    "target_description",
    "study_id",
    "fields",
]


def keys(texts, cleaned=True):
    """The key of each text, as ``pairs.common.key`` writes it.

    Each distinct text is keyed once.

    Args:
        texts (pd.Series): Texts.
        cleaned (bool): Whether the texts are clean already. A benchmark's
            texts are not, so they go through ``pairs.common.clean`` first.

    Returns:
        pd.Series: The keys.
    """
    texts = texts.fillna("")
    unique = texts.unique()
    prepared = unique if cleaned else [clean(text) for text in unique]
    return texts.map(dict(zip(unique, map(key, prepared))))


def benchmark_texts(name, targets, mappings):
    """A benchmark's source and expected-target texts, as keys.

    Args:
        name (str): The benchmark, such as ``topmed``.
        targets (pd.DataFrame): Its targets, from ``read_benchmark``.
        mappings (pd.DataFrame): Its mappings, from ``read_benchmark``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The source texts, one row per text:
            ``benchmark``, ``row``, ``variable``, ``field`` and ``key``. The
            expected targets' texts, one row per text: the same, and
            ``target``.
    """
    rows = mappings.reset_index(drop=True)
    source_texts = pd.concat(
        [
            pd.DataFrame(
                {
                    "row": rows.index,
                    "variable": rows.source_variable_name,
                    "field": f"source {field}",
                    "key": keys(rows[column], cleaned=False),
                }
            )
            for field, column in (
                ("name", "source_variable_name"),
                ("description", "source_description"),
            )
        ]
    )
    every_target = pd.concat(
        [
            pd.DataFrame(
                {
                    "target": targets.target_property,
                    "field": "target name",
                    "key": keys(
                        targets.target_property.str.rpartition(".")[2], cleaned=False
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "target": targets.target_property,
                    "field": "target description",
                    "key": keys(targets.target_description, cleaned=False),
                }
            ),
        ]
    )
    expected = (
        rows.assign(target=rows.expected_target.str.split(TARGET_SEPARATOR))
        .explode("target")
        .reset_index(names="row")[["row", "source_variable_name", "target"]]
        .rename(columns={"source_variable_name": "variable"})
    )
    missing = sorted(set(expected.target) - set(targets.target_property))
    if missing:
        raise ValueError(
            f"{name}'s expected targets are not among its targets: {missing}"
        )
    repeated = sorted(
        targets.target_property[targets.target_property.duplicated()].unique()
    )
    if repeated:
        raise ValueError(f"{name}'s targets list a property twice: {repeated}")
    target_texts = expected.merge(every_target, on="target")
    return (
        source_texts[source_texts.key != ""].assign(benchmark=name),
        target_texts[target_texts.key != ""].assign(benchmark=name),
    )


def side_keys(pairs):
    """Each pair's side names and descriptions, as keys, one row per text.

    Args:
        pairs (pd.DataFrame): The pairs, with ``anchor_name`` and the other
            field columns.

    Returns:
        pd.DataFrame: ``pair`` (the pair's index), ``side``, ``side_field``
            and ``key``, without empty keys.
    """
    side_texts = pd.concat(
        [
            pd.DataFrame(
                {
                    "pair": pairs.index,
                    "side": side,
                    "side_field": field,
                    "key": keys(pairs[f"{side}_{field}"]),
                }
            )
            for side in SIDES
            for field in FIELDS
        ]
    )
    return side_texts[side_texts.key != ""]


def shingles(words):
    """The sequences of SHINGLE words in a list of words, in order."""
    return [" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)]


def near_duplicates(texts, references, min_words):
    """The texts and references where one contains the other.

    Only texts and references with at least ``min_words`` words can match, since
    the shorter of the two needs them. A text contains another when the other
    appears in it as whole words. Each reference is indexed by its shingles,
    its sequences of SHINGLE words. A text is then compared only with the
    references that could contain it, or that it could contain.

    Args:
        texts (Iterable[str]): Keys, as ``keys`` writes them.
        references (Iterable[str]): Keys to compare them with.
        min_words (int): The fewest words of the shorter key, such as
            NEAR_DUPLICATE_WORDS.

    Returns:
        set[tuple[str, str]]: (text, reference), for each match that is not
            already an equal key.

    Raises:
        ValueError: If ``min_words`` is fewer than SHINGLE.
    """
    if min_words < SHINGLE:
        raise ValueError(f"a near-duplicate needs at least {SHINGLE} words")
    long_enough = {
        reference for reference in references if len(reference.split()) >= min_words
    }
    # Each reference, by its first shingle and by every shingle it has.
    first, anywhere = defaultdict(set), defaultdict(set)
    for reference in long_enough:
        parts = shingles(reference.split())
        first[parts[0]].add(reference)
        for part in parts:
            anywhere[part].add(reference)
    matched = set()
    for text in set(texts):
        split = text.split()
        if len(split) < min_words:
            continue
        parts = shingles(split)
        # Spaces around both texts make the check below match whole words
        # only, so "abcd" is not found inside "xabcd".
        padded = f" {text} "
        # A reference that contains the text holds the text's first shingle.
        candidates = set(anywhere.get(parts[0], ()))
        # A reference inside the text has its first shingle among the text's.
        for part in parts:
            candidates |= first.get(part, set())
        # The index only narrows the candidates. The containment check decides.
        for reference in candidates:
            if reference == text:
                continue
            if f" {reference} " in padded or padded in f" {reference} ":
                matched.add((text, reference))
    return matched


def matches(sides, texts, near_matches):
    """The side texts that match benchmark texts, exactly or as near-duplicates.

    Args:
        sides (pd.DataFrame): From ``side_keys``.
        texts (pd.DataFrame): Benchmark texts, from ``benchmark_texts``.
        near_matches (pd.DataFrame): The side texts that contain, or are
            contained in, a benchmark description: ``sides``' columns, and
            ``reference``, the description's key.

    Returns:
        pd.DataFrame: One row per match: the side's columns, the benchmark
            text's columns, and ``how``, ``=`` for an equal key or ``~`` for a
            near-duplicate.
    """
    exact = sides.merge(texts, on="key").assign(how="=")
    descriptions = texts[texts.field.str.endswith("description")]
    linked = (
        near_matches.drop(columns="key")
        .merge(descriptions.rename(columns={"key": "reference"}), on="reference")
        .drop(columns="reference")
        .assign(how="~")
    )
    return pd.concat([exact, linked], ignore_index=True)


def answer_keys(sides, source_texts, target_texts):
    """The pairs with a benchmark source on one side and its target on the other.

    Args:
        sides (pd.DataFrame): The pairs' side texts, from ``side_keys``.
        source_texts (pd.DataFrame): Every benchmark's source texts.
        target_texts (pd.DataFrame): Every benchmark's expected-target texts.

    Returns:
        pd.DataFrame: One row per pair and benchmark that it leaks: ``pair``,
            ``benchmark``, ``row``, ``source_variable_name``,
            ``target_property`` and ``fields``, which names the two matches,
            such as ``anchor name = source name, positive description ~ target
            description``.
    """
    references = pd.concat(
        [
            frame.key[frame.field.str.endswith("description")]
            for frame in (source_texts, target_texts)
        ]
    )
    near_matches = sides.merge(
        pd.DataFrame(
            sorted(near_duplicates(sides.key, references, NEAR_DUPLICATE_WORDS)),
            columns=["key", "reference"],
        ),
        on="key",
    )
    on_source = matches(sides, source_texts, near_matches)
    on_target = matches(sides, target_texts, near_matches)
    both = on_source.merge(
        on_target, on=["pair", "benchmark", "row"], suffixes=("_source", "_target")
    )
    both = both[both.side_source != both.side_target]

    def described(end):
        """One of the two matches as text, such as ``anchor name = source name``."""
        parts = [both[f"{part}_{end}"] for part in ("side_field", "how", "field")]
        return both[f"side_{end}"].str.cat(parts, sep=" ")

    both = both.assign(
        fields=described("source") + ", " + described("target"),
        source_variable_name=both.variable_source,
        target_property=both.target,
    )
    both = both.sort_values(["pair", "benchmark", "row", "target", "fields"])
    return both.drop_duplicates(["pair", "benchmark"])[
        [
            "pair",
            "benchmark",
            "row",
            "source_variable_name",
            "target_property",
            "fields",
        ]
    ]


def study_matches(pairs, studies):
    """The pairs whose ``study`` holds a benchmark row's source study.

    Args:
        pairs (pd.DataFrame): The pairs, with ``study``: dbGaP studies joined
            by ``|``.
        studies (pd.DataFrame): ``benchmark`` and ``study_id``, one row per
            benchmark and study.

    Returns:
        pd.DataFrame: ``pair``, ``benchmark`` and ``study_id``, the study that
            the pair and the benchmark share, one row per pair and benchmark.
    """
    named = pairs.study[pairs.study != ""].str.split("|").explode()
    matched = (
        pd.DataFrame({"pair": named.index, "study_id": named.to_numpy()})
        .merge(studies, on="study_id")
        .sort_values(["pair", "benchmark", "study_id"])
    )
    return matched.drop_duplicates(["pair", "benchmark"])


def shared_vocabulary(kept_sides, source_texts, target_texts):
    """Benchmark texts that the kept pairs also hold, on their own.

    Args:
        kept_sides (pd.DataFrame): The side texts of the pairs that leakage
            prevention keeps, from ``side_keys``.
        source_texts (pd.DataFrame): Every benchmark's source texts.
        target_texts (pd.DataFrame): Every benchmark's expected-target texts.

    Returns:
        dict: Per benchmark: ``rows_with_shared_source``,
            ``rows_with_shared_target``, and ``most_common``, the shared
            texts that the most benchmark rows hold, each with that count.
    """
    held = set(kept_sides.key)
    vocabulary = {}
    for benchmark in source_texts.benchmark.unique():
        shared_sources = source_texts[
            (source_texts.benchmark == benchmark) & source_texts.key.isin(held)
        ]
        shared_targets = target_texts[
            (target_texts.benchmark == benchmark) & target_texts.key.isin(held)
        ]
        shared_texts = pd.concat([shared_sources, shared_targets]).drop_duplicates(
            ["row", "key"]
        )
        common = shared_texts.key.value_counts().head(SHARED_SHOWN)
        vocabulary[benchmark] = {
            "rows_with_shared_source": int(shared_sources.row.nunique()),
            "rows_with_shared_target": int(shared_targets.row.nunique()),
            "most_common": [
                {"text": text, "rows": int(count)} for text, count in common.items()
            ],
        }
    return vocabulary


def audit(pairs, benchmarks):
    """Find the pairs that leak a benchmark, and the vocabulary shared with one.

    Args:
        pairs (pd.DataFrame): The combined pairs, with the field columns and
            ``study``.
        benchmarks (dict[str, tuple[pd.DataFrame, pd.DataFrame]]): Each
            benchmark's targets and mappings, from ``read_benchmark``.

    Returns:
        tuple[pd.DataFrame, dict]: The findings, one row per pair and
            benchmark that it leaks, in COLUMNS. An answer key comes before a
            benchmark study, so a pair has one finding per benchmark. An answer
            key also has the benchmark's own descriptions of its variable and
            its target. And the shared vocabulary, per benchmark.
    """
    per_benchmark = [
        benchmark_texts(name, *tables) for name, tables in benchmarks.items()
    ]
    source_texts = pd.concat([texts[0] for texts in per_benchmark], ignore_index=True)
    target_texts = pd.concat([texts[1] for texts in per_benchmark], ignore_index=True)
    studies = pd.concat(
        [
            pd.DataFrame({"benchmark": name, "study_id": mappings.source_study_id})
            for name, (_, mappings) in benchmarks.items()
        ]
    )
    studies = studies[studies.study_id != ""].drop_duplicates()
    sides = side_keys(pairs)
    findings = pd.concat(
        [
            answer_keys(sides, source_texts, target_texts).assign(finding=ANSWER_KEY),
            study_matches(pairs, studies).assign(finding=BENCHMARK_STUDY),
        ],
        ignore_index=True,
    ).drop_duplicates(["pair", "benchmark"])
    source_descriptions = pd.concat(
        [
            pd.DataFrame(
                {
                    "benchmark": name,
                    "row": pd.array(range(len(mappings)), dtype="Int64"),
                    "source_variable_description": (
                        mappings.source_description.to_numpy()
                    ),
                }
            )
            for name, (_, mappings) in benchmarks.items()
        ]
    )
    target_descriptions = pd.concat(
        [
            pd.DataFrame(
                {
                    "benchmark": name,
                    "target_property": targets.target_property.to_numpy(),
                    "target_description": targets.target_description.to_numpy(),
                }
            )
            for name, (targets, _) in benchmarks.items()
        ]
    )
    findings = (
        findings.astype({"row": "Int64"})
        .merge(source_descriptions, on=["benchmark", "row"], how="left")
        .merge(target_descriptions, on=["benchmark", "target_property"], how="left")
    )
    findings = findings.reindex(columns=COLUMNS).sort_values(["pair", "benchmark"])
    texts_only = [column for column in COLUMNS if column not in ("pair", "row")]
    findings[texts_only] = findings[texts_only].fillna("")
    findings = findings.astype({"pair": "int64", "row": "Int64"}).reset_index(drop=True)
    kept_sides = sides[~sides.pair.isin(findings.pair)]
    return findings, shared_vocabulary(kept_sides, source_texts, target_texts)
