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
tokenizer normalises them, case-folded, with punctuation as spaces. A side
text also matches any benchmark text, a name or a description, when the two
keys share at least MIN_OVERLAP_WORDS words in a row. A pair can hold a
description in a name field, such as a ``name_to_description`` pair's
positive, so every side text is compared, whatever its field.

A name also matches by its words, with a description on the other side. When
one side of a pair matches a benchmark row's description, the other side also
matches a name of that row when it holds every word of the name, in any order,
and the name has MIN_NAME_WORDS words or more. For example, the anchor
``Waist in cm`` holds the words of the source name ``Waist.cm``. It matches
that name when the positive holds the target's description.

A finding is a reason to drop a pair, found against one benchmark. There are
two kinds:

- ``answer_key``: one side's name or description matches the source
  variable's name or description, and the other side's name or description
  matches one of that variable's expected targets, by its property name or
  description. Either side can match the source.
  For example, a benchmark maps the variable ``body_weight`` to
  ``vitals.weight``. Then the pair ``Weight`` -> ``Body Weight`` is an answer
  key: its positive matches the source variable, and its anchor matches the
  expected target. A pair with the same item, anchor and canonical as an
  answer key goes with it: each anchor gives a ``name`` pair and a
  ``name_description`` pair, and often only the second holds the description
  that matches.
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
from embedding_model.pairs.common import STUDY_SEPARATOR, clean, key

# A side text overlaps a benchmark text when the two share at least this many
# words in a row. GPT-3's authors check for their test data in their training
# data the same way (Brown et al., 2020, Appendix C). Their rule gives 8 words
# for texts as short as the benchmarks' descriptions. A shorter text, such as
# most names, matches only an equal text or, with a description on the other
# side, by its words.
MIN_OVERLAP_WORDS = 8
# The minimum number of words of a benchmark name that a side can match by its
# words. A one-word name, such as "age", matches only an equal text, since a
# side that holds one shared word is mostly general vocabulary.
MIN_NAME_WORDS = 2
# What a twin's ``fields`` start with, before its answer key's own fields.
TWIN = "twin of: "
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


def runs(words, length):
    """The runs of ``length`` words in a row in a list of words, in order."""
    return [" ".join(words[i : i + length]) for i in range(len(words) - length + 1)]


def overlaps(texts, references, length):
    """The texts and references that share ``length`` words in a row.

    The words are whole words, as ``str.split`` cuts a key. Each reference is
    indexed by its runs of ``length`` words, and each text looks up its own.

    Args:
        texts (Iterable[str]): Keys, as ``keys`` writes them.
        references (Iterable[str]): Keys to compare them with.
        length (int): The words in a run, such as MIN_OVERLAP_WORDS.

    Returns:
        set[tuple[str, str]]: (text, reference), for each text and reference
            that share a run and are not already equal keys.
    """
    index = defaultdict(set)
    for reference in set(references):
        for run in runs(reference.split(), length):
            index[run].add(reference)
    matched = set()
    for text in set(texts):
        for run in runs(text.split(), length):
            matched.update(
                (text, reference)
                for reference in index.get(run, ())
                if reference != text
            )
    return matched


def word_matches(texts, names, min_words):
    """The texts and names where the text holds every word of the name.

    The words may come in any order. Only names of ``min_words`` words or
    more are matched. Each text is indexed by its words, and a name looks up
    the texts that hold its rarest word, then keeps those that hold the rest.

    Args:
        texts (Iterable[str]): Keys, as ``keys`` writes them.
        names (Iterable[str]): Benchmark names' keys.
        min_words (int): The minimum number of words of a name that can
            match, such as MIN_NAME_WORDS.

    Returns:
        set[tuple[str, str]]: (text, name), for each text that holds the
            name's words and is not already the name's equal key.
    """
    holding = defaultdict(set)
    for text in set(texts):
        for word in set(text.split()):
            holding[word].add(text)
    matched = set()
    for name in set(names):
        words = sorted(set(name.split()), key=lambda word: len(holding[word]))
        if len(words) < min_words:
            continue
        found = set(holding[words[0]])
        for word in words[1:]:
            found &= holding[word]
        matched.update((text, name) for text in found if text != name)
    return matched


def matches(sides, texts):
    """Every match of a side text with a benchmark text, in each of the three ways.

    The three ways are an equal key (``=``), at least MIN_OVERLAP_WORDS words
    in a row (``~``), and a name's words (``>=``), for names of MIN_NAME_WORDS
    words or more.

    Args:
        sides (pd.DataFrame): From ``side_keys``.
        texts (pd.DataFrame): Benchmark texts, from ``benchmark_texts``: the
            source texts and the target texts together.

    Returns:
        pd.DataFrame: One row per match: the side's columns, the benchmark
            text's columns, and ``how``, the way that it matched.
    """
    names = texts[texts.field.str.endswith("name")]

    def joined(matched_keys, candidates, how):
        """The side texts and benchmark texts of (side key, benchmark key) tuples."""
        matched = pd.DataFrame(sorted(matched_keys), columns=["key", "reference"])
        return (
            sides.merge(matched, on="key")
            .drop(columns="key")
            .merge(candidates.rename(columns={"key": "reference"}), on="reference")
            .drop(columns="reference")
            .assign(how=how)
        )

    return pd.concat(
        [
            sides.merge(texts, on="key").assign(how="="),
            joined(overlaps(sides.key, texts.key, MIN_OVERLAP_WORDS), texts, "~"),
            joined(word_matches(sides.key, names.key, MIN_NAME_WORDS), names, ">="),
        ],
        ignore_index=True,
    )


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
    side_matches = matches(sides, pd.concat([source_texts, target_texts]))
    on_source = side_matches[side_matches.field.str.startswith("source")]
    on_target = side_matches[side_matches.field.str.startswith("target")]
    both = on_source.merge(
        on_target, on=["pair", "benchmark", "row"], suffixes=("_source", "_target")
    )
    # The source and the target must match on two sides of the pair. One side
    # that matches both, as "Race" matches the source "Race" and the target
    # "race", is shared vocabulary.
    both = both[both.side_source != both.side_target]
    # A name's words count only when the pair's other side matched a
    # description of the same row. That needs both sides, so it is checked
    # here, once they are joined.
    source_counts = (both.how_source != ">=") | both.field_target.str.endswith(
        "description"
    )
    target_counts = (both.how_target != ">=") | both.field_source.str.endswith(
        "description"
    )
    both = both[source_counts & target_counts]

    def as_text(end):
        """One of the two matches as text, such as ``anchor name = source name``."""
        parts = [both[f"{part}_{end}"] for part in ("side_field", "how", "field")]
        return both[f"side_{end}"].str.cat(parts, sep=" ")

    both = both.assign(
        fields=as_text("source") + ", " + as_text("target"),
        source_variable_name=both.variable_source,
        target_property=both.target_target,
    )
    both = both.sort_values(["pair", "benchmark", "row", "target_property", "fields"])
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


def answer_key_twins(pairs, answer_key_findings):
    """The pairs with the same item, anchor and canonical as an answer key.

    An anchor gives a ``name`` pair and a ``name_description`` pair. The
    second holds the canonical's description, which can match the target
    while the first matches nothing. Kept, the first would still pair the
    source's text with the canonical of the answer key.

    Args:
        pairs (pd.DataFrame): The combined pairs.
        answer_key_findings (pd.DataFrame): The answer keys, from
            ``answer_keys``.

    Returns:
        pd.DataFrame: The twins that are not answer keys themselves, in the
            columns of ``answer_keys``. A twin's ``fields`` are its answer
            key's, after TWIN, such as ``twin of: anchor name = source name,
            positive description = target description``.
    """
    identity = ["source", "item", "anchor_name", "positive_name"]
    identities = pairs[identity].assign(pair=pairs.index)
    twins = (
        answer_key_findings.merge(identities, on="pair")
        .drop(columns="pair")
        .merge(identities, on=identity)
    )
    twins = twins[~twins.pair.isin(answer_key_findings.pair)]
    return twins.assign(fields=TWIN + twins.fields).drop_duplicates(
        ["pair", "benchmark"]
    )[list(answer_key_findings.columns)]


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
    named = pairs.study[pairs.study != ""].str.split(STUDY_SEPARATOR).explode()
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
    answer_key_findings = answer_keys(sides, source_texts, target_texts)
    findings = pd.concat(
        [
            answer_key_findings.assign(finding=ANSWER_KEY),
            answer_key_twins(pairs, answer_key_findings).assign(finding=ANSWER_KEY),
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
