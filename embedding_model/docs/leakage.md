# Leakage prevention

`corpus build` drops every pair that would leak a benchmark, before it splits
the pairs into training and validation pairs. A pair leaks a benchmark when
it holds one of the benchmark's
[mappings](benchmarks.md#benchmark_mappingscsv), or when it comes from one of
the benchmark's studies. Leakage prevention compares the pairs with every
benchmark built under `embedding_model/inputs/benchmarks/`.

A finding is the reason that `corpus build` drops a pair, for one benchmark.
Each dropped pair has a row in `dropped.parquet` for each benchmark that it
leaks, with its finding.
[What `corpus build` writes](corpus.md#what-corpus-build-writes) lists the
file's columns. There are two kinds of finding:

| finding | what it means | its rule |
|---|---|---|
| `answer_key` | the pair holds a mapping: a source variable on one side, one of its expected targets on the other | [the answer key rule](#the-answer-key-rule-answer_key) |
| `benchmark_study` | the pair comes from a [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) variable of a benchmark's study | [the study rule](#the-study-rule-benchmark_study) |

## The answer key rule: `answer_key`

A pair is an answer key when its two sides match one benchmark row, a row of
[`<benchmark>_mappings.csv`](benchmarks.md#benchmark_mappingscsv):

- one side's name or description matches the source variable's name or
  description
- the other side's name or description matches one of that variable's
  expected targets, by the target's property name or description

Either side can match the source variable. For example, a benchmark maps the
variable `body_weight` to `vitals.weight`. The pair `Weight` -> `Body Weight`,
written as anchor -> positive, is then an answer key:

- its positive, `Body Weight`, matches the source variable's name
  `body_weight`
- its anchor, `Weight`, matches the target's property name `weight`

Leakage prevention drops the pair.

### The fields compared

Each side of a pair is a text written from its fields: a name, a type, a
description and values. Leakage prevention compares the fields in this table,
not the written texts. It never compares values.

| side | fields compared |
|---|---|
| each side of a pair, the anchor and the positive | its name and its description |
| a benchmark's source variable | its name and its description |
| each of its expected targets | its property name, without the node, and its description |

### How two texts match

Two texts match by their `text_key`: the text after the tokenizer's
normalisation, in lower case, with punctuation turned into spaces. They match
in one of three ways, each with a mark:

| way | mark | the two texts match when |
|---|---|---|
| equal | `=` | their `text_key` is the same |
| at least 8 words in a row | `~` | they share at least 8 words in a row |
| a name's words | `>=` | one side holds every word of a benchmark name, in any order. The name has two different words or more. The other side matches a description of the row by `=` or `~`. For the source variable's name, that is a target's description. For a target's name, it is the source variable's description. |

For example, the anchor `Waist in cm` holds the words of the source variable's
name `Waist.cm`. The pair is an answer key when its positive's description
matches the target's description.

In `dropped.parquet`, the column `fields` names the two matches of each pair,
the match with the source variable first. So
`anchor name >= source name, positive description = target description` means
that:

- the anchor's name holds every word of the source variable's name
- the positive's description equals the target's description

### An answer key's twin

A twin is a pair whose anchor name and positive name have the same
`text_key` values as an answer key's anchor name and positive name. Leakage
prevention drops the twin with the answer key, for the same benchmark row. In
`dropped.parquet`, a twin's `fields` start with `twin of: `, followed by the
answer key's `fields`.

## The study rule: `benchmark_study`

A pair gets the finding `benchmark_study` when its `study` holds the
`source_study_id` of a benchmark row, in
[`<benchmark>_mappings.csv`](benchmarks.md#benchmark_mappingscsv). For
example, a benchmark row comes from the study `phs999901`. Leakage prevention
then drops every pair made from a dbGaP variable of `phs999901`. A pair that is
also an answer key of the same benchmark gets the finding `answer_key`.

## Shared vocabulary

A benchmark text in a pair is not leakage by itself. For example, a
benchmark row has the source variable `age`. A pair has `age` on one side,
but its other side matches none of the row's expected targets. Leakage
prevention keeps the pair, and the corpus's manifest counts the text as
shared vocabulary.
