# Leakage prevention

`corpus build` drops every pair that would leak a benchmark. It does so after
it combines the sources, and before it splits the pairs into training and
validation pairs. [corpus.md](corpus.md) describes the three steps of
`corpus build`. This doc gives the rules.
[design_decisions.md](design_decisions.md#corpus) gives the reasons for them.

A pair leaks a benchmark when it holds one of the benchmark's
[mappings](benchmarks.md#benchmark_mappingscsv), or when it comes from one of
the benchmark's studies. A model trained on such a pair has seen the
benchmark, and the model's score on that benchmark would look better than it
is. So leakage prevention compares the pairs with every benchmark built under
`embedding_model/inputs/benchmarks/`, and drops each pair that leaks. If no
benchmark is built, `corpus build` builds no corpus.

A finding is a reason to drop a pair, found against one benchmark. There are
two kinds:

| finding | what it means | its rule |
|---|---|---|
| `answer_key` | the pair holds a mapping: a source variable on one side, one of its expected targets on the other | [the answer key rule](#the-answer-key-rule-answer_key) |
| `benchmark_study` | the pair comes from a [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) variable of a benchmark's study | [the study rule](#the-study-rule-benchmark_study) |

Leakage prevention checks each benchmark on its own, with the same rules. It
drops a pair when the pair has a finding against any benchmark. Against one
benchmark, a pair has at most one finding: an answer key comes before a
benchmark study. A pair that leaks two benchmarks has a finding for each
benchmark. A benchmark text in a pair is not leakage by itself, so
[shared vocabulary](#shared-vocabulary) stays in the corpus.

## The answer key rule: `answer_key`

A pair is an answer key when its two sides match one benchmark row, a row of
[`<benchmark>_mappings.csv`](benchmarks.md#benchmark_mappingscsv):

- one side's name or description matches the source variable's name or
  description
- the other side's name or description matches one of that variable's
  expected targets, by the target's property name or description

Either side can match the source. For example, a benchmark maps the variable
`body_weight` to `vitals.weight`. The pair `Weight` -> `Body Weight` is then
an answer key, and leakage prevention drops it.

### The fields compared

Each side of a pair is a text formatted from its fields: a name, a type, a
description and values. For example, the anchor `ABCD_SC (integer)` is
formatted from the name `ABCD_SC` and the type `integer`. Leakage prevention
compares the name and description fields rather than the formatted texts.
The fields compared are:

| side | fields compared |
|---|---|
| each side of a pair, the anchor and the positive | its name and its description, the pair's side texts |
| a benchmark's source variable | its name and its description |
| each of its expected targets | its property name, without the node, and its description |

Leakage prevention compares each side text with the benchmark's names and
descriptions, whatever the side text's field. It never compares values.

### How two texts match

Leakage prevention first cleans the benchmark texts with `pairs.common.clean`,
the cleaning that `pairs prepare` gives every text. The cleaning removes
markup, and normalises characters and spaces. Then the side text and the
benchmark text both go through three changes:

- the tokenizer's normalisation, which removes invisible characters,
  lowercases and removes accents
- case folding, which also makes `ß` and `ss` equal
- punctuation turned into spaces

These changes give each text its `text_key`, as `pairs.common.text_key`
writes it. Two texts match by their `text_key` in one of three ways. Each way
has a mark:

| way | mark | for |
|---|---|---|
| equal | `=` | any text |
| at least 8 words in a row | `~` | any text |
| a name's words, with a description on the other side | `>=` | a name of two different words or more |

The marks appear in `dropped.parquet`, the corpus's file of dropped pairs.
[What `corpus build` writes](corpus.md#what-corpus-build-writes) describes the
file. In `dropped.parquet`, the column `fields` names the two matches of each
pair, the match with the source first. Each match reads as follows:

| in `fields` | it reads |
|---|---|
| `anchor name = source name` | the anchor's name equals the source variable's name |
| `positive description ~ target description` | the positive's description shares at least 8 words in a row with the target's description |
| `anchor name >= source name` | the anchor's name holds every word of the source variable's name |

So `anchor name >= source name, positive description = target description`
means that:

- the anchor's name holds every word of the source variable's name
- the positive's description equals the target's description

#### Equal

The two texts are the same after the changes. So two texts always match when
the tokenizer normalises them to one text.

#### At least 8 words in a row

A side text matches a benchmark text when the two share at least 8 words in a
row. For example, a benchmark copies a terminology's definition,
changes a word, and adds a sentence:

| text | from |
|---|---|
| `The total number of lymph nodes that were examined by a pathologist in the surgical specimen.` | a pair's positive description |
| `The total number of lymph nodes that were examined by the pathologist in the surgical specimen. Includes nodes from every site.` | a benchmark target's description |

They share 10 words in a row: `the total number of lymph nodes that were
examined by`. So the two texts match.

#### A name's words, with a description on the other side

Most names are shorter than 8 words, so a name has one more way to match: by
its words. One side of a pair matches a name of a benchmark row by the name's
words when all of these hold:

- the name has two different words or more
- the side holds every word of the name, in any order
- the pair's other side matches the row's other description, equal or by at
  least 8 words in a row:
  - for the source's name, an expected target's description
  - for a target's name, the source variable's description

The name can be the source's or a target's. For example, a benchmark row maps
the source variable `Waist.cm` to a target described as "The waist
circumference of the participant in centimeters.":

| side | text | it matches |
|---|---|---|
| anchor | `Waist in cm` | the source name `Waist.cm`, by its words `waist` and `cm` |
| positive | `Participant Waist Circumference: The waist circumference of the participant in centimeters.` | the target's description, equal |

So the pair is an answer key. The first and the last conditions keep general
vocabulary:

- a name of one word, for example `age`, matches only an equal text
- without the row's other description on the other side, a name's words do
  not count. So
  `Waist in cm` -> `Waist` is kept, because `Waist` does not match the
  target's description.

### An answer key's twin

An anchor gives two pairs when its [canonical](pairs.md#the-canonical) has a
description. The `name` pair holds the canonical alone, and the
`name_description` pair holds the canonical and its description. Often only
the second pair matches a benchmark row, by its description. A pair whose
anchor name and positive name read as an answer key's is the answer key's
twin. Leakage prevention drops the twin with the answer key, for the same
benchmark row.

Leakage prevention compares the names by their `text_key`, which ignores
case, accents and punctuation. So a pair of another item or source can be a
twin too. In `dropped.parquet`, a twin's `fields` are its answer key's
`fields`, after `twin of: `. For example:

```
twin of: anchor name = source name, positive description = target description
```

The anchor `Waist in cm` of the example above gives two pairs:

| kind | positive | finding |
|---|---|---|
| `name_description` | `Participant Waist Circumference: The waist circumference of the participant in centimeters.` | an answer key, as above |
| `name` | `Participant Waist Circumference` | its twin, dropped with it |

## The study rule: `benchmark_study`

A pair gets the finding `benchmark_study` when its `study` holds the source
study of a benchmark row. For example, a benchmark row comes from the study
`phs999901`. Leakage prevention then drops every pair made from a
[dbGaP](https://www.ncbi.nlm.nih.gov/gap/) variable of `phs999901`. Only
PhenX's pairs have a study: PhenX maps its variables to dbGaP study variables,
as [pairs.md](pairs.md#phenx) describes.

A pair that several sources share is kept once, and it takes the studies of
all its copies. So the study rule also sees a copy that came from a
benchmark's study. [Combining the sources](corpus.md#combining-the-sources)
describes the copies.

## Shared vocabulary

A benchmark text can also occur in a pair without a finding. For example,
`age` is one side of a pair whose other side matches no expected target of
that benchmark row. That is not leakage. Leakage prevention keeps the pair,
and the corpus's manifest counts the text as shared vocabulary.

## What a build writes

Each dropped pair has a row in `dropped.parquet`, with its finding. The
corpus's manifest counts the findings and the shared vocabulary.
[What `corpus build` writes](corpus.md#what-corpus-build-writes) lists the
columns and the counts.
