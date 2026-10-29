# Leakage prevention

`corpus build` drops every training pair that would leak a benchmark. It does
so after it combines the sources, and before it splits the pairs into training
and validation pairs (see [corpus.md](corpus.md)). This doc gives the rules.

A pair leaks a benchmark when it holds one of the benchmark's mappings, or
when it comes from one of the benchmark's studies. A model trained on such a
pair has seen the benchmark, and its score there would look better than it
is. So leakage prevention compares the pairs with every benchmark built under
`embedding_model/inputs/benchmarks/`, and drops each pair that leaks. Without
a built benchmark, the corpus is not built.

A finding is a reason to drop a pair, found against one benchmark. There are
two kinds:

| finding | what it means | its rule |
|---|---|---|
| `answer_key` | the pair holds a mapping: a source variable on one side, one of its expected targets on the other | [the answer key rule](#the-answer-key-rule-answer_key) |
| `benchmark_study` | the pair comes from a dbGaP variable of a benchmark's study | [the study rule](#the-study-rule-benchmark_study) |

Each benchmark is checked on its own, with the same rules, and a pair is
dropped if any benchmark finds it. Against one benchmark, a pair has at most
one finding: an answer key comes before a benchmark study. A pair that leaks
two benchmarks has a finding for each. A benchmark text in a pair is not
leakage by itself: [shared vocabulary](#shared-vocabulary) is kept.

## The answer key rule: `answer_key`

A pair is an answer key when its two sides match one benchmark row:

- one side's name or description matches the source variable's name or
  description;
- the other side's name or description matches one of that variable's
  expected targets, by the target's property name or description.

Either side can match the source. For example, a benchmark maps the variable
`body_weight` to `vitals.weight`. The pair `Weight` -> `Body Weight` is then
an answer key, and it is dropped.

### The fields compared

Each side of a pair is a text formatted from its fields: a name, a type, a
description and values. For example, the anchor `ABCD_SC (integer)` is
formatted from the name `ABCD_SC` and the type `integer`. Leakage prevention
compares the name and description fields rather than the formatted texts. The
same name can come with another type or other values, and its formatted text
then differs. The fields compared are:

| side | fields compared |
|---|---|
| each side of a pair, the anchor and the positive | its name and its description, the pair's side texts |
| a benchmark's source variable | its name and its description |
| each of its expected targets | its property name, without the node, and its description |

Every side text is compared, since some pairs keep a description in a name
field. Values are never compared: a common list, such as
`Unknown | Not Reported | No | Yes`, matches many targets.

### How two texts match

The benchmark texts are cleaned first with `pairs.common.clean`, the
cleaning that `pairs prepare` gives every text: markup is removed, and
characters and spaces are normalised. Then both texts go through three
changes:

- the tokenizer's normalisation, which removes invisible characters,
  lowercases and removes accents;
- case folding, which also makes `ß` and `ss` equal;
- punctuation turned into spaces.

After these changes, two texts match in one of three ways. Each way has a
mark:

| way | mark | for |
|---|---|---|
| equal | `=` | any text |
| at least 8 words in a row | `~` | any text |
| a name's words, with a description on the other side | `>=` | a name of two words or more |

The marks show in `dropped.parquet`, the corpus's file of dropped pairs (see
[What a corpus holds](corpus.md#what-a-corpus-holds)). There, the column `fields` names
each pair's two matches, the source's first. Each match reads as follows:

| in `fields` | it reads |
|---|---|
| `anchor name = source name` | the anchor's name equals the source variable's name |
| `positive description ~ target description` | the positive's description shares at least 8 words in a row with the target's description |
| `anchor name >= source name` | the anchor's name holds every word of the source variable's name |

So `anchor name >= source name, positive description = target description`
means that the anchor holds the words of the source's name, and the positive
holds the target's description.

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
| `The total number of lymph nodes that were examined by the pathologist. Includes nodes from every site.` | a benchmark target's description |

They share 10 words in a row: `the total number of lymph nodes that were
examined by`. So the two texts match.

#### A name's words, with a description on the other side

Most names are shorter than 8 words, so a name has one more way to match. Say
that one side of a pair matches a description of a benchmark row, equal or by
at least 8 words in a row. The other side then matches a name of that row when it holds every word
of the name, in any order. The name can be the source's or a target's. For
example, a benchmark row maps the source variable `Waist.cm` to a target
described as "The waist circumference of the participant in centimeters.":

| side | text | it matches |
|---|---|---|
| anchor | `Waist in cm` | the source name `Waist.cm`, by its words `waist` and `cm` |
| positive | `Participant Waist Circumference: The waist circumference of the participant in centimeters.` | the target's description, equal |

So the pair is an answer key. Two limits keep general vocabulary:

- a name of one word, such as `age`, matches only an equal text;
- without a description on the other side, a name's words do not count. So
  `Waist in cm` -> `Waist` is kept, since `Waist` matches only the target's
  name.

## The study rule: `benchmark_study`

A pair gets the finding `benchmark_study` when its `study` holds the source
study of a benchmark row. For example, a benchmark row comes from the study
`phs999901`. Every pair made from a dbGaP variable of `phs999901` is then
dropped. The study rule catches what text matching cannot see: the other
variables of a benchmark's studies.

A pair that several sources share is kept once, and it takes the studies of
all its copies. So the study rule also sees a copy that came from a
benchmark's study. [Combining the sources](corpus.md#combining-the-sources) describes
the copies.

## Shared vocabulary

A benchmark text can also occur in a pair without a finding. For example,
`age` is one side of a pair whose other side matches no expected target of
that benchmark row. That is not leakage: the pair is kept, and the manifest
counts the text as shared vocabulary.

## What a build records

Each dropped pair has a row in `dropped.parquet`, with its finding. The
corpus's manifest counts the findings and the shared vocabulary.
[What a corpus holds](corpus.md#what-a-corpus-holds) lists the columns and the
counts.
