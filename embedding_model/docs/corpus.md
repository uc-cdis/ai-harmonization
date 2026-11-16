# Corpus

`corpus build` combines the [pairs](pairs.md) of the [sources](sources.md)
into one training corpus. It works in three steps, in this order:

1. combine the sources, and keep only one copy of a pair that two sources share
2. drop the pairs that would [leak](leakage.md) a [benchmark](benchmarks.md)
3. split the kept pairs into training and validation pairs

The split sees only the pairs that [leakage prevention](leakage.md), step 2,
keeps. The later steps train the tokenizer on the training pairs alone.

`corpus build` writes the kept pairs, the dropped pairs and the corpus's
[manifest](manifests.md) to the corpus's folder.
[What `corpus build` writes](#what-corpus-build-writes) lists the files.
[Details](#details) describes each of the three steps.

[design_decisions.md](design_decisions.md#corpus) gives the reasons for the
rules below.

## Running `corpus build`

```
python -m embedding_model corpus build --name all-sources              # every source
python -m embedding_model corpus build --name example --source nhanes  # some of them
```

`corpus build` prints how many pairs each step keeps and drops, and the
corpus's manifest holds the counts. The counts depend on the sources and the
benchmarks of the build.

`corpus build` stops, and writes no corpus, when:

- the name is not a folder name, or a source is not one of the
  [sources](pairs.md#each-source)
- `--validation-share` is not between 0.001 and 0.999, in steps of 0.001, as
  [The split](#the-split) describes
- a benchmark cannot be read, or no benchmark is built, so the pairs cannot
  be checked for leakage
- a source's pairs cannot be used, as
  [Combining the sources](#combining-the-sources) describes
- the sources of the build hold no pairs
- leakage prevention drops every pair

## What `corpus build` writes

`corpus build` writes to `embedding_model/inputs/corpora/<name>/`, or under the
folder that `--inputs` names. It writes the folder whole. So if a build fails,
the previous corpus stays as it was.
[Inputs and outputs](../README.md#inputs-and-outputs) describes how a step
writes a folder whole.

```
corpora/<name>/
    pairs.parquet      the kept pairs, as training and validation pairs
    dropped.parquet    the dropped pairs, each with its finding
    manifest.json      the corpus's manifest
```

### `pairs.parquet`

The kept pairs, with the columns that [pairs.md](pairs.md#pairsparquet)
lists, and `split`. `split` is `train` or `validation`: the pair's side of
[the split](#the-split). A pair that several sources hold is kept once, and
its `study` holds the studies of every copy.

### `dropped.parquet`

Every dropped pair, with the columns of `pairs.parquet` except `split`, and
the columns of its [finding](leakage.md). A pair that leaks two benchmarks
has a row for each benchmark. The finding's columns are:

- `finding`: `answer_key` or `benchmark_study`
- `benchmark`: the benchmark that the pair leaks
- for an answer key:
  - `benchmark_row`: the position of the matched row in
    `<benchmark>_mappings.csv`, counted from 0
  - `benchmark_source_variable_name` and
    `benchmark_source_variable_description`: the source variable, as the
    benchmark gives it
  - `benchmark_target_property` and `benchmark_target_description`: the
    expected target that matched
  - `fields`: the two matches, each from a field of the pair to a field of
    the benchmark, for example `anchor name = source name, positive name =
    target name`. [How two texts match](leakage.md#how-two-texts-match) says
    how to read `fields`.
- for the study rule: `benchmark_study_id`, the benchmark's study that the
  pair holds

### `manifest.json`

The corpus's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: `pairs.parquet` and `dropped.parquet`, each with its row count,
  its file's `sha256` and its table's `content_sha256`
- `sources`: each source's pairs, with the source's version and the pairs'
  checksums, and each benchmark, with its version and its files' checksums
- `settings`:
  - `format`: the [text format](scoring.md#text-formats) that every source's
    pairs share
  - `min_overlap_words`: for a match by
    [words in a row](leakage.md#at-least-8-words-in-a-row), the minimum
    number of words in a row that the two texts share
  - `min_name_words`: for a match by
    [a name's words](leakage.md#a-names-words-with-a-description-on-the-other-side),
    the minimum number of different words in the benchmark name
  - `validation_share` and `buckets`: the share of groups that
    [the split](#the-split) holds out for validation, and the number of hash
    buckets
- `counts`:
  - `rows`:
    - `combined`: the pairs of all the sources
    - `duplicates_removed`: the copies of an earlier source's pairs
    - `leakage_removed`: the pairs that leakage prevention drops
    - `kept`: the pairs kept
    - `train` and `validation`: the kept pairs on each side of the split
  - `duplicates_by_source`: the copies removed, by the source that held them
  - `leakage_findings`: the findings, counted by finding, benchmark, source
    and fields. A pair that leaks two benchmarks counts twice.
  - `shared_vocabulary`: for each benchmark, its
    [shared vocabulary](leakage.md#shared-vocabulary):
    - `rows_with_shared_source`: how many of the benchmark's rows share a
      source text with the kept pairs
    - `rows_with_shared_target`: how many of the benchmark's rows share a
      target text with the kept pairs
    - `most_common`: the 20 most common shared texts, each with the number of
      benchmark rows that hold it

## Details

### Combining the sources

`corpus build` reads the sources in the order that the
[table in pairs.md](pairs.md#each-source) lists. Each source's pairs must have
the checksum that their manifest holds. When the source's fetch is in the
inputs folder, the pairs must come from that fetch. The manifest must also
record the text format of the pairs, one of the formats that
[scoring.md](scoring.md#text-formats) lists. All the sources' pairs must be
in one format.

The build names each source whose pairs fail a check, with the command to
run, and then stops. The command prepares the pairs again. When the fetch's
manifest cannot be read, is not a JSON object, or is another source's, the
command fetches the source again.

A pair that copies an earlier source's pair is kept once, in the earlier
source. Two pairs are copies when the tokenizer
[normalises](tokenizer.md#step-1-normalize) their texts to the same two texts.
[pairs.md](pairs.md#duplicates) shows examples. The pair kept takes the
studies of all its copies. So the
[study rule](leakage.md#the-study-rule-benchmark_study) still sees the
studies of the copies.

### Leakage prevention

Leakage prevention drops each pair that would leak a benchmark. Such a pair
holds one of the benchmark's mappings, or comes from one of the benchmark's
studies. Leakage prevention compares the pairs with every benchmark built
under `embedding_model/inputs/benchmarks/`. [leakage.md](leakage.md) gives
the rules, with examples.

### The split

The split puts all pairs of a group on the same side, training or
validation. A group holds one or more
[items](pairs.md#the-shape-a-star-around-each-item), with all their pairs.
Two items are in one group when they share:

- an `item_id`: an item's source and id, for example `cadsr:9999901`.
  [NCIt](sources.md#ncit)'s concepts and [CDISC](sources.md#cdisc)'s terms
  both take NCIt's concept code, for example `ncit:C999901`. So a CDISC term
  joins its NCIt concept. The other sources' `item_id` values never match
  across sources.
- or a [canonical](pairs.md#the-canonical) text's `text_key`, as
  [leakage prevention](leakage.md#how-two-texts-match) compares texts: the
  text after the tokenizer's normalisation, with case folded and punctuation
  turned into spaces. [Each source](pairs.md#each-source) lists each source's
  canonical. A canonical made of punctuation alone links nothing.

For example, one item's canonical is `Body Weight`, and another's is
`body-weight`. Compared by their `text_key`, they are one text.

Items can also link through other items. For example, item A shares a text
with item B, and B shares a code with C. Then A, B and C are one group. So
the same concept never has pairs on both sides. A pair that two sources share
is kept once, in the earlier source's item, so it goes to that item's group.

Validation holds 2% of the groups by default. `--validation-share` sets
another share, in steps of 0.001, for example `--validation-share 0.1`. A
small corpus can use a larger share, so that validation holds enough pairs.

A hash of the group decides its side, with no random draw and no seed. The
hash reads the group's `group_id`: the first `item_id` of its items in sorted
order, for example `cadsr:9999901` or `ncit:C999901`. The hash puts each group
into one of 1,000 buckets, numbered 0 to 999. The first buckets go to
validation, and the rest go to training:

| share | validation | training |
|---|---|---|
| 2%, the default | buckets 0 to 19 | buckets 20 to 999 |
| 10% | buckets 0 to 99 | buckets 100 to 999 |

So a group in validation at 2% is in validation at 10% too. Groups differ in
size, so the share of pairs in validation is close to the validation share,
but not exactly that share.

A build from the same sources and the same benchmarks gives the same split.
When a source or a benchmark changes, the groups can change, and so can the
split.
