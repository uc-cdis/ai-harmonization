# Corpus

`corpus build` combines the sources' [pairs](pairs.md) into one training
corpus:

```
python -m embedding_model corpus build --name all-sources              # every source
python -m embedding_model corpus build --name example --source nhanes  # some of them
```

It works in three steps, in this order:

1. combine the sources, and keep a pair that two sources share once;
2. drop the pairs that would leak a benchmark;
3. split the kept pairs into training and validation pairs.

The split sees only the pairs that leakage prevention keeps. The later steps
train the tokenizer on the training pairs alone.

`corpus build` prints how many pairs each step keeps and drops, and the
corpus's manifest records the counts. They depend on the sources and the
benchmarks of the build.

The build stops, and writes no corpus, in three cases:

- no benchmark is built, so the pairs cannot be checked for leakage;
- the sources named hold no pairs;
- leakage prevention drops every pair.

## Combining the sources

The sources are read in the order that the
[table in pairs.md](pairs.md#each-source) lists. Each source's pairs must have
the checksum that their manifest records. When the source's fetch is in the
inputs folder, the pairs must come from that fetch. A pair that copies an
earlier source's pair is kept once, in the earlier source. Two pairs are
copies when the tokenizer normalises their texts to the same two texts.
[pairs.md](pairs.md#duplicates) shows examples. The pair kept takes the
studies of all its copies, so the
[study rule](#the-study-rule-benchmark_study) still sees them. NCIt comes
first, since CDISC publishes its terms as NCIt subsets.

## Leakage prevention

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

### The answer key rule: `answer_key`

A pair is an answer key when its two sides match one benchmark row:

- one side's name or description matches the source variable's name or
  description;
- the other side's name or description matches one of that variable's
  expected targets, by the target's property name or description.

Either side can match the source. For example, a benchmark maps the variable
`body_weight` to `vitals.weight`. The pair `Weight` -> `Body Weight` is then
an answer key, and it is dropped.

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

The benchmark texts are cleaned first with `pairs.common.clean`, the
cleaning that `pairs prepare` gives every text: markup is removed, and
characters and spaces are normalised. Two texts match when they are equal after three changes:

- the tokenizer's normalisation, which removes invisible characters,
  lowercases and removes accents;
- case folding, which also makes `ß` and `ss` equal;
- punctuation turned into spaces.

So two texts always match when the tokenizer normalises them to one text.

A side text also matches a benchmark description when one contains the other
as whole words, and the shorter has at least 10 words. For example, a
benchmark describes a target with a terminology's definition and one
sentence of its own. A pair that holds the same definition then matches that
description. Every side text is compared, since some pairs keep a description
in a name field. Values are never compared: a common list, such as
`Unknown | Not Reported | No | Yes`, matches many targets.

### The study rule: `benchmark_study`

A pair gets the finding `benchmark_study` when its `study` holds the source
study of a benchmark row. For example, a benchmark row comes from the study
`phs999901`. Every pair made from a dbGaP variable of `phs999901` is then
dropped. The study rule catches what text matching cannot see: the other
variables of a benchmark's studies.

A pair that several sources share is kept once, and it takes the studies of
all its copies. So the study rule also sees a copy that came from a
benchmark's study. [Combining the sources](#combining-the-sources) describes
the copies.

### Shared vocabulary

A benchmark text can also occur in a pair without a finding. For example,
`age` is one side of a pair whose other side matches no expected target of
that benchmark row. That is not leakage: the pair is kept, and the manifest
counts the text as shared vocabulary.

## The split

Validation holds about 2% of the groups. A group is an item with all its
pairs. NCIt and CDISC share NCIt's concept codes, so a concept's pairs from
both sources form one group. All pairs of a group go to the same side. A pair
that two sources share is kept once, in the earlier source's item, so it goes
to that item's side.

A hash of the group decides its side, with no random draw and no seed. The
hash puts each group into one of 1,000 buckets, and the groups in buckets 0 to
19 go to validation. So the split is the same on every build, and a group keeps
its side as sources are added. Groups differ in size, so the share of pairs in
validation is close to 2%, but not exactly 2%.

The split is not stratified. Each source has many groups, so its share of
validation pairs is close to 2%.

## What a corpus holds

`corpus build` writes to `embedding_model/inputs/corpora/<name>/`, or under the
folder that `--inputs` names. It writes the folder whole, so a failed build
leaves the previous corpus as it was; see
[Inputs and outputs](../README.md#inputs-and-outputs).

- **`pairs.parquet`:** the kept pairs, with the columns that
  [pairs.md](pairs.md#what-a-pair-holds) lists, and `split`: `train` or
  `validation`.
- **`dropped.parquet`:** every dropped pair, with its columns and its finding.
  A pair that leaks two benchmarks has a row for each:
  - `finding`: `answer_key` or `benchmark_study`, and `benchmark`;
  - for an answer key:
    - `benchmark_row`: the row's position in `<benchmark>_mappings.csv`,
      counted from 0;
    - `benchmark_source_variable_name` and
      `benchmark_source_variable_description`: the source variable, as the
      benchmark gives it;
    - `benchmark_target_property` and `benchmark_target_description`: the
      expected target that matched;
    - `fields`: the two fields that matched, such as `anchor name = source
      name, positive name = target name`. A `~` marks a near-duplicate;
  - for the study rule: `benchmark_study_id`, the benchmark's study that the
    pair holds.
- **`manifest.json`:** the corpus's manifest, with the fields that
  [manifests.md](manifests.md#its-fields) lists:
  - `files`: `pairs.parquet` and `dropped.parquet`, each with its row count,
    its file's `sha256` and its table's `csv_sha256`;
  - `sources`: each source's pairs, with the source's version and the pairs'
    checksum, and each benchmark, with its version and its files' checksums;
  - `settings`: the near-duplicate words, the validation share and the number
    of buckets;
  - `counts`:
    - `rows`:
      - `combined`: the pairs of all the sources;
      - `duplicates_removed`: the copies of an earlier source's pairs;
      - `leakage_removed`: the pairs that leakage prevention drops;
      - `kept`, split into `train` and `validation`;
    - `duplicates_by_source`;
    - `leakage_findings`, counted by finding, benchmark, source and fields. A
      pair that leaks two benchmarks counts twice;
    - `shared_vocabulary`, per benchmark.
