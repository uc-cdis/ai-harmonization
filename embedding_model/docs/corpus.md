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
train the tokenizer on the training pairs alone, so a dropped text never
shapes the vocabulary.

`corpus build` prints how many pairs each step keeps and drops, and the
corpus's manifest records the counts. They depend on the sources and the
benchmarks of the build.

## Combining the sources

The sources are read in the order that the
[table in pairs.md](pairs.md#each-source) lists. Each source's pairs must have
the checksum that their manifest records. When the source's fetch is in the
inputs folder, the pairs must come from that fetch. A pair that copies an
earlier source's pair is kept once, in the earlier source. Two pairs are
copies when the tokenizer normalises their texts to the same two texts.
[pairs.md](pairs.md#duplicates) shows examples. The pair kept takes the
studies of all its copies, so the study rule still sees them. NCIt comes
first, since CDISC publishes its terms as NCIt subsets.

## Leakage prevention

Leakage prevention compares the pairs with every benchmark built under
`embedding_model/inputs/benchmarks/`. Without a built benchmark, the corpus
is not built.

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

The benchmark texts are cleaned first, as `pairs prepare` cleans the pairs'
texts. Two texts match when they are equal after three changes:

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

A finding is a reason to drop a pair, found against one benchmark. There are
two kinds:

| finding | when | example |
|---|---|---|
| `answer_key` | one side's name or description matches the source variable's name or description, and the other side's name or description matches one of that variable's expected targets, by its property name or description. Either side can match the source | a benchmark maps the variable `body_weight` to `vitals.weight`. The pair `Weight` -> `Body Weight` is then an answer key, and it is dropped |
| `benchmark_study` | the study rule: the pair's `study` holds the source study of a benchmark row | a benchmark row comes from the study `phs999901`. Every pair made from a dbGaP variable of `phs999901` is then dropped |

The study rule catches what text matching cannot see: the other variables of
a benchmark's studies. Each benchmark is checked on its own, with the same
rules, and a pair is dropped if any benchmark finds it. Against one benchmark,
a pair has at most one finding: an answer key comes before a benchmark study.
A pair that leaks two benchmarks has a finding for each.

**Shared vocabulary is kept.** A benchmark text can also occur in a pair
without a finding. For example, `age` is one side of a pair whose other side
matches no expected target of that benchmark row. That is not leakage: the
pair is kept, and the manifest counts the text as shared vocabulary.

## The split

Validation holds 2% of the groups. A group is an item with all its pairs.
NCIt and CDISC share NCIt's concept codes, so a concept's pairs from both
sources form one group. All pairs of a group go to the same side, so no
validation pair has a text that training saw from the same item.

A hash of the group decides its side, with no random draw and no seed. The
hash puts each group into one of 1,000 buckets, and the groups in buckets 0 to
19 go to validation. So the split is the same on every build, and a group keeps
its side as sources are added. Groups differ in size, so the share of pairs in
validation is close to 2%, but not exactly 2%.

The split is not stratified. Each source has many groups, so its share of
validation pairs is close to 2%.

## What a corpus holds

`corpus build` writes to `embedding_model/inputs/corpora/<name>/`, or under the
folder that `--inputs` names. It writes to a staging folder first, so a failed
build leaves the last corpus as it was.

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
  - `files`: `pairs.parquet` and `dropped.parquet`, each with its row count
    and SHA-256;
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
