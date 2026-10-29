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
[study rule](leakage.md#the-study-rule-benchmark_study) still sees them. NCIt comes
first, since CDISC publishes its terms as NCIt subsets.

## Leakage prevention

Leakage prevention drops each pair that would leak a benchmark. Such a pair
holds one of a benchmark's mappings, or comes from one of its studies. Leakage
prevention compares the pairs with every benchmark built under
`embedding_model/inputs/benchmarks/`. [leakage.md](leakage.md) gives its rules,
with examples.

## The split

Validation holds 2% of the groups by default. `--validation-share` sets
another share, in steps of 0.001, such as `--validation-share 0.1`. A small
corpus can use a larger share, so that validation holds enough pairs.

A group is an item with all its pairs. NCIt and CDISC share NCIt's concept
codes, so a concept's pairs from both sources form one group. All pairs of a
group go to the same side. A pair
that two sources share is kept once, in the earlier source's item, so it goes
to that item's side.

A hash of the group decides its side, with no random draw and no seed. The
hash puts each group into one of 1,000 buckets. The groups in the first
buckets go to validation: buckets 0 to 19 for a share of 2%. So the split is
the same on every build, and a group keeps its side as sources are added. A
larger share adds buckets, so a group in validation at 2% is in validation at
10% too. Groups differ in size, so the share of pairs in validation is close to
the share asked for, but not exactly that share.

The split is not stratified. Each source has many groups, so its share of
validation pairs is close to the share asked for.

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
      name, positive name = target name`. [How two texts match](leakage.md#how-two-texts-match)
      says how to read it;
  - for the study rule: `benchmark_study_id`, the benchmark's study that the
    pair holds.
- **`manifest.json`:** the corpus's manifest, with the fields that
  [manifests.md](manifests.md#its-fields) lists:
  - `files`: `pairs.parquet` and `dropped.parquet`, each with its row count,
    its file's `sha256` and its table's `csv_sha256`;
  - `sources`: each source's pairs, with the source's version and the pairs'
    checksum, and each benchmark, with its version and its files' checksums;
  - `settings`:
    - `min_overlap_words`: the minimum number of words in a row that two
      texts share to match;
    - `min_name_words`: the minimum number of words of a name that can match
      by its words;
    - `validation_share` and `buckets`: the split's share and its hash
      buckets;
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
