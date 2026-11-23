# Corpus

`corpus build` combines the [pairs](pairs.md) of the [sources](sources.md)
into one training corpus. It works in three steps, in this order:

1. combine the sources, and keep only one copy of a pair that two sources share
2. drop the pairs that would [leak](leakage.md) a [benchmark](benchmarks.md)
3. split the kept pairs into training and validation pairs

**Build every benchmark that you will score on first,** as
[The workflow](../README.md#the-workflow) says. `corpus build` checks the
pairs against the benchmarks built so far, and stops when none is built.

## Running `corpus build`

```
python -m embedding_model corpus build --name all-sources              # every source
python -m embedding_model corpus build --name example --source cdisc   # some of them
```

- **`--name`** names the corpus.
- **`--source`** names the sources by the ids that [sources.md](sources.md)
  lists. It refuses any other id.
- **Without `--source`**, `corpus build` combines every source. Each source
  must have its pairs, or the build stops. To leave a source out, for example
  LOINC without a loinc.org account, give `--source` with the other sources.
- **`--validation-share`** sets the share of groups that validation holds:
  0.02 by default, from 0.001 to 0.999, in steps of 0.001. A group holds one
  [item](pairs.md#the-shape-a-star-around-each-item) and the items linked to
  it. Two items link when they share an NCIt concept code or a canonical
  text. Links chain: if A links to B, and B links to C, then A, B and C are
  one group. A small corpus can use a larger share, so that validation holds
  enough pairs.

`corpus build` prints how many pairs each of the three steps above keeps and
drops.

## What `corpus build` writes

`corpus build` writes to `embedding_model/inputs/corpora/<corpus>/`.

```
corpora/<corpus>/
    pairs.parquet      the kept pairs, as training and validation pairs
    dropped.parquet    the dropped pairs, each with its finding
    manifest.json      the corpus's manifest
```

### `pairs.parquet`

The kept pairs, with the columns that [pairs.md](pairs.md#pairsparquet)
lists, and `split`: `train` or `validation`. A pair that several sources hold
is kept once, and its `study` holds the studies of every copy.

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
  - `fields`: the two matches, for example `anchor name = source name,
    positive name = target name`.
    [How two texts match](leakage.md#how-two-texts-match) says how to read
    `fields`.
- for the [study rule](leakage.md#the-study-rule-benchmark_study):
  `benchmark_study_id`, the benchmark's study that the pair holds

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
  - `min_overlap_words`: the fewest words in a row that two texts must share
    for a `~` [match](leakage.md#how-two-texts-match)
  - `min_name_words`: the fewest different words that a benchmark name must
    have for a `>=` match
  - `validation_share`: the share of groups that validation holds
  - `buckets`: 1,000. The split hashes each group into one of 1,000 buckets,
    so the validation share goes in steps of 0.001.
- `counts`:
  - `rows`:
    - `combined`: the pairs of all the sources
    - `duplicates_removed`: the copies that step 1 removes
    - `leakage_removed`: the pairs that step 2 drops
    - `kept`, `train` and `validation`: the pairs kept, and each side of the
      split
  - `duplicates_by_source`: the copies removed, by the source that held them
  - `leakage_findings`: the findings, counted as `by_finding`,
    `by_benchmark`, `by_source` (the source of training data) and
    `by_fields`. A pair that leaks two benchmarks counts twice.
  - `shared_vocabulary`: for each benchmark, its
    [shared vocabulary](leakage.md#shared-vocabulary):
    - `rows_with_shared_source`: how many of the benchmark's rows have a
      source variable text that a kept pair also holds
    - `rows_with_shared_target`: how many of the benchmark's rows have a
      target text that a kept pair also holds
    - `most_common`: the 20 most common shared texts, each with the number of
      benchmark rows that hold it
