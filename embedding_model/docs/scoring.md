# Scoring

`python -m embedding_model score` measures how well a model ranks each source
variable's expected targets on a [benchmark](benchmarks.md). Build the
benchmarks first, with
[`benchmarks build`](benchmarks.md#running-benchmarks-build).

## Running `score`

```
python -m embedding_model score --model uc-ctds/bge-large-en-v1.5-bio-mapping --benchmark topmed --format name_description
```

| flag | what it sets |
|---|---|
| `--model` | a Hugging Face model id, a model folder on disk, or the folder of a [run](training.md) that `train` wrote. A run's `model/` folder counts as its run. |
| `--benchmark` | one or more benchmarks. `--benchmark topmed gdc` loads the model once and scores it on both. |
| `--format` | the [text format](#text-formats) of the source and target texts. By default, the format of the run's training pairs, and `compact` for any other model. |
| `--batch-size` | how many texts the model encodes at a time, 32 by default |
| `--device` | `--device cpu` makes `score` use the CPU. By default, `score` uses a GPU when there is one, and the CPU otherwise. |

You can score a released model in the format of its own training samples:
`name_description` for
[`uc-ctds/bge-large-en-v1.5-bio-mapping`](https://huggingface.co/uc-ctds/bge-large-en-v1.5-bio-mapping),
and `stdelement` for `uc-ctds/StdElement-Mapper`. `uc-ctds/StdElement-Mapper`
is private: to score it, you need a Hugging Face login with access to it. A
large model, for example `uc-ctds/StdElement-Mapper`, is much slower on a CPU.

**Build the benchmarks before the corpus**, as
[The workflow](../README.md#the-workflow) says. `score` refuses a run if
`corpus build` did not check the run's corpus for [leakage](leakage.md)
against each benchmark that `--benchmark` names. Each benchmark must have
the same files as when `corpus build` checked it.

For a model that is not a run, `score` prints a note that it cannot check the
model's training data.

A benchmark's build can have another version than its committed manifest,
`embedding_model/benchmarks/manifests/<benchmark>.json`. Then `score` still
scores the model on it, and prints a warning.

## What `score` writes

`score` writes a run's scores to the run's own folder, as
`<run>/scores/<benchmark>/<format>/`. It writes any other model's scores to
`embedding_model/outputs/scores/<model>/<benchmark>/<format>/`.

```
scores/<model>/<benchmark>/<format>/
    metrics.csv         one row: the model's metrics
    per_variable.csv    one row per source variable, with its rank
    config.yaml         how score computed the scores
```

A run's folder holds the same files, in `scores/<benchmark>/<format>/`.

`<model>` is:

- for a Hugging Face model, its owner and name:
  `scores/uc-ctds/StdElement-Mapper/topmed/stdelement/`
- for a model saved on disk, the name of its folder: `--model
  path/to/abcd-model` writes to `scores/abcd-model/topmed/compact/`

Two model folders with the same name share a scores folder. If you score the
second, its scores replace the first one's, and `score` prints a warning.

### `metrics.csv`

One row, with these columns:

- `model`, `benchmark`, `benchmark_version` and `format`
- `n`: the number of source variables scored
- `targets`: the number of targets ranked
- `parameters`: the model's number of parameters
- [the metrics](#the-metrics): `mrr`, `mrr@10`, `ndcg@10`, `accuracy@1`,
  `accuracy@3`, `accuracy@5`, `accuracy@10` and `median_rank`

### `per_variable.csv`

One row per source variable, with these columns:

- `source_study_id`, `source_table_id`, `source_variable_id` and
  `source_variable_name`
- `expected_target`
- `rank`, as [How a model is scored](#how-a-model-is-scored) defines it
- `top1_target`: the target that the model ranked first
- `top1_similarity`: the cosine similarity between the source variable and
  `top1_target`

### `config.yaml`

`config.yaml` records how `score` computed the scores:

- `model`, `benchmark`, `format`, `device` and `batch_size`: the settings
- `model_revision`: the model's Hugging Face revision, for a model given by
  its id
- `benchmark_manifest`: the benchmark's version, build date and file
  checksums, from its manifest
- `ai_harmonization`: the commit of the code, and whether the code had
  uncommitted changes
- `libraries`: the versions of Python and of the libraries
- `scored`: the date and time of the scoring

## Details

### Text formats

| `--format` | the text |
|---|---|
| `compact` (default) | `node.name (type): description Values: a \| b`, with the description and the values only when there are any. This pipeline trains its models in this format. |
| `name_description` | `node.name: description`, the format of the training samples of `uc-ctds/bge-large-en-v1.5-bio-mapping` |
| `name_type_description` | `node.name (type): description`, the format that the harmonization approaches of `ai_harmonization` write by default |
| `name_type_description_values` | `node.name (type): description Values: a \| b` |
| `name_type_values` | `node.name (type): Values: a \| b` |
| `stdelement` | `name: ... \| description: ... \| values: a; b`, the format of the [NCIt](sources.md#ncit) training pairs of `uc-ctds/StdElement-Mapper` |

A source variable's text has no node, in every format. A `stdelement` text has
no node for a target either. Every format that shows values shows the first
12 values.

### How a model is scored

For each source variable, the model ranks every target of the benchmark by
cosine similarity. The variable's rank is the place of its best-placed
expected target: rank 1 if it comes first.

### The metrics

- **MRR**, mean reciprocal rank: the average of 1 / rank over the source
  variables
- **MRR@10**: MRR with every rank past 10 counted as 0. Training reports it
  on the validation pairs.
- **nDCG@10**, the normalized discounted cumulative gain at 10: it counts
  every expected target among the first 10, not only the best placed one
- **accuracy@k**, for k of 1, 3, 5 and 10: the share of source variables with
  an expected target among the first k
- **median rank**: the median of the source variables' ranks
