# Scoring

`python -m embedding_model score` measures how well a model ranks each source
variable's expected targets on a benchmark. A benchmark lists source
variables, each with its expected targets: the target properties, written
`node.property`, that experts mapped it to.
[`<benchmark>_mappings.csv`](benchmarks.md#benchmark_mappingscsv) lists the
source variables and their expected targets.
Build the benchmarks first, with
[`benchmarks build`](benchmarks.md#running-benchmarks-build), as
[The workflow](../README.md#the-workflow) shows.

`score` writes each benchmark's scores to a separate folder.
[What `score` writes](#what-score-writes) lists the files.
[Details](#details) gives the text formats, how a model is scored, and the
metrics.

[design_decisions.md](design_decisions.md#scoring) gives the reasons for the
choices below.

## Running `score`

```
python -m embedding_model score --model uc-ctds/bge-large-en-v1.5-bio-mapping --benchmark topmed --format name_description
```

`--model` is a Hugging Face model id, a model folder on disk, or the folder
of a run that `train` wrote.

`--format` sets the format of the source and target texts.
[Text formats](#text-formats) lists the formats.
[`compare`](comparing.md) compares models in one format, `compact` by
default. This pipeline trains its models in `compact`. You can also score a
released model in the format of its own training samples:
`name_description` for
[`uc-ctds/bge-large-en-v1.5-bio-mapping`](https://huggingface.co/uc-ctds/bge-large-en-v1.5-bio-mapping),
and `stdelement` for `uc-ctds/StdElement-Mapper`. `score` scores a run in its
training format, unless `--format` names another format.

`uc-ctds/StdElement-Mapper` is private. To score it, you need a Hugging Face
login with access to it.

`corpus build` checks the pairs of a [corpus](corpus.md) against the
benchmarks built so far, for [leakage](leakage.md). If the check of a run's
corpus missed a benchmark, or covered another build of it, `score` refuses
the run before it scores any benchmark.
[Scoring a run](training.md#scoring-a-run) says how `score` matches a
benchmark with the check: by its name and its files' checksums. For any
other model, `score` cannot check its training data, and it prints a note
that says so.

`--benchmark` takes one or more benchmarks. `--benchmark topmed gdc` loads the
model once and scores it on both. `--batch-size` sets how many texts the model
encodes at a time. The default is 32.

`score` checks each benchmark file against its
[SHA-256](https://doi.org/10.6028/NIST.FIPS.180-4) in the build's
`manifest.json`. If a file differs, `score` stops, and the error gives the
command that rebuilds the benchmark. If a build's version is not the one in
the committed manifest, in `embedding_model/benchmarks/manifests/`, `score`
still scores the build and prints a warning.
[manifests.md](manifests.md#the-committed-manifest-and-the-builds-own) tells
the two manifests apart.

`score` uses a GPU when there is one, and the CPU otherwise. A large model,
for example `uc-ctds/StdElement-Mapper`, is much slower on a CPU.
`--device cpu` makes `score` use the CPU.

## What `score` writes

`score` writes a run's scores to the run's own folder, as
`<run>/scores/<benchmark>/<format>/`. It writes any other model's scores to
`embedding_model/outputs/scores/<model>/<benchmark>/<format>/`, or under the
folder that `--outputs` names. It writes each folder whole. So if scoring a
benchmark fails, that benchmark's previous scores stay as they were.
[Inputs and outputs](../README.md#inputs-and-outputs) describes how a step
writes a folder whole.

```
scores/<model>/<benchmark>/<format>/
    metrics.csv         one row: the model's metrics
    per_variable.csv    one row per source variable, with its rank
    config.yaml         how score computed the scores
```

`<format>` is the [text format](#text-formats) of the scores. So a model that
you score in two formats keeps both scores. `<model>` is:

- for a Hugging Face model, its owner and name:
  `scores/uc-ctds/StdElement-Mapper/topmed/stdelement/`
- for a model saved on disk, the name of its folder: `--model
  path/to/abcd-model` writes to `scores/abcd-model/topmed/compact/`

Two model folders with the same name share a scores folder. If you score the
second, its scores replace the first one's, and `score` prints a warning.
`metrics.csv` and `config.yaml` record a model on disk by its full path. So
they show which model the folder holds.

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

`config.yaml` records how `score` computed the scores. Scores have no
manifest, and `config.yaml` takes its place.

- `model`, `benchmark`, `format`, `device` and `batch_size`: the settings
- `model_revision`: the model's Hugging Face revision, for a model given by
  its id
- `benchmark_manifest`: the benchmark's version, build date and file
  checksums, from its manifest
- `ai_harmonization`:
  - `commit`: the ai-harmonization commit of the code that computed the
    scores
  - `uncommitted_changes`: true when a file of the code was new or changed
    since that commit. The code is the Python files of `ai_harmonization/`
    and `embedding_model/`, with `pyproject.toml` and `uv.lock`. An
    uncommitted edit to any other file, for example a doc, leaves it false.
- `libraries`: the versions of Python and of the libraries, for example
  [sentence-transformers](https://sbert.net)
- `scored`: the date and time of the scoring

## Details

### Text formats

`--format` takes one of six formats. Four are the formatters of
`ai_harmonization.formatters`, under their own names. The harmonization
approaches of `ai_harmonization` suggest targets for each source variable,
and they write texts with these formatters.
`embedding_model/texts.py` defines the other two, `compact` and `stdelement`.

| `--format` | the text |
|---|---|
| `compact` (default) | `node.name (type): description Values: a \| b`, with the description and the values only when there are any. This pipeline trains its models in this format. |
| `name_description` | `node.name: description`, the format of the training samples of `uc-ctds/bge-large-en-v1.5-bio-mapping` |
| `name_type_description` | `node.name (type): description`, the format that the harmonization approaches build, unless you give them another formatter |
| `name_type_description_values` | `node.name (type): description Values: a \| b` |
| `name_type_values` | `node.name (type): Values: a \| b` |
| `stdelement` | `name: ... \| description: ... \| values: a; b`, the format of the [NCIt](sources.md#ncit) training pairs of `uc-ctds/StdElement-Mapper` |

`compact` writes the description, and the `": "` before it, only when a
property has one. The formatters treat values the same way. A target
`visit.visit_id` with no description and no values reads
`visit.visit_id (string)`. A source with values but no description reads
`smoking_status (string) Values: Current | Former | Never`. The training
pairs use `compact` too.

In every format but `stdelement`, a target's text includes its node, as in
`vitals.body_weight (float): Body weight in kilograms`. A `stdelement` text
has no node, as in the training pairs of StdElement-Mapper. So two targets on
different nodes have the same `stdelement` text when they have the same
name, description and values. Each copy of a wrong target then takes its own
place in the ranking. Copies of one text tie, and
[sentence-transformers](https://sbert.net) 6 orders them by their key,
`node.property`.

A source variable's text has no node, in every format. Its node would be its
[dbGaP](https://www.ncbi.nlm.nih.gov/gap/) table, for example
`pht999999.v1`, or the paper that it comes from. The text starts at the
variable's name: `body_weight (float): Body weight in kilograms`. The
harmonization approaches do write a source's node, as in
`pht999999.v1.body_weight (float): ...`.

Every format that shows values keeps the first 12 values. The formats of the
two released models copy the training samples that their model cards show.

### How a model is scored

For each source variable, the model ranks every target of the benchmark by
cosine similarity, as the harmonization approaches do. The variable's rank is
the place of its expected target: rank 1 if it comes first.

For example, for a source variable `age`, the model ranks every target from
most to least similar. If `participant.age_at_visit` comes third, that
variable's rank is 3. When a variable has more than one expected target, its
rank is the place of the best placed one.

### The metrics

MRR, accuracy@k and the median rank come from the ranks. nDCG@10 uses the
place of every expected target.

- **MRR**, mean reciprocal rank: the average of 1 / rank over the source
  variables. An expected target at the top counts fully, second counts half,
  and so on. MRR has no cut-off: it counts every rank.
- **MRR@10**: MRR with every rank past 10 counted as 0. Training reports it
  on the validation pairs.
- **nDCG@10**, the normalized discounted cumulative gain at 10: each expected
  target among the first 10 adds 1 / log2(p + 1), where p is its place. That
  is 1 at place 1, 0.63 at place 2 and 0.5 at place 3. nDCG@10 divides the
  sum of these values by the sum for a perfect ranking, which puts every
  expected target first. So a variable whose expected targets all come first
  scores 1. Unlike MRR and accuracy@k, nDCG@10 counts every expected target,
  not only the best placed. BEIR and MTEB report the same metric.
- **accuracy@k**, for k of 1, 3, 5 and 10: the share of source variables with
  an expected target among the first k. It is what a data curator who
  reviews the top k suggestions sees.
- **median rank**: the median of the source variables' ranks. At least half
  the variables have a rank at the median or better.

For example, a variable has two expected targets, at places 1 and 3. A
perfect ranking would put them at places 1 and 2. So the variable's nDCG@10
is (1 + 0.5) / (1 + 0.63) = 0.92, while its MRR is 1.

The `InformationRetrievalEvaluator` of
[sentence-transformers](https://sbert.net), which Hugging Face maintains, ranks
the targets and computes MRR, MRR@10, nDCG@10 and accuracy@k. `score` gives
the evaluator the model's default prompt for both the source texts and the
target texts. It is the prompt that `model.encode()` adds. So the evaluator
adds no `query` or `document` prompt.

`per_variable.csv` keeps the rank of every source variable.
