# Scoring

`python -m embedding_model score` measures how well a model ranks each source
variable's expected targets on a benchmark. The [benchmarks](benchmarks.md) are
built first; see [The workflow](../README.md#the-workflow). It writes each
benchmark's scores to a folder of their own.
[What `score` writes](#what-score-writes) lists them.
[Details](#details) gives the text formats, how a model is scored, and the
metrics.

[design_decisions.md](design_decisions.md#scoring) gives the reasons for the
choices below.

## Running `score`

```
python -m embedding_model score --model uc-ctds/bge-large-en-v1.5-bio-mapping --benchmark topmed --format name_description
```

`--model` is a Hugging Face model id, a model folder on disk, or the folder
of a run that `train` wrote. A run is scored in its training format, unless
`--format` names another. It is scored only on benchmarks that its corpus
was checked against for leakage; see
[Scoring a run](training.md#scoring-a-run). For any other model, scoring notes
that its leakage is unknown. `--format` is the shape that the source and
target texts are written in.
[Text formats](#text-formats) lists them. Models are
compared in the default shape, the one that this pipeline's models are trained
in. A released model can also be scored in
the shape of its own training samples: `name_description` for
[`uc-ctds/bge-large-en-v1.5-bio-mapping`](https://huggingface.co/uc-ctds/bge-large-en-v1.5-bio-mapping), and `stdelement` for
`uc-ctds/StdElement-Mapper`. `uc-ctds/StdElement-Mapper` is private: scoring
it needs a Hugging Face login with access to it.

`--benchmark` takes one or more benchmarks. `--benchmark topmed gdc` loads the
model once and scores it on both. `--batch-size` sets how many texts are
encoded at a time, 32 by default.

`score` checks each benchmark file against the [SHA-256](https://doi.org/10.6028/NIST.FIPS.180-4) that its manifest
holds. A file that differs stops scoring, and the error gives the command
that rebuilds it. A build whose version is not the version committed under
`embedding_model/benchmarks/manifests/` is scored, with a warning.

Scoring uses a GPU when there is one, and the CPU otherwise. A large model,
such as `uc-ctds/StdElement-Mapper`, is much slower on a CPU. `--device cpu`
keeps scoring on the CPU.

## What `score` writes

`score` writes a run's scores to the run's own folder, as
`<run>/scores/<benchmark>/<format>/`. It writes any other model's scores to
`embedding_model/outputs/scores/<model>/<benchmark>/<format>/`, or under the
folder that `--outputs` names. It writes the folder whole, so a scoring that
fails leaves the previous scores as they were; see
[Inputs and outputs](../README.md#inputs-and-outputs).

```
scores/<model>/<benchmark>/<format>/
    metrics.csv         one row: the model's metrics
    per_variable.csv    one row per source variable, with its rank
    config.yaml         what the scores were computed with
```

`<format>` is the [text format](#text-formats) used, so a model scored in
two formats keeps both scores. `<model>` is:

- for a Hugging Face model, its owner and name:
  `scores/uc-ctds/StdElement-Mapper/topmed/stdelement/`;
- for a model saved on disk, the name of its folder: `--model
  path/to/abcd-model` writes to `scores/abcd-model/topmed/compact/`.

Two model folders with the same name share a scores folder. Scoring the
second replaces the first one's scores, with a warning. `metrics.csv` and
`config.yaml` record a model on disk by its full path, so they show which
model the folder holds.

### `metrics.csv`

One row, with these columns:

- `model`, `benchmark`, `benchmark_version` and `format`;
- `n`, the source variables scored; `targets`, the targets ranked; and
  `parameters`, the model's number of parameters;
- [the metrics](#the-metrics): `mrr`, `mrr@10`, `ndcg@10`, `accuracy@1`,
  `accuracy@3`, `accuracy@5`, `accuracy@10` and `median_rank`.

### `per_variable.csv`

One row per source variable: `source_study_id`, `source_table_id`,
`source_variable_id`, `source_variable_name`, `expected_target` and `rank`,
as [How a model is scored](#how-a-model-is-scored) defines it. Then the target
that the model ranked first, `top1_target`, with its cosine similarity,
`top1_similarity`.

### `config.yaml`

What the scores were computed with. Scores have no manifest:
`config.yaml` takes its place.

- `model`, `benchmark`, `format`, `device` and `batch_size`: the settings;
- `model_revision`: the model's Hugging Face revision, for a model loaded by
  its id;
- `benchmark_manifest`: the benchmark's version, build date and file
  checksums, from its manifest;
- `ai_harmonization`:
  - `commit`: the ai-harmonization commit of the code that computed them;
  - `uncommitted_changes`: true when a file of the code was new or changed
    since that commit. The code is the Python files of `ai_harmonization/`
    and `embedding_model/`, with `pyproject.toml` and `uv.lock`. An
    uncommitted edit to any other file, such as a doc, leaves it false;
- `libraries`: the versions of Python and of the libraries, such as
  [sentence-transformers](https://sbert.net);
- `scored`: the date and time of the scoring.

## Details

### Text formats

`--format` takes one of six formats. Four are the formatters of
`ai_harmonization.formatters`, under their own names. Two, `compact` and
`stdelement`, are defined in `embedding_model/texts.py`:

| `--format` | the text |
|---|---|
| `compact` (default) | `node.name (type): description Values: a \| b`, each part only when present: the shape that this pipeline's models are trained on |
| `name_description` | `node.name: description`, the shape `uc-ctds/bge-large-en-v1.5-bio-mapping` was trained on |
| `name_type_description` | `node.name (type): description`, what the harmonization approaches build unless given another formatter |
| `name_type_description_values` | `node.name (type): description Values: a \| b` |
| `name_type_values` | `node.name (type): Values: a \| b` |
| `stdelement` | `name: ... \| description: ... \| values: a; b`, the shape of `uc-ctds/StdElement-Mapper`'s [NCIt](sources.md#ncit) training pairs |

`compact` writes the description, and the `": "` before it, only when a
property has one. The formatters treat values the same way. A target
`visit.visit_id` with no description and no values reads
`visit.visit_id (string)`. A source with values but no description reads
`smoking_status (string) Values: Current | Former | Never`. The training
pairs are written in `compact` too.

In every format but `stdelement`, a target's text includes its node, as in
`vitals.body_weight (float): Body weight in kilograms`. `stdelement` leaves the
node out, as StdElement-Mapper's training pairs do. So a property that several
nodes carry alike has the same text in each. Each copy of a wrong target then
takes a rank of its own. Identical texts tie. [sentence-transformers](https://sbert.net) 6
breaks a tie by target key. Earlier releases break it in no fixed order.

A source variable's text leaves its node out, in every format. Its node would
be its [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) table, such as `pht999999.v1`, or the paper that it comes from.
The text starts at the variable's name: `body_weight (float): Body weight in
kilograms`. The harmonization approaches do write a source's node, as in
`pht999999.v1.body_weight (float): ...`.

Every format that shows values keeps the first 12. The two released models'
shapes follow the training samples on their model cards.

### How a model is scored

For each source variable, the model ranks every target of the benchmark, by
cosine similarity, as the harmonization approaches do. The variable's score is
where its expected target lands: rank 1 if it comes first.

For example, for a source variable `age`, the model sorts every target from
most to least similar. If `participant.age_at_visit` comes third, that
variable's rank is 3. Some variables have more than one expected target, and
then the best placed of them counts.

### The metrics

From the rankings:

- **MRR**, mean reciprocal rank: the average of 1 / rank. An expected target
  at the top counts fully, second counts half, and so on. MRR is one number with
  no cut-off.
- **MRR@10**: MRR with every rank past 10 counted as 0. Training reports it
  on the validation pairs.
- **nDCG@10**, the normalized discounted cumulative gain at 10. Each expected
  target among the first 10 counts 1 / log2(rank + 1): 1 at rank 1, 0.63 at
  rank 2, 0.5 at rank 3. The sum is divided by the sum of a perfect ranking,
  which puts every expected target first. So a variable whose expected
  targets all come first scores 1. Unlike MRR and accuracy@k, it counts every
  expected target, not only the best placed. For example, a variable with two
  expected targets, at ranks 1 and 3, scores (1 + 0.5) / (1 + 0.63) = 0.92. A
  perfect ranking would put them at ranks 1 and 2. Its MRR is 1. BEIR and MTEB report nDCG@10, so it compares with their
  scores.
- **accuracy@k**, for k of 1, 3, 5 and 10: how often an expected target is
  among the first k. It is what a data curator who reviews the top k suggestions sees.
- **median rank**: where the expected target typically lands.

The `InformationRetrievalEvaluator` of
[sentence-transformers](https://sbert.net), which Hugging Face maintains, ranks
the targets and computes MRR, MRR@10, nDCG@10 and accuracy@k. It embeds both
sides with the model's default prompt only: a model's `query` and `document`
prompts are not added.

Every source variable's rank is kept, in `per_variable.csv`.
