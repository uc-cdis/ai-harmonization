# Embedding model

Tools to build, train and evaluate a small embedding model that maps study
variables to data-model properties: a benchmark of expert mappings, the
training data, a tokenizer, the model, and the experiments that decide
between versions of it. The model is a static embedding table trained from
scratch, small enough to encode thousands of texts a second on a CPU.
Everything here runs from the command line, from a checkout of this
repository.

### How it fits with the rest of the repository

- Every text a model reads -- a study variable, a data-model property -- is
  written by `ai_harmonization.formatters`, the same functions the
  harmonization approaches use, so a model is trained and evaluated on texts
  laid out the way it will see them in use: the same structure, never the
  same texts. The corpus is audited for leakage, so no benchmark text, and
  no near copy of one, reaches training.
- A trained model is used through `ai_harmonization`'s embedding approach,
  like any other sentence-transformers model, and can be evaluated end to end
  with `get_metrics_for_approach`
  (`ai_harmonization/harmonization_benchmark.py`).
- `jupyter/embedding_model_training.ipynb` fine-tunes a pretrained model;
  this trains a small model from scratch. Both produce sentence-transformers
  models.

## Setup

```
uv sync
uv pip install -e .
```

Everything runs on a CPU, model training included. Training is where a GPU
matters, and it uses one if the machine has one (`--device cpu` keeps it on
the CPU). The [quick start](#quick-start) examples train on a CPU in a few
minutes; a model trained on all the sources takes minutes on a GPU or hours
on a CPU, and a series of experiments trains many models.

A series' runs are trained in one of two ways:

- `experiment submit` queues each run as its own SLURM job (`sbatch`), with
  one GPU each, so they train in parallel. Use it on a cluster, passing your
  cluster's settings, such as partition and time limit, through `--slurm`.
- `experiment submit --local` trains them one after another on the machine
  the command runs on, on its GPU if it has one and otherwise on the CPU: a
  laptop, a workstation, or a cluster node you have been allocated, such as
  inside `srun --gpus=1 --pty bash` -- not a shared login node. Use it for
  small series and quick tests.

## Quick start

All commands here, and in the rest of this guide, run from the repository
root.

`embedding_model/examples/` holds two files to start from:
`independent_run_example.yaml`, the configuration for one training run, and
`small_series_example.yaml`, a small series of experiments. Each stands on
its own.

Both use one benchmark, [TOPMed](BENCHMARKS.md#topmed), and one training
source, [NHANES](SOURCES.md#nhanes) -- about 21,000 pairs of variable names,
descriptions and values from CDC's public codebooks -- with a small vocabulary
and a short run, so they finish in minutes. That is a small example to see
every step work; a model meant for use should be trained on more sources.

**Train and score one model**:

```
python -m embedding_model benchmarks build --topmed
python -m embedding_model sources fetch --source nhanes
python -m embedding_model pairs prepare --source nhanes
python -m embedding_model pairs audit                  # against the benchmarks built: here TOPMed alone
python -m embedding_model corpus build --name example --sources nhanes
python -m embedding_model tokenizer train --corpus example
python -m embedding_model train --config embedding_model/examples/independent_run_example.yaml --seed 0
python -m embedding_model score --run independent_run_example_s0 --benchmark topmed
```

A run is named after its configuration file and seed, so this one is
`outputs/independent_runs/independent_run_example_s0/`.

**Run a small series**: `small_series_example.yaml` declares a base
configuration and two arms, each changing one setting, at five seeds: fifteen
runs. The short example series shows how a series of experiments is run and
read. A series meant to decide something needs more seeds, as
[PROCESS.md](PROCESS.md) explains.

```
# The same data as the example above: skip these if you have run it.
python -m embedding_model benchmarks build --topmed
python -m embedding_model sources fetch --source nhanes
python -m embedding_model pairs prepare --source nhanes
python -m embedding_model pairs audit
python -m embedding_model corpus build --name example --sources nhanes
python -m embedding_model tokenizer train --corpus example

python -m embedding_model cache build --corpus example
python -m embedding_model experiment submit embedding_model/examples/small_series_example.yaml --local
python -m embedding_model experiment status embedding_model/examples/small_series_example.yaml
python -m embedding_model experiment report embedding_model/examples/small_series_example.yaml
```

`--local` trains the fifteen runs on the machine, one after another, useful
for a quick test on a laptop; without it each run is submitted to SLURM as its
own job, and they train at the same time.

## How the model works

A static embedding table: one vector per token of a vocabulary, and a text's
embedding is the average of its tokens' vectors. A table has no attention and
no layers, so it is small (a few million parameters) and fast.

The tokenizer works in three steps, as a Hugging Face
[`tokenizers`](https://huggingface.co/docs/tokenizers) pipeline does. Text is
lowercased and stripped of accents; then split at whitespace and at every
punctuation character, the rule BERT uses (`BertPreTokenizer`), so
`SUBJECT_ID` becomes `subject`, `_`, `id`; then each piece is split into
vocabulary tokens by [byte-pair encoding
(BPE)](https://huggingface.co/learn/llm-course/chapter6/5). BPE learns its
vocabulary from the corpus: it starts from characters and keeps merging the
most frequent adjacent pair, so common words become one token and a rare word
splits into known pieces rather than becoming unknown. A token never spans
two words.

The table is trained from scratch with a contrastive loss (in-batch
negatives) on pairs of texts that mean the same thing -- a variable and its
description, a term and its synonym, two names for one data-model property.
Each pair is pulled together, and pushed apart from every other text in its
batch.

## The workflow

Every step is a subcommand of one command, run from the repository root:
`python -m embedding_model <step>`, and `python -m embedding_model --help`
lists them. Each step reads what the one before it wrote, so a step can be
rerun on its own.

| | step | what it does | writes |
|---|---|---|---|
| benchmarks | `benchmarks build --topmed` | builds [TOPMed](BENCHMARKS.md#topmed), the benchmark models are developed on | `inputs/benchmarks/` |
| | `benchmarks build --gdc` | builds [GDC](BENCHMARKS.md#gdc), held back: scored once per series' pick (the series' winner) | `inputs/benchmarks/` |
| | `score --model <model> --benchmark <benchmark>` | scores any model, trained here or elsewhere | `scores/<benchmark>/` in the run's folder, or `outputs/scores/<model>/` for a model trained elsewhere: the metrics and every mapping's rank |
| training data | `sources fetch [--source <source>]` | downloads the training sources, or one of them | `inputs/sources/` |
| | `pairs prepare [--source <source>]` | turns each source into pairs of texts that mean the same thing | `inputs/pairs/` |
| | `pairs audit` | checks no pair leaks a text of any benchmark built so far; must pass before a corpus is built | `inputs/pairs/audit.json`: what was checked, against which benchmarks, and the verdict |
| | `corpus build --name <corpus> [--sources <source>,...]` | combines the sources' pairs into a training corpus | `inputs/corpora/<corpus>/` |
| | `tokenizer train --corpus <corpus>` | learns the vocabulary from the corpus | `inputs/tokenizers/` |
| | `cache build --corpus <corpus>` *(optional)* | tokenizes the corpus once, before a series | `inputs/token_cache/` |
| one model | `train --config <config.yaml> --seed <seed>` | trains one model on its own | `outputs/independent_runs/` |
| a series | `experiment submit <series.yaml> [--local]` | trains every run the series declares -- each arm at each seed: one SLURM job per run, so many train at once, or with `--local` one after another on the machine the command runs on | `outputs/series/<series>/runs/` |
| | `experiment status <series.yaml>` | says which runs are done, running or missing | -- |
| | `experiment report <series.yaml>` | builds tables and figures, with each comparison's verdict | `outputs/series/<series>/` |
| | `experiment pick <series.yaml>` | writes the series' pick record | `outputs/series/<series>/pick.md` |

`<series.yaml>` is an experiment file: the arms of a series, the seeds, and
the comparisons to make.
[examples/small_series_example.yaml](examples/small_series_example.yaml) is
one.

Training builds the token cache itself on first use if `cache build` has not
run, and `--no-token-cache` tokenizes inside each training step instead: the
same model, more slowly. `cache build` is worth running before a series, so
that many runs starting at once do not each wait for it.

The audit checks the pairs against every benchmark built so far, and each
corpus records which benchmarks its pairs were checked against. Scoring on a
benchmark refuses a model whose corpus was not checked against that benchmark
for leakage.

## Evaluating a model

`score` gives the metrics model development needs. For each source variable,
the model ranks every target of the benchmark by similarity -- all 78 on
TOPMed, all 1,963 on GDC -- and the variable's score is where its expected
target lands: rank 1 if it comes first. For the variable `age`, say, the model
sorts the 1,963 GDC targets from most to least similar; if
`demographic.age_at_index` comes third, that mapping's rank is 3. Some
variables have more than one correct target -- `age` has four -- and then the
best placed of them counts.
From those ranks:

- **MRR**, mean reciprocal rank: the average of 1 / rank, so a right answer
  at the top counts fully, second counts half, and so on. One number, with no
  cut-off, which is what models are compared on.
- **recall@k**, for k of 1, 5 and 10: how often the expected target is among
  the first k -- what someone reviewing the top k suggestions sees.
- **median rank**: where the expected target typically lands.

A rank is kept for every mapping, so two models, or two sets of training
runs, are compared mapping by mapping rather than only on their averages.

There are two benchmarks:

- **[TOPMed](BENCHMARKS.md#topmed)**: dbGaP study variables mapped by experts
  to the harmonized phenotypes of the TOPMed Data Coordinating Center, 1,940
  mappings against 78 targets. Models are developed on it.
- **[GDC](BENCHMARKS.md#gdc)**: the schema-matching pairs published with the
  Magneto paper (Zenodo record 14963588), 165 source variables from ten papers
  against the GDC data dictionary. It is held back: scored once, on a series'
  pick (the series' winner). It is built from the published inputs -- that
  ground truth, the ten papers' source tables and the live GDC submission
  dictionary -- and since the ground truth names a property rather than a
  node, a property that several GDC nodes carry is an accepted answer on each
  of them.

[BENCHMARKS.md](BENCHMARKS.md) describes each one: what it is built from, its
licence, and how to cite it.

`get_metrics_for_approach` evaluates a harmonization approach end to end: the
approach suggests a set of mappings for a source and a target model, scored
by precision, recall and F1. Use it to evaluate a trained model as part of an
approach.

## Inputs and outputs

The steps that prepare data -- benchmarks, sources, pairs, corpora,
tokenizers, the token cache -- write under `embedding_model/inputs/`.
Training runs read what they need from there and write everything they
produce under `embedding_model/outputs/`: the model, its logs, its scores.
Both folders are gitignored. `--inputs DIR` and `--outputs DIR` set a
different location for either one, which is useful for a full corpus's token
cache: it can run to tens of gigabytes.

```
inputs/
    benchmarks/                  <benchmark>_targets.csv and <benchmark>_mappings.csv
    sources/<source>/<date>/     a training source as downloaded, with its manifest
    pairs/                       <source>.parquet: each source's pairs of texts
        audit.json               what the leakage audit checked, and its verdict
    corpora/<corpus>/            training and validation pairs; corpus.json records what went in
    tokenizers/<tokenizer>/      tokenizer.json
    token_cache/                 token ids, one folder per corpus and tokenizer
outputs/
    independent_runs/
        <run>/                   a run trained on its own
            config.yaml          every setting, so the run can be reproduced
            train_log.txt
            loss_curve.csv
            model/               the trained model, loadable with sentence-transformers
            scores/<benchmark>/  metrics.csv and per_mapping.csv: the metrics, and every mapping's rank
        ledger.csv               one row per run: its settings, validation loss and training time
        index.csv                one row per scoring: the run, the benchmark and the metrics
    series/<series>/
        runs/<run>/              as above
        ledger.csv, index.csv    as above, for the series' runs
        tables/                  the comparison tables `experiment report` writes
        figures/                 its figures, one per comparison family
        pick.md                  the record `experiment pick` writes
    scores/<model>/<benchmark>/  the scores of a model trained elsewhere, such as uc-ctds/StdElement-Mapper
```

A series is read from two tables: the ledger, with one row per training run,
and the index, with one row per scoring. Both can be rebuilt from the run
folders. A run trained on its own has the same two, but no tables or
figures, which come from comparing a series' runs.

The only files the tools write that are committed are the benchmark
manifests, under `embedding_model/manifests/`: the SHA-256 of every benchmark
file, and the version and licence of each source, so a rebuilt benchmark
either matches its manifest or names the file that does not. The benchmark
files themselves carry third-party text, such as dbGaP variable descriptions
and the GDC dictionary, so they are built rather than redistributed.

## Running experiments

Model development is a series of experiments, each a set of arms -- versions
of the training configuration that differ in one setting -- trained at several
seeds, so that a difference between arms can be told apart from the difference
between seeds. An experiment file --
[examples/small_series_example.yaml](examples/small_series_example.yaml) is
one -- lists the arms, each as a change from a base configuration, and the
comparisons to make between them, grouped into families. `experiment report`
gives each comparison's verdict -- which arm the decision rule keeps -- and
that arm is the base the next family is declared against. `experiment pick`
writes the series' pick record once the series is complete: the reproduction,
the confirmation on fresh seeds, the run that ships, and how it compares with
the model in use.

[PROCESS.md](PROCESS.md) describes how a series is planned, run, read and
decided: the stages a series goes through, from choosing the corpus to
confirming the pick on fresh seeds; the decision rule; and how many seeds a
comparison needs.

## Tests

The tests live with the rest of the repository's, in `tests/`, as
`tests/test_embedding_model_*.py`. From the repository root:

```
uv run pytest tests/ -k embedding_model
```

runs them; `uv run pytest tests/` runs every test in the repository.
