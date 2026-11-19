# Embedding model

Tools to build, train and evaluate a small embedding model. The model maps
study variables to data-model properties. In a mapping, the study variable is
the source variable. The data-model property is its target: a property of a
node of the data model, written `node.property`. The tools cover:

- the [benchmarks](docs/benchmarks.md) of expert mappings
- the training data
- a [tokenizer](docs/tokenizer.md)
- the model
- the [experiments](docs/experiments.md) that decide between versions of the
  model

By default, the model is a [static embedding table](#how-the-model-works),
trained from scratch. It is small enough to encode thousands of texts a second
on a CPU. Everything here runs from the command line, from a checkout of this
repository.

The tools build, train and score the model with
[sentence-transformers](https://sbert.net), the library for embedding models
that UKP Lab created and Hugging Face maintains. The trainer of
sentence-transformers builds on the `Trainer` of Hugging Face's
[`transformers`](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer#transformers.Trainer). So a trained model
is a sentence-transformers model, and it loads wherever one does.
[docs/design_decisions.md](docs/design_decisions.md#the-model-is-built-trained-and-scored-with-sentence-transformers)
says why.

## How it fits with the rest of the repository

The harmonization approaches of `ai_harmonization` suggest mappings from a
source data model to a target data model. One of them, the embedding approach, uses a trained model as
it uses any other [sentence-transformers](https://sbert.net) model. You can
evaluate the model end to end, as part of that approach, with
`get_metrics_for_approach` (`ai_harmonization/harmonization_benchmark.py`).
`jupyter/embedding_model_training.ipynb` fine-tunes a pretrained model, while
`embedding_model` trains a small model from scratch. Both produce
sentence-transformers models.

A model reads texts in training and in scoring, for example the text of a
study variable or of a data-model property. By default, these texts have one
[format](docs/scoring.md#text-formats), `compact`, which
`embedding_model/texts.py` defines. `compact` is the format
`name_type_description_values`, which a formatter of
`ai_harmonization.formatters` writes, with one change: `compact` writes a description only when there is one.

The harmonization approaches write their texts with the formatters of
`ai_harmonization.formatters`. So a model trains and is scored on texts in
almost the format that it reads in use.
[docs/scoring.md](docs/scoring.md#text-formats) gives the differences.
[docs/design_decisions.md](docs/design_decisions.md#a-source-variables-text-leaves-its-node-out)
says why a source variable's text has no node.

By default, training texts and benchmark texts have the same format.
[`corpus build`](docs/corpus.md) drops the [pairs](docs/pairs.md) that would
teach a model a benchmark's mappings. An example is a benchmark's source
variable, paired with its expected target: the target that experts mapped it
to. A text that a pair shares with a benchmark, for example `age`, is not
[leakage](docs/leakage.md) by itself. A shared text alone does not make
`corpus build` drop the pair.

## Setup

```
uv sync
uv pip install -e .
```

`uv sync` installs everything into the environment `.venv/`. Run the commands
of this guide in that environment. Activate it once in each terminal, with
`source .venv/bin/activate`. Or start each command with `uv run`, for example
`uv run python -m embedding_model --help`.

`uv sync` also installs `embedding_model`'s own dependencies, for example
[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) and
[pyarrow](https://arrow.apache.org/docs/25.0/python/index.html). They are in the
`embedding-model` dependency group of `pyproject.toml`, so installing
`ai_harmonization` as a package leaves them out. With pip 25.1 or later,
`pip install --group embedding-model` installs them.

Everything runs on a CPU, model training included. Training and scoring use a
GPU if the machine has one. `train --use_cpu` and `score --device cpu` make
them use the CPU. The [quick start](#quick-start) examples train on a CPU in a
few minutes. A static model on all the sources trains in minutes too, on a
CPU. A [BERT](docs/training.md#architectures) model trains far more slowly,
and needs a GPU for all the sources.

`train` trains one model. An [experiment](docs/experiments.md) asks whether a
change to the settings helps. So `experiment train` trains many models: one
with the settings, and one with each change, each several times with different
seeds.

`train` and `experiment train` train on the machine where they run.
`experiment train` trains one model per GPU at a time, or one at a time
without a GPU. On a cluster,
either command runs inside a job that asks for its GPUs.
[docs/training.md](docs/training.md#on-a-cluster-with-slurm) and
[docs/experiments.md](docs/experiments.md#on-a-cluster-with-slurm) show how
with [SLURM](https://slurm.schedmd.com/).

## Quick start

All commands here, and in the rest of this guide, run from the repository
root.

`embedding_model/examples/run_example.yaml` is a config to start from: a YAML
file that holds the settings to train one model. It uses one benchmark, the
[TOPMed benchmark](docs/benchmarks.md#topmed), and one source of training
data, [CDISC](docs/sources.md#cdisc). CDISC gives pairs of the terms that
clinical trials report their data with, for example the lab test code `HGB`
and its name, `Hemoglobin Measurement`. Its manifest pins each release, so
every fetch reads the same files. The example uses a small vocabulary and a
short training, so it finishes in minutes. It is a small example to see every
step work. A model meant for use should train on more sources.

### Train and score one model

```
python -m embedding_model benchmarks build --topmed
python -m embedding_model sources fetch --source cdisc
python -m embedding_model pairs prepare --source cdisc
python -m embedding_model corpus build --name example --source cdisc  # checked against the benchmarks built: here the TOPMed benchmark
python -m embedding_model tokenizer train --corpus example --vocab-size 8000
python -m embedding_model train --config embedding_model/examples/run_example.yaml --seed 0
python -m embedding_model score --model embedding_model/outputs/runs/run_example_s0 --benchmark topmed
```

Each training is a run: one config, trained with one seed. `train` names a run
after its config and its seed. So this run's folder is
`outputs/runs/run_example_s0/`. [docs/training.md](docs/training.md) says what
a run's folder holds.

### Compare it with a released model

```
python -m embedding_model score --model uc-ctds/bge-large-en-v1.5-bio-mapping --benchmark topmed
python -m embedding_model compare --name example \
    --model embedding_model/outputs/runs/run_example_s0 uc-ctds/bge-large-en-v1.5-bio-mapping \
    --reference embedding_model/outputs/runs/run_example_s0 \
    --benchmark topmed
```

`score` downloads the released model from Hugging Face, about 1.3 GB, and
scores it. `compare` writes to `outputs/comparisons/example/`:

- each model's metrics, most with a 95% confidence interval
- the paired tests of the other model against the reference, the example run
- four figures

[docs/comparing.md](docs/comparing.md) says what each figure shows.

### Run a small experiment

`embedding_model/examples/experiment_example.yaml` is a small
[experiment](docs/experiments.md) on the same data. Its base is the example
run's config. Its two arms are versions of the base: one makes the table
smaller, and one makes it larger. At five seeds, the base and the two arms make
fifteen runs.

```
# The same data as the first example: skip these if you have run it.
python -m embedding_model benchmarks build --topmed
python -m embedding_model sources fetch --source cdisc
python -m embedding_model pairs prepare --source cdisc
python -m embedding_model corpus build --name example --source cdisc
python -m embedding_model tokenizer train --corpus example --vocab-size 8000

python -m embedding_model experiment train --config embedding_model/examples/experiment_example.yaml
python -m embedding_model experiment status --config embedding_model/examples/experiment_example.yaml
python -m embedding_model experiment compare --config embedding_model/examples/experiment_example.yaml
```

`experiment train` trains the fifteen runs, one after another on a CPU, or one
per GPU at a time. `experiment status` then shows each arm's runs as done.
`experiment compare` compares each arm with the base. It prints each arm's
[verdict](docs/experiments.md#comparing-the-arms) and the arm that it
recommends, and draws the figure `comparison/figures/arms_and_differences.png`.

All three commands use `outputs/experiments/experiment_example/`. The
experiment is a small example to see the experiment tools work.
[docs/experiments.md](docs/experiments.md) says how to write your own.

## How the model works

The model is a static embedding table. The table has one row per token of the
[tokenizer](docs/tokenizer.md)'s vocabulary. Each row holds a vector: a list
of numbers. This example shows 3 numbers per row, with made-up ids and
numbers:

| id | token | vector |
|---|---|---|
| 812 | `body` | 0.2, 0.9, 0.1 |
| 1407 | `weight` | 0.7, 0.1, 0.4 |
| 2350 | `mass` | 0.6, 0.2, 0.5 |

The table has two sizes:

- the vocabulary size sets the number of rows, one per token
- the dimension sets how many numbers each row holds

So a table of 16,000 tokens and 384 dimensions holds 16,000 × 384, about 6.1
million numbers. It has no attention and no layers, so it is small and fast.

A text's embedding is the average of its tokens' vectors. The tokenizer cuts
`Body Weight` into `body` and `weight`. The model averages their two rows,
number by number. So the embedding of `Body Weight` is 0.45, 0.5, 0.25.

Cosine similarity compares two embeddings. A value near 1 means that the two
texts are alike. At the start, every row holds random numbers. Training sets
the numbers, as the paragraphs below describe.

The tokenizer is a Hugging Face
[`tokenizers`](https://huggingface.co/docs/tokenizers/v0.23.2/en/index) pipeline, in three
steps:

1. It cleans a text, lowercases it and drops its accents.
2. It splits the text at whitespace and punctuation.
3. It cuts each piece into tokens by
   [byte-pair encoding (BPE)](https://huggingface.co/learn/llm-course/chapter6/5).

[docs/tokenizer.md](docs/tokenizer.md) explains each step, with examples.

`train` trains the table from scratch with a contrastive loss (in-batch
negatives). The table learns from [pairs](docs/pairs.md) of texts that mean
the same thing, for example:

- a variable and its description
- a term and its synonym
- two names for one data-model property

A pair has two sides, the [anchor and the positive](docs/pairs.md). In each
batch, the loss pulls an anchor toward its own positive, and pushes the anchor
away from the other pairs' positives.

The static embedding model is the default. A config can choose another
architecture: `bert`, a transformer embedding model with BERT's architecture,
trained from scratch in the same way. Its attention layers let a token's
vector depend on the other tokens of the text. It is larger and slower than
the static model.
[docs/training.md](docs/training.md#architectures) lists the settings of
both.

## The workflow

Every step is a subcommand of one command, run from the repository root:
`python -m embedding_model <step>`. `python -m embedding_model --help` lists
the steps. Each step reads the files that the step before it wrote. So you can
run a step again on its own.

| | step | what it does | writes |
|---|---|---|---|
| benchmarks | `benchmarks build --topmed` | builds the [TOPMed benchmark](docs/benchmarks.md#topmed) | `inputs/benchmarks/` |
| | `benchmarks build --gdc` | builds the [GDC benchmark](docs/benchmarks.md#gdc) | `inputs/benchmarks/` |
| | `score --model <model> --benchmark <benchmark> [<benchmark> ...] [--format <format>]` | scores any model, trained here or elsewhere, on one or more benchmarks | `scores/<benchmark>/<format>/` in the run's folder, or `outputs/scores/<model>/<benchmark>/<format>/` for a model trained elsewhere: the metrics and every source variable's rank |
| | `compare --name <comparison> --model <model> [<model> ...] [--reference <model>] --benchmark <benchmark> [<benchmark> ...] [--format <format>] [--colors <color> [<color> ...]]` | [compares](docs/comparing.md) scored models: their metrics with 95% confidence intervals, paired tests of each model against a reference, and figures | `outputs/comparisons/<comparison>/` |
| training data | `sources fetch [--source <source> ...]` | downloads the [sources of training data](docs/sources.md), or some of them | `inputs/sources/` |
| | `pairs prepare [--source <source> ...]` | turns each source into [pairs](docs/pairs.md) of texts that mean the same thing | `inputs/pairs/<source>/` |
| | `corpus build --name <corpus> [--source <source> ...] [--validation-share <share>]` | combines the sources' pairs into a [corpus](docs/corpus.md): drops the pairs that would leak a benchmark built so far, and splits the rest into training and validation pairs | `inputs/corpora/<corpus>/` |
| | `tokenizer train --corpus <corpus> [--vocab-size <size>] [--min-frequency <n>] [--name <tokenizer>]` | learns the [tokenizer](docs/tokenizer.md)'s vocabulary from the corpus's training pairs | `inputs/tokenizers/<tokenizer>/` |
| one model | `train --config <config.yaml> --seed <seed>` | [trains](docs/training.md) one model, as one run | `outputs/runs/` |
| experiments | `experiment train --config <experiment.yaml>` | trains each run of an [experiment](docs/experiments.md) that is not done: the base and each arm, at each seed, one run per GPU at a time | `outputs/experiments/<experiment>/` |
| | `experiment status --config <experiment.yaml>` | says which runs are done, started or not started | -- |
| | `experiment compare --config <experiment.yaml>` | [compares](docs/experiments.md#comparing-the-arms) each arm with the base, by its seeds' validation scores, and recommends an arm to build on | `comparison/` in the experiment's folder |

`<experiment.yaml>` is an experiment's file: its base, its arms, its seeds and
its decision rule.
[docs/experiments.md](docs/experiments.md#writing-an-experiments-file) says
how to write one.

`corpus build` checks the pairs against every benchmark built so far. It drops
the pairs that would leak a benchmark, as [docs/leakage.md](docs/leakage.md)
describes. A corpus's [manifest](docs/manifests.md) lists the benchmarks that
`corpus build` checked the corpus's pairs against. `score` refuses a run if the
run's corpus was not checked for leakage against one of the benchmarks that
`--benchmark` names.

## Evaluating a model

`score` ranks every target of a benchmark for each source variable. It reports
where the expected targets land: the mean reciprocal rank (MRR), MRR@10,
nDCG@10, accuracy@1, @3, @5 and @10, and the median rank.
[docs/scoring.md](docs/scoring.md) explains:

- how `score` scores a model, and what each metric means
- how to run `score`, and in which format
- what its output files hold

There are two benchmarks:

- **[TOPMed](docs/benchmarks.md#topmed)**:
  [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) study variables that experts
  mapped to the harmonized variables of the TOPMed Data Coordinating Center
- **[GDC](docs/benchmarks.md#gdc)**: the GDC-SM ground truth, published with
  the [Magneto paper](https://doi.org/10.14778/3742728.3742757): variables from
  cancer papers, mapped by experts to the GDC data dictionary

[docs/benchmarks.md](docs/benchmarks.md) describes each benchmark's inputs, and
how to build the benchmark or make a new version of it.

`compare` puts scored models side by side. It gives each model's metrics,
with a 95% confidence interval for MRR, MRR@10 and each accuracy@k. With `--reference`, it also tests each other
model against that model. It draws three figures, ready for print, and a
fourth with `--reference`. [docs/comparing.md](docs/comparing.md) says how, and what each
figure shows.

`get_metrics_for_approach` evaluates a harmonization approach end to end. The
approach suggests a set of mappings for a source data model and a target data
model. `get_metrics_for_approach` scores the suggestions by precision, recall
and F1. Use it to evaluate a trained model as part of an approach.

## Inputs and outputs

The steps that prepare data write under `embedding_model/inputs/`:
`benchmarks build`, `sources fetch`, `pairs prepare`, `corpus build` and
`tokenizer train`. A training run reads what it needs from there. `train`,
`score`, `compare` and `experiment` write under `embedding_model/outputs/`:
the runs with their models and records, the scores, the experiments and the
comparisons. Git ignores both folders.

`--inputs DIR` and `--outputs DIR` set a different location for either one,
for example a larger disk.

The large tables, for example a corpus's pairs, are
[parquet](https://parquet.apache.org/) files. The benchmarks and the scores are
CSV files. To look at a parquet file, read it with pandas. Or open it in VS
Code with an extension, for example
[Parquet Explorer](https://marketplace.visualstudio.com/items?itemName=AdamViola.parquet-explorer)
or Microsoft's
[Data Wrangler](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.datawrangler).

A step writes each of these folders whole:

- `benchmarks build`: a benchmark
- `sources fetch`: a source's tables
- `pairs prepare`: a source's pairs
- `corpus build`: a corpus
- `tokenizer train`: a tokenizer
- `train`: a training run
- `score`: a model's scores
- `compare`: a comparison
- `experiment train` and `experiment compare`: an experiment's configs and its
  comparison

For example, `corpus build --name abcd` first writes the new corpus into a
hidden staging folder, `corpora/.abcd~x1y2/`, next to `corpora/abcd/`. The new
corpus replaces `corpora/abcd/` only when the new corpus is complete. So a
build that fails leaves the previous corpus as it was.

If you stop a step with Ctrl-C, the step deletes its staging folder. A step
that is killed, for example by `scancel` or a time limit, cannot delete its
staging folder. The next `corpus build --name abcd` then prints a warning with
that folder's path. The warning says what to do with the folder. Usually
nothing in it is needed, so you delete it.

A step can also be killed while it swaps the two folders. Then the staging
folder holds the previous corpus, and the warning says how to move the
previous corpus back.

```
inputs/
    benchmarks/
        <benchmark>/             <benchmark>_mappings.csv, <benchmark>_targets.csv, and the build's manifest.json
        downloads/<benchmark>/   the published inputs as downloaded, kept for the next build
    sources/
        <source>/                a source's tables, as <table>.parquet, and the fetch's manifest.json
        downloads/<source>/      the source's files as downloaded, kept for the next fetch
    pairs/<source>/              pairs.parquet: a source's pairs of texts; manifest.json, their manifest
    corpora/<corpus>/            pairs.parquet, the kept pairs with their split; dropped.parquet; manifest.json, the corpus's manifest
    tokenizers/<tokenizer>/      tokenizer.json; manifest.json, the tokenizer's manifest
outputs/
    runs/
        <run>/                   one run, trained on its own
            config.yaml          the run's settings, so the run can be repeated
            manifest.json        the run's record: its corpus, tokenizer, validation metrics and versions
            trainer_state.json   the trainer's log: the training loss and the validation metrics, step by step
            model/               the trained model, loadable with sentence-transformers
            scores/<benchmark>/<format>/  metrics.csv, per_variable.csv and config.yaml: the metrics, every source variable's rank, and what they were computed with
    experiments/<experiment>/
        experiment.yaml          the experiment's file, as it was first trained
        configs/                 each arm's full config, the base's first
        runs/<arm>_s<seed>/      each run, as above
        logs/                    each run's output
        comparison/              what `experiment compare` writes: arms.csv, differences.csv, manifest.json, and figures/ with its figure
    scores/<model>/              the scores of a model trained elsewhere, such as uc-ctds/bge-large-en-v1.5-bio-mapping
        <benchmark>/<format>/    one folder per benchmark and text format
    comparisons/<comparison>/    what `compare` writes: metrics.csv, differences.csv, manifest.json, and figures/ with the figures
```

Of the files that the tools write, only the manifests of the benchmarks and of
the sources of training data are committed. They are under
`embedding_model/benchmarks/manifests/` and
`embedding_model/sources/manifests/`. A manifest holds the
[SHA-256](https://doi.org/10.6028/NIST.FIPS.180-4) of every file that its build
makes, and the version and license of each input. So a rebuilt benchmark or a
new fetch either matches its manifest, or the build names what differs.
[docs/manifests.md](docs/manifests.md#making-a-new-version) says how to accept
a new version.

The built files carry third-party text, for example
[dbGaP](https://www.ncbi.nlm.nih.gov/gap/) variable descriptions, the
[GDC dictionary](docs/benchmarks.md#gdc) and
[LOINC's names](docs/sources.md#loinc). So each machine builds them, and the
repository does not redistribute them. [docs/licenses.md](docs/licenses.md)
gives the license and reference of each benchmark and source.

## Running experiments

Model development is a series of experiments. Each experiment asks one
question: does a change to the base help? The base is a run's config, and the
experiment's arms are versions of the base that change some settings.
`experiment train` trains the base and each arm at several seeds. So
`experiment compare` can tell a difference between the arms from the
difference between seeds.

Each experiment builds on an arm of the experiment before it. You choose that
arm. `experiment compare` recommends an arm, and prints the line that names
the arm as the next experiment's base. The
[guide to experiments](docs/experiments.md) says how to write an experiment's
file, train it, follow it and compare its arms. The quick start's
[small experiment](#run-a-small-experiment) shows one.
[docs/design_decisions.md](docs/design_decisions.md#experiments) gives the
reasons for the rules of experiments.

## Tests

The tests live with the repository's other tests, in `tests/`, as
`tests/test_embedding_model_*.py`. This command runs them from the repository
root:

```
uv run pytest tests/ -k embedding_model
```

`uv run pytest tests/` runs every test in the repository.
