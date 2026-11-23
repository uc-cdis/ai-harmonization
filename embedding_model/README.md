# Embedding model

Tools to build, train and evaluate a small embedding model. The model maps
study variables to data-model properties. In a mapping, the study variable is
the source variable. The data-model property is its target: a property of a
node of the data model, written `node.property`.

By default, the model is a [static embedding table](docs/training.md#architectures),
trained from scratch. It is small enough to encode thousands of texts a second
on a CPU. The tools build, train and score the model with
[sentence-transformers](https://sbert.net). So a trained model is a
sentence-transformers model, and it loads wherever one does. Everything runs
from the command line, from a checkout of this repository.

The benchmarks and the training data come from third-party sources, each under
its publisher's license. This repository does not redistribute them. It gives
you the tools that download them from their publishers, and build them on
your machine. [docs/licenses.md](docs/licenses.md) gives each license and
reference.

## Setup

```
uv sync
uv pip install -e .
```

`uv sync` installs everything into the environment `.venv/`. Run the commands
of this guide in that environment. Activate it once in each terminal, with
`source .venv/bin/activate`. Or start each command with `uv run`, for example
`uv run python -m embedding_model --help`.

`embedding_model`'s own dependencies are in the `embedding-model` dependency
group of `pyproject.toml`. With pip 25.1 or later,
`pip install --group embedding-model` installs that group.

Everything runs on a CPU, model training included. Training and scoring use a
GPU if the machine has one. `train --use_cpu` and `score --device cpu` make
them use the CPU. [GPUs and CPUs](docs/training.md#gpus-and-cpus) says how
long training takes on each. On a cluster, run `train` or `experiment train`
inside a job that asks for its GPUs.
[docs/training.md](docs/training.md#on-a-cluster-with-slurm) and
[docs/experiments.md](docs/experiments.md#on-a-cluster-with-slurm) show how
with [SLURM](https://slurm.schedmd.com/).

## Quick start

All commands here, and in the rest of this guide, run from the repository
root. In this guide, `inputs/` and `outputs/` are `embedding_model/inputs/`
and `embedding_model/outputs/`.

A run is one training of one config with one seed.
`embedding_model/examples/run_example.yaml` is a config to start from: a YAML
file that holds the settings to train one model. It names the corpus and the
tokenizer. Its training settings are the defaults, except pinned memory,
which a CPU does not use. With the commands below,
the run trains on one source of training data,
[CDISC](docs/sources.md#cdisc), and `score` scores it on the
[TOPMed benchmark](docs/benchmarks.md#topmed). The corpus is small, so the
run finishes in minutes on a CPU. A model meant for use should train on more
sources.

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

`train` names a run after its config and its seed. So this run's folder is
`outputs/runs/run_example_s0/`. [docs/training.md](docs/training.md) says what
a run's folder holds. Training a run again with the same settings replaces
its folder, and with it the run's scores. With other settings, `train`
refuses: delete the run's folder, or save the changed config under a new
name.

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
- the paired tests of the released model against the reference, which is
  the example run
- four figures

[docs/comparing.md](docs/comparing.md) says what each figure shows.

### Run a small experiment

`embedding_model/examples/experiment_example.yaml` is a small
[experiment](docs/experiments.md) on the same data. Its base is the example
run's config. Its two arms are versions of the base: one makes the model's
embedding table smaller, and one makes it larger. At five seeds, the base and the two arms make
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

All three commands use `outputs/experiments/experiment_example/`.
`experiment train` trains the fifteen runs, one after another on a CPU, or one
per GPU at a time. `experiment status` then shows each arm's runs as done.
`experiment compare` compares each arm with the base. It prints each arm's
[verdict](docs/experiments.md#comparing-the-arms) and the arm that it
recommends. In the experiment's folder, it draws the figure
`comparison/figures/arms_and_differences.png`.
[docs/experiments.md](docs/experiments.md) says how to write your own
experiment.

## The workflow

Every step is a subcommand of one command, run from the repository root:
`python -m embedding_model <step>`. `python -m embedding_model --help` lists
the steps. Each step reads the files that the step before it wrote. So you can
run a step again on its own.

| | step | what it does | writes |
|---|---|---|---|
| benchmarks | `benchmarks build --topmed` | builds the [TOPMed benchmark](docs/benchmarks.md#topmed) | `inputs/benchmarks/` |
| | `benchmarks build --gdc` | builds the [GDC benchmark](docs/benchmarks.md#gdc) | `inputs/benchmarks/` |
| training data | `sources fetch [--source <source> ...]` | downloads the [sources of training data](docs/sources.md), or some of them | `inputs/sources/` |
| | `pairs prepare [--source <source> ...]` | turns each source into [pairs](docs/pairs.md) of texts that mean the same thing | `inputs/pairs/<source>/` |
| | `corpus build --name <corpus> [--source <source> ...] [--validation-share <share>]` | combines the sources' pairs into a [corpus](docs/corpus.md): drops the pairs that would leak a benchmark built so far, and splits the rest into training and validation pairs | `inputs/corpora/<corpus>/` |
| | `tokenizer train --corpus <corpus> [--vocab-size <size>] [--min-frequency <n>] [--name <tokenizer>]` | learns the [tokenizer](docs/tokenizer.md)'s vocabulary from the corpus's training pairs | `inputs/tokenizers/<tokenizer>/` |
| one model | `train --config <config.yaml> --seed <seed>` | [trains](docs/training.md) one model, as one run | `outputs/runs/` |
| evaluation | `score --model <model> --benchmark <benchmark> [<benchmark> ...] [--format <format>]` | scores any model, trained here or elsewhere, on one or more benchmarks | `scores/<benchmark>/<format>/` in the run's folder, or `outputs/scores/<model>/<benchmark>/<format>/` for a model trained elsewhere: the metrics and every source variable's rank |
| | `compare --name <comparison> --model <model> [<model> ...] [--reference <model>] --benchmark <benchmark> [<benchmark> ...] [--format <format>] [--colors <color> [<color> ...]]` | [compares](docs/comparing.md) scored models: their metrics, paired tests of each model against a reference, and figures | `outputs/comparisons/<comparison>/` |
| experiments | `experiment train --config <experiment.yaml>` | trains each run of an [experiment](docs/experiments.md) that is not done: the base and each arm, at each seed, one run per GPU at a time | `outputs/experiments/<experiment>/` |
| | `experiment status --config <experiment.yaml>` | says which runs are done, started or not started | -- |
| | `experiment compare --config <experiment.yaml>` | [compares](docs/experiments.md#comparing-the-arms) each arm with the base, by its seeds' validation scores, and recommends an arm to build on | `comparison/` in the experiment's folder |

`<experiment.yaml>` is an experiment's file: its base, its arms, its seeds and
its decision rule.
[docs/experiments.md](docs/experiments.md#writing-an-experiments-file) says
how to write one.

[docs/leakage.md](docs/leakage.md) says which pairs `corpus build` drops.

**Build every benchmark that you will score on before `corpus build`.**
`score` refuses a run whose corpus was not checked against the current build
of each benchmark that `--benchmark` names. After a new benchmark build, build
the corpus again, then train its tokenizer again, then train again.

## Inputs and outputs

The steps that prepare data write under `embedding_model/inputs/`. `train`,
`score`, `compare` and `experiment` write under `embedding_model/outputs/`.
Git ignores both folders. `--inputs DIR` and `--outputs DIR` set another place
for either folder, for example a larger disk:

- Every step takes `--inputs DIR`, except `experiment status` and
  `experiment compare`.
- `train`, `score`, `compare` and the three `experiment` commands take
  `--outputs DIR`.

| folder | what it holds | written by |
|---|---|---|
| `inputs/benchmarks/<benchmark>/` | `<benchmark>_mappings.csv`, `<benchmark>_targets.csv` and the build's `manifest.json` | `benchmarks build` |
| `inputs/sources/<source>/` | the source's tables, as `<table>.parquet`, and the fetch's `manifest.json` | `sources fetch` |
| `inputs/benchmarks/downloads/`, `inputs/sources/downloads/` | the files as downloaded, kept for the next build or fetch | `benchmarks build`, `sources fetch` |
| `inputs/pairs/<source>/` | `pairs.parquet`, the source's pairs, and their `manifest.json` | `pairs prepare` |
| `inputs/corpora/<corpus>/` | `pairs.parquet`, the kept pairs with their split, `dropped.parquet` and the corpus's `manifest.json` | `corpus build` |
| `inputs/tokenizers/<tokenizer>/` | `tokenizer.json` and the tokenizer's `manifest.json` | `tokenizer train` |
| `outputs/runs/<run>/` | the trained model, the run's config, manifest and trainer's state, and its scores | `train`, `score` |
| `outputs/experiments/<experiment>/` | the experiment's file, each arm's config, the runs, their logs and the comparison | `experiment train`, `experiment compare` |
| `outputs/scores/<model>/` | the scores of a model trained elsewhere | `score` |
| `outputs/comparisons/<comparison>/` | the metrics, the differences, the manifest and the figures | `compare` |

A step writes each of its folders whole: if the step fails, the previous
folder stays as it was. For `score`, each benchmark and format has its own
folder, `scores/<benchmark>/<format>/`, so a model's other scores stay. There
are two exceptions:

- The downloads are kept file by file.
- An experiment's folder fills run by run, and its logs grow as the runs
  train. Its configs and its comparison are written whole.

## Tests

The tests live with the repository's other tests, in `tests/`, as
`tests/test_embedding_model_*.py`. This command runs them from the repository
root:

```
uv run pytest tests/ -k embedding_model
```

`uv run pytest tests/` runs every test in the repository.
