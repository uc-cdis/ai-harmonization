# Embedding model

Tools to build, train and evaluate a small embedding model. The model maps
study variables to data-model properties. The tools cover:

- a benchmark of expert mappings;
- the training data;
- a tokenizer;
- the model;
- the experiments that decide between versions of it.

The model is a static embedding table trained from scratch. It is small enough
to encode thousands of texts a second on a CPU. Everything here runs from the
command line, from a checkout of this repository.

## How it fits with the rest of the repository

- A model trains and is scored on texts such as a study variable or a
  data-model property. They are written in one format, `compact`, defined in
  `embedding_model/texts.py`. It is the layout of
  `ai_harmonization.formatters`, which the harmonization approaches use, with
  one change: a description appears only when there is one. So a model is
  trained and evaluated on texts laid out almost as it will see them in use.
  See [docs/scoring.md](docs/scoring.md#running-score) for the differences,
  and for why a source's node is left out. Training and benchmark texts share
  their layout, but never a text. The corpus is audited for leakage, so no
  benchmark text, and no near copy of one, reaches training.
- A trained model is used through `ai_harmonization`'s embedding approach,
  like any other sentence-transformers model. It can be evaluated end to end
  with `get_metrics_for_approach`
  (`ai_harmonization/harmonization_benchmark.py`).
- `jupyter/embedding_model_training.ipynb` fine-tunes a pretrained model, while
  `embedding_model` trains a small model from scratch. Both produce
  sentence-transformers models.

## Setup

```
uv sync
uv pip install -e .
```

`uv sync` also installs `embedding_model`'s own dependencies, such as
BeautifulSoup and pyarrow. They are in the `embedding-model` dependency group
of `pyproject.toml`, so installing `ai_harmonization` as a package leaves them
out. With pip 25.1 or later, `pip install --group embedding-model` installs
them.

Everything runs on a CPU, model training included. A GPU matters for training,
and training uses one if the machine has one. `--device cpu` keeps it on the
CPU. The [quick start](#quick-start) examples train on a CPU in a few minutes.
A model trained on all the sources takes minutes on a GPU or hours on a CPU,
and a series of experiments trains many models.

A series' runs are trained in one of two ways:

- `experiment submit` queues each run as its own SLURM job (`sbatch`), with
  one GPU each, so they train in parallel. Use it on a cluster. Pass your
  cluster's settings, such as partition and time limit, through `--slurm`.
- `experiment submit --local` trains them one after another on the machine that
  runs the command. It uses that machine's GPU if it has one, and otherwise the
  CPU. The machine can be a laptop, a workstation, or a cluster node that you
  have been allocated, such as inside `srun --gpus=1 --pty bash`. Do not use a
  shared login node. Use `--local` for small series and quick tests.

## Quick start

All commands here, and in the rest of this guide, run from the repository
root.

`embedding_model/examples/` holds two files to start from. Each stands on its
own:

- `independent_run_example.yaml`: the configuration for one training run;
- `small_series_example.yaml`: a small series of experiments.

Both use one benchmark, [TOPMed](docs/benchmarks.md#topmed), and one training
source, [NHANES](docs/sources.md#nhanes). NHANES gives about 29,000 pairs of
variable names, descriptions and values from CDC's public codebooks. The
examples use a small vocabulary and a short run, so they finish in minutes.
They are small examples to see every step work. A model meant for use should
be trained on more sources.

### Train and score one model

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

### Run a small series

`small_series_example.yaml` declares a base configuration and two arms at five
seeds. Each arm changes one setting, so the series has fifteen runs. The
example shows how a series of experiments is run and read. A series meant to
decide something needs more seeds, as [docs/process.md](docs/process.md) explains.

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

`--local` trains the fifteen runs on the machine, one after another. That is
useful for a quick test on a laptop. Without `--local`, each run is submitted
to SLURM as its own job, and they train at the same time.

## How the model works

The model is a static embedding table. It holds one vector per token of a
vocabulary, and a text's embedding is the average of its tokens' vectors. A
table has no attention and no layers, so it is small (a few million
parameters) and fast.

The tokenizer is a Hugging Face
[`tokenizers`](https://huggingface.co/docs/tokenizers) pipeline with three
steps:

1. Text is lowercased and stripped of accents.
2. It is split at whitespace and at every punctuation character, by BERT's rule
   (`BertPreTokenizer`). So `SUBJECT_ID` becomes `subject`, `_`, `id`.
3. Each piece is split into vocabulary tokens by
   [byte-pair encoding (BPE)](https://huggingface.co/learn/llm-course/chapter6/5).

BPE learns its vocabulary from the corpus. It starts from characters and keeps
merging the most frequent adjacent pair. So common words become one token,
and a rare word splits into known pieces rather than becoming unknown. A token
never spans two words.

The table is trained from scratch with a contrastive loss (in-batch
negatives). It learns from pairs of texts that mean the same thing, such as:

- a variable and its description;
- a term and its synonym;
- two names for one data-model property.

Each pair is pulled together, and pushed apart from every other text in its
batch.

## The workflow

Every step is a subcommand of one command, run from the repository root:
`python -m embedding_model <step>`. `python -m embedding_model --help` lists
them. Each step reads the files that the step before it wrote, so a step can be
rerun on its own.

| | step | what it does | writes |
|---|---|---|---|
| benchmarks | `benchmarks build --topmed` | builds [TOPMed](docs/benchmarks.md#topmed), the benchmark that models are developed on | `inputs/benchmarks/` |
| | `benchmarks build --gdc` | builds [GDC](docs/benchmarks.md#gdc), held back: scored once per series' pick (the series' winner) | `inputs/benchmarks/` |
| | `score --model <model> --benchmark <benchmark> [<benchmark> ...] [--format <format>]` | scores any model, trained here or elsewhere, on one or more benchmarks | `scores/<benchmark>/` in the run's folder, or `outputs/scores/<model>/<benchmark>/<format>/` for a model trained elsewhere: the metrics and every source variable's rank |
| training data | `sources fetch [--source <source> ...]` | downloads the [training sources](docs/sources.md), or some of them | `inputs/sources/` |
| | `pairs prepare [--source <source> ...]` | turns each source into [pairs](docs/pairs.md) of texts that mean the same thing | `inputs/pairs/` |
| | `pairs audit` | checks that no pair leaks a text of any benchmark built so far; must pass before a corpus is built | `inputs/pairs/audit.json`: what was checked, against which benchmarks, and the verdict |
| | `corpus build --name <corpus> [--sources <source>,...]` | combines the sources' pairs into a training corpus | `inputs/corpora/<corpus>/` |
| | `tokenizer train --corpus <corpus>` | learns the vocabulary from the corpus | `inputs/tokenizers/` |
| | `cache build --corpus <corpus>` *(optional)* | tokenizes the corpus once, before a series | `inputs/token_cache/` |
| one model | `train --config <config.yaml> --seed <seed>` | trains one model on its own | `outputs/independent_runs/` |
| a series | `experiment submit <series.yaml> [--local]` | trains every run that the series declares -- each arm at each seed: one SLURM job per run, so many train at once, or with `--local` one after another on the machine that runs the command | `outputs/series/<series>/runs/` |
| | `experiment status <series.yaml>` | says which runs are done, running or missing | -- |
| | `experiment report <series.yaml>` | builds tables and figures, with each comparison's verdict | `outputs/series/<series>/` |
| | `experiment pick <series.yaml>` | writes the series' pick record | `outputs/series/<series>/pick.md` |

`<series.yaml>` is an experiment file: the arms of a series, the seeds, and
the comparisons to make.
[examples/small_series_example.yaml](examples/small_series_example.yaml) is
one.

Training builds the token cache itself on first use if `cache build` has not
run. `--no-token-cache` tokenizes inside each training step instead: the same
model, more slowly. `cache build` is worth running before a series, so that
many runs starting at once do not each wait for it.

The audit checks the pairs against every benchmark built so far. Each corpus
records which benchmarks its pairs were checked against. Scoring on a
benchmark refuses a model whose corpus was not checked against that benchmark
for leakage.

## Evaluating a model

`score` ranks every target of a benchmark for each source variable. It reports
where the expected targets land: the mean reciprocal rank (MRR), accuracy@1, @5
and @10, and the median rank. [docs/scoring.md](docs/scoring.md) explains:

- how a model is scored, and what each metric means;
- how to run `score`, and in which text format;
- what its output files hold.

There are two benchmarks:

- **[TOPMed](docs/benchmarks.md#topmed)**: dbGaP study variables mapped by experts
  to the harmonized variables of the TOPMed Data Coordinating Center. As built
  here, it has 1,162 source variables and 148 targets. Models are developed on
  it.
- **[GDC](docs/benchmarks.md#gdc)**: the GDC-SM ground truth, published with the
  Magneto paper: variables from ten cancer papers, mapped by experts to the
  GDC data dictionary. As built here, it has 165 source variables and 1,217
  targets. It is held back, and scored once on a series' pick (the series'
  winner).

[docs/benchmarks.md](docs/benchmarks.md) describes what each one is built from,
and how to build it or make a new version.

`get_metrics_for_approach` evaluates a harmonization approach end to end. The
approach suggests a set of mappings for a source and a target model, and the
suggestions are scored by precision, recall and F1. Use it to evaluate a
trained model as part of an approach.

## Inputs and outputs

The steps that prepare data write under `embedding_model/inputs/`. Those
steps are the benchmarks, sources, pairs, corpora, tokenizers and the token
cache. Training runs read what they need from there. They write everything that
they produce under `embedding_model/outputs/`: the model, its logs and its
scores. Both folders are gitignored.

`--inputs DIR` and `--outputs DIR` set a different location for either one.
That is useful for a full corpus's token cache, which can run to tens of
gigabytes.

```
inputs/
    benchmarks/                  <benchmark>_mappings.csv, <benchmark>_targets.csv, and the build's <benchmark>_manifest.json
        downloads/<benchmark>/   the published inputs as downloaded, kept for the next build
    sources/
        <source>/                a training source's tables, as <table>.parquet, and the fetch's manifest.json
        downloads/<source>/      the source's files as downloaded, kept for the next fetch
    pairs/                       <source>.parquet: each source's pairs of texts; <source>.json, their report
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
            scores/<benchmark>/  metrics.csv and per_variable.csv: the metrics, and every source variable's rank
        ledger.csv               one row per run: its settings, validation loss and training time
        index.csv                one row per scoring: the run, the benchmark and the metrics
    series/<series>/
        runs/<run>/              as above
        ledger.csv, index.csv    as above, for the series' runs
        tables/                  the comparison tables `experiment report` writes
        figures/                 its figures, one per comparison family
        pick.md                  the record `experiment pick` writes
    scores/<model>/              the scores of a model trained elsewhere, such as uc-ctds/StdElement-Mapper
        <benchmark>/<format>/    one folder per benchmark and text format
```

A series is read from two tables: the ledger, with one row per training run,
and the index, with one row per scoring. Both can be rebuilt from the run
folders. A run trained on its own has the same ledger and index. It has no
comparison tables or figures, since those come from comparing a series' runs.

Of the files that the tools write, only the manifests of the benchmarks and the
training sources are committed. They are under
`embedding_model/benchmarks/manifests/` and
`embedding_model/sources/manifests/`. A manifest holds the SHA-256 of every
file built, and the version and license of each input. So a rebuilt benchmark
or a new fetch either matches its manifest or names what does not.
[docs/manifests.md](docs/manifests.md) says how a new version is accepted. The
files themselves carry third-party text, such as dbGaP variable descriptions,
the GDC dictionary and LOINC's names. So they are built rather than
redistributed. [docs/licenses.md](docs/licenses.md) gives the license and
reference of each benchmark and source.

## Running experiments

Model development is a series of experiments. Each experiment is a set of
arms: versions of the training configuration that differ in one setting. Each
arm is trained at several seeds, so that a difference between arms can be told
apart from the difference between seeds.

An experiment file, such as
[examples/small_series_example.yaml](examples/small_series_example.yaml),
lists the arms and the comparisons to make between them. Each arm is a change
from a base configuration, and the comparisons are grouped into families.
`experiment report` gives each comparison's verdict: which arm the decision
rule keeps. The next family is declared against that arm.

`experiment pick` writes the series' pick record once the series is complete.
The record holds:

- the reproduction;
- the confirmation on fresh seeds;
- the run that ships;
- how it compares with the model in use.

[docs/process.md](docs/process.md) describes how a series is planned, run, read and
decided. It covers:

- the stages a series goes through, from choosing the corpus to confirming
  the pick on fresh seeds;
- the decision rule;
- how many seeds a comparison needs.

## Tests

The tests live with the repository's other tests, in `tests/`, as
`tests/test_embedding_model_*.py`. This command runs them from the repository
root:

```
uv run pytest tests/ -k embedding_model
```

`uv run pytest tests/` runs every test in the repository.
