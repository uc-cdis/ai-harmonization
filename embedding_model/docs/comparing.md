# Comparing models

`python -m embedding_model compare` compares models that are already scored.
It reads the scores that [`score`](scoring.md) wrote for each model, on one
or more benchmarks. It writes a table of metrics and three figures. With
`--reference`, you name one of the models as the reference. Then `compare`
also writes each other model's difference from the reference, and a fourth
figure.

## Running `compare`

Score each model first, on the same benchmarks and in the same
[format](scoring.md#text-formats). Then compare them. This example compares
the model of the [quick start](../README.md#quick-start) with a released
model, on the TOPMed benchmark:

```
python -m embedding_model score --model uc-ctds/bge-large-en-v1.5-bio-mapping --benchmark topmed
python -m embedding_model compare --name example \
    --model embedding_model/outputs/runs/run_example_s0 uc-ctds/bge-large-en-v1.5-bio-mapping \
    --reference embedding_model/outputs/runs/run_example_s0 \
    --benchmark topmed
```

| flag | what it sets |
|---|---|
| `--name` | the comparison's name, and its folder: `embedding_model/outputs/comparisons/<name>/`. A second comparison of the same name replaces the first. |
| `--model` | one or more models, up to 8: a Hugging Face model id, a model folder on disk, or the folder of a run that [`train`](training.md) wrote. |
| `--reference` | one of the models: the reference. `compare` measures each other model against the reference by paired tests. Without `--reference`, `compare` gives each model's metrics, without tests. |
| `--benchmark` | one or more [benchmarks](benchmarks.md). Every model must have scores on each of them. |
| `--format` | the [text format](scoring.md#text-formats) of the scores, `compact` by default. |
| `--colors` | one color per `--model`, in the same order. [Colors](#colors) says how. |

`compare` prints each model's MRR and accuracy@1 on each benchmark. Then it
prints the comparison's folder, and the path of each figure's PNG and of each
table.

**Give each model folder its own name.** Two model folders of one name share
one scores folder, as [What `score` writes](scoring.md#what-score-writes)
says.

## What `compare` writes

`compare` writes to `embedding_model/outputs/comparisons/<name>/`.

```
comparisons/<name>/
    metrics.csv
    differences.csv           with --reference only
    figures/
        accuracy_and_mrr.eps   accuracy_and_mrr.pdf   accuracy_and_mrr.png
        accuracy_by_k.eps      accuracy_by_k.pdf      accuracy_by_k.png
        mrr_by_size.eps        mrr_by_size.pdf        mrr_by_size.png
        differences_from_reference.eps, .pdf, .png        with --reference only
    manifest.json
```

### `metrics.csv`

One row per model and benchmark, with these columns:

- `model`, `benchmark`, `benchmark_version` and `format`
- `n`: the number of source variables
- `targets`: the number of the benchmark's targets
- `parameters`: the model's number of parameters
- `mrr`, `mrr@10`, `accuracy@1`, `accuracy@3`, `accuracy@5` and
  `accuracy@10`, each with its interval in `<metric>_low` and
  `<metric>_high`
- `ndcg@10` and `median_rank`

Each interval is a 95% confidence interval.

### `differences.csv`

`compare` writes `differences.csv` only with `--reference`. It has one row
for each benchmark, each model other than the reference, and each tested
metric: `mrr`, `mrr@10`, `accuracy@1`, `accuracy@3`, `accuracy@5` and
`accuracy@10`. Its columns are:

- `benchmark`, `metric`, `reference` and `model`
- `difference`: the model minus the reference
- `low` and `high`: the difference's 95% confidence interval
- `test`: the name of the paired test
- `p`: the test's p-value

### `figures/`

- `accuracy_and_mrr`: a row per benchmark, with accuracy@1, @5 and @10 on the
  left and MRR on the right
- `accuracy_by_k`: a panel per benchmark, with accuracy at every k from 1 to
  50
- `mrr_by_size`: a panel per benchmark, with each model's MRR by its number
  of parameters
- `differences_from_reference`, with `--reference` only: a panel per
  benchmark, with each other model's difference from the reference on the
  metrics of `accuracy_and_mrr`

Each figure comes as a `.png`, a `.pdf` and an `.eps`.
[Reading the figures](#reading-the-figures) says how to read them.

### `manifest.json`

The comparison's manifest has the fields that
[manifests.md](manifests.md#its-fields) lists. In the comparison's manifest,
`sources` and `settings` hold:

- `sources`: each scores folder that `compare` read, with the SHA-256 of its
  `metrics.csv` and `per_variable.csv`
- `settings`: the text format and the confidence level. For each metric
  with a test, it also names the methods of the metric's interval, of the
  difference's interval and of the test.

It also has six fields of its own:

- `models`: the models, by their names in the tables
- `model_colors`: each model's color as drawn, an 8-digit hex code with its
  opacity, for example `#3f90daff`
- `reference`: the reference, or null without `--reference`
- `benchmarks`: each benchmark, with its version
- `ai_harmonization`: the commit of the code, and whether it had uncommitted
  changes
- `libraries`: the versions of Python and of the libraries that compared
  the models

## Details

### Reading the figures

scoring.md defines accuracy@k and MRR, in
[The metrics](scoring.md#the-metrics).

- **In `accuracy_and_mrr` and `mrr_by_size`, an error bar** is the model's
  95% confidence interval.
- **In `accuracy_and_mrr`, a bracket with stars** marks a significant
  difference between the model and the reference, on the same source
  variables. Its stars give the p-value: `*` p < 0.05, `**` p < 0.01, `***`
  p < 0.001. No bracket means no significant difference.
- **The error bars of a model and of the reference can overlap** while a
  bracket marks a difference.
- **In `differences_from_reference`,** a point is the model minus the
  reference, with its 95% confidence interval. A positive difference means
  that the model does better. An interval that crosses 0 means that the
  benchmark cannot tell the model from the reference.
- **In `mrr_by_size`, models of the same size** sit side by side. The small
  shift along the x axis is not a difference in size.

### Colors

By default, each model takes the color of its place in `--model`, from
matplotlib's `petroff10` sequence, and keeps it in every figure. So models
given in the same order keep their colors from one comparison to the next.

`--colors` gives each model's color instead, one per `--model`, in the same
order. Each color can be in any form that matplotlib reads, as
[matplotlib's guide to colors](https://matplotlib.org/3.11.1/users/explain/colors/colors.html#color-formats)
lists them, for example `'#3f90da'` or `'tab:purple'`. Quote a hex code,
because the shell reads `#` as the start of a comment.
