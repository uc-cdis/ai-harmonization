# Comparing models

`python -m embedding_model compare` compares models that are already scored.
It reads the scores that [`score`](scoring.md) wrote for each model, on one
or more benchmarks. A benchmark lists source variables, each with its
expected targets: the target properties that experts mapped it to. A model's
scores on a benchmark hold the [metrics](scoring.md#the-metrics), and each
source variable's [rank](scoring.md#how-a-model-is-scored): the place of the
variable's best-placed expected target.

`compare` writes a table of metrics and three figures. With `--reference`,
you name one of the models as the reference. Then `compare` also writes each
other model's difference from the reference, and a fourth figure.
[What `compare` writes](#what-compare-writes) lists the files.
[Details](#details) explains what each figure and table shows, and how it is
computed.

[design_decisions.md](design_decisions.md#comparing) gives the reasons for
the choices below.

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
| `--name` | the comparison's name, and its folder: `outputs/comparisons/<name>/`. A second comparison of the same name replaces the first. |
| `--model` | one or more models, up to 8: a Hugging Face model id, a model folder on disk, or the folder of a run that [`train`](training.md) wrote. |
| `--reference` | one of the models: the reference. `compare` measures each other model against the reference by paired tests. Without `--reference`, `compare` gives each model's metrics, without tests. |
| `--benchmark` | one or more [benchmarks](benchmarks.md). Every model must have scores on each of them. If you name a benchmark twice, `compare` compares it once. |
| `--format` | the [text format](scoring.md#text-formats) of the scores, `compact` by default. |
| `--colors` | one color per `--model`, in the same order, in any form that matplotlib reads, for example `'#3f90da'` or `'tab:blue'`. Without `--colors`, each model takes a color of the default `petroff10` palette, by its place in `--model`, as [Colors](#colors) says. |

`compare` prints each model's MRR and accuracy@1 on each benchmark. Then it
prints the comparison's folder, after `->`, and the full path of each
figure's PNG and of each table, one per line. In a terminal that links
paths, for example VS Code's, each path opens with a click.

`compare` stops, and names the `score` command to run, when a model:

- has no scores on a benchmark in that format
- has a scores folder that holds another model's scores. Two model folders
  of one name share one scores folder, so the folder holds the scores of the
  one scored last.
- has scores from another build of the benchmark than the current build.
  One of the benchmark's files no longer has the checksum that `score`
  recorded.

It also stops when:

- `--name` is not a folder name
- a benchmark is not built. The message names the command that builds it.
- `--model` names more than 8 models
- `--reference` is not one of the models, or `--model` gives no other model
- `--model` names one model twice, for example by its full path and by a
  relative path
- two models have one name, for example two model folders named
  `abcd-model`
- two models' scores list other source variables, or list them in another
  order. Then the paired tests cannot match their variables.
- `--colors` gives more or fewer colors than `--model` gives models, or one
  color to two models
- `--colors` gives a color that matplotlib does not read, for example
  `tab:bleu`, or a fully transparent one, for example `none`

## What `compare` writes

`compare` writes to `embedding_model/outputs/comparisons/<name>/`, or under
the folder that `--outputs` names. It writes the folder whole. So if a
comparison fails, the previous comparison of that name stays as it was.
[Inputs and outputs](../README.md#inputs-and-outputs) describes how a step
writes a folder whole.

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

### `differences.csv`

`compare` writes `differences.csv` only with `--reference`. It has one row
per benchmark, other model and metric with a test: `mrr`, `mrr@10`,
`accuracy@1`, `accuracy@3`, `accuracy@5` and `accuracy@10`. Its columns are:

- `benchmark`, `metric`, `reference` and `model`
- `difference`: the mean of the source variables' differences, the model
  minus the reference
- `low` and `high`: the difference's interval
- `test`: the name of the paired test
- `p`: the test's p

### `figures/`

- `accuracy_and_mrr`: a row per benchmark, with accuracy@1, @5 and @10 on the
  left and MRR on the right
- `accuracy_by_k`: a panel per benchmark, with accuracy at every k from 1 to
  50
- `mrr_by_size`: a panel per benchmark, with each model's MRR by its number
  of parameters
- `differences_from_reference`, with `--reference` only: a panel per
  benchmark, with each other model's difference from the reference on the
  metrics of `accuracy_and_mrr`. The panels share one x axis.

Each figure is 7.2 inches wide, and comes in three files: a `.png`, at 300
dpi, and a `.pdf` and an `.eps`. The `.pdf` and the `.eps` are vector images,
so they stay sharp at any size.

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

### What it shows

| in the figure or table | what it shows | how it is computed |
|---|---|---|
| a benchmark's title | the benchmark, its version, its number of source variables and its number of targets. The source variables set how precise each number is. The targets set how many candidates a model ranks for each variable | the version from the benchmark's current build. The source variables from the scores' `per_variable.csv`, and the targets from their `metrics.csv` |
| a model's name, in the legends and the tables | which model it is | for a run, the run's name, for example `run_example_s0`. For a model folder, the folder's name. For a Hugging Face model, its id |
| the first bar of each group, in `accuracy_and_mrr`, with `--reference` | the reference, marked "(reference)" in the legends | |
| a bar of accuracy@k, in `accuracy_and_mrr` | the share of source variables whose expected target is among the model's first k suggestions | the accuracy@k that `score` reports, in the scores' `metrics.csv` |
| a bar of MRR, in `accuracy_and_mrr` | how high the best-placed expected target ranks, on average: 1 if first, 1/2 if second, 1/3 if third | the MRR that `score` reports, in the scores' `metrics.csv` |
| an error bar, in `accuracy_and_mrr` and `mrr_by_size` | how precise that model's number is: its 95% confidence interval | accuracy@k: the Wilson score interval, from scipy's `binomtest`. MRR: the t interval of the mean, from scipy's `ttest_1samp`, cut at 0 and 1 |
| a bracket with stars, in `accuracy_and_mrr` | that model and the reference differ, on the same source variables: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001, as the note under the legend says. No bracket means no significant difference: p ≥ 0.05 | accuracy@k: McNemar's exact test, a binomial test on the variables where the two models differ. MRR: the paired t-test, scipy's `ttest_rel` |
| a line in `accuracy_by_k` | the share of source variables whose expected target is among the first k suggestions, for every k from 1 to 50 | from each source variable's rank, in the scores' `per_variable.csv` |
| a point in `mrr_by_size` | the model's MRR, with its 95% confidence interval, by the model's number of parameters, on a log scale. Models of the same size sit side by side | the MRR that `score` reports, with its t interval. The parameters from the scores' `metrics.csv` |
| a point in `differences_from_reference` | that model minus the reference, with its 95% confidence interval. A positive difference means that the model does better than the reference, and a negative one that it does worse | the rows of `differences.csv` |
| `metrics.csv` | each model's metrics on each benchmark. MRR, MRR@10 and accuracy@k have a 95% confidence interval | each metric as `score` reports it. accuracy@k has the Wilson interval. MRR and MRR@10 have the t interval, cut at 0 and 1 |
| `differences.csv` | each other model minus the reference: the difference, its interval and p | accuracy@k: Newcombe's interval for paired shares, and McNemar's exact test. MRR and MRR@10: the t interval of the source variables' differences, and the paired t-test |

#### Where the numbers come from

Every metric but the median rank is a fraction from 0 to 1, as `score`
reports it.

Each bar of `accuracy_and_mrr` is the number that `score` reports: the
`InformationRetrievalEvaluator` of [sentence-transformers](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#informationretrievalevaluator)
computes it. The intervals and the tests need each source variable's own
value, not only the mean. So `compare` takes each value from the variable's
rank, which `score` keeps in `per_variable.csv`: 1 / rank for MRR, and 1 or 0
for accuracy@k.

The scores' `metrics.csv` holds accuracy@k only for k of 1, 3, 5 and 10. So
each line of `accuracy_by_k` comes from the `per_variable.csv` ranks too. At
each k from 1 to 50, it is the share of source variables whose rank is k or
less. At k of 1, 3, 5 and 10, it is the accuracy@k that `score` reports.

A model's size, in `mrr_by_size`, is its number of parameters, as `score`
counts them on the loaded model. A static model's parameters are almost all
its embedding table, which costs little to run. So a static model can run
much faster than a transformer with the same number of parameters.

nDCG@10 and the median rank are in `metrics.csv` as plain numbers, as `score`
reports them. They have no interval and no test, and the figures do not show
them. See [scoring.md](scoring.md#the-metrics) for each metric.

#### Intervals and tests

In `accuracy_and_mrr`, two error bars can overlap while a bracket still marks
a difference. The design entries for
[accuracy@k](design_decisions.md#accuracyk-wilson-intervals-newcombes-paired-interval-and-mcnemars-exact-test)
and for
[MRR](design_decisions.md#mrr-and-mrr10-the-t-interval-and-the-paired-t-test)
say why.

`differences_from_reference` shows a small difference at its own scale. Its
panels share one x axis, so their 0s line up, and a difference has the same
length in each benchmark's panel. An interval that crosses 0 means that the
benchmark cannot tell the model from the reference.

For MRR, a difference's interval and its p always agree: the interval
excludes 0 exactly when p is below 0.05. For accuracy@k, the two can disagree
near 0: the interval can just miss 0 while p is just above 0.05, or the
reverse. The design entry for
[accuracy@k](design_decisions.md#accuracyk-wilson-intervals-newcombes-paired-interval-and-mcnemars-exact-test)
says why.

#### Colors

By default, each model takes the color of its place in `--model`, and
keeps it in every figure. So models given in the same order keep their
colors from one comparison to the next. `compare` draws the reference first,
whatever its place. The colors are matplotlib's `petroff10` sequence.
[design_decisions.md](design_decisions.md#colors-from-matplotlibs-petroff10-not-its-default-tab10)
says why.

`--colors` gives each model's color instead, one per `--model`, in the same
order. So a model can keep one color through different comparisons. In
this comparison, the run is blue and the released model purple, whatever
colors their places would give them:

```
python -m embedding_model compare --name example \
    --model embedding_model/outputs/runs/run_example_s0 uc-ctds/bge-large-en-v1.5-bio-mapping \
    --colors '#3f90da' 'tab:purple' \
    --reference embedding_model/outputs/runs/run_example_s0 \
    --benchmark topmed
```

Each color can be in any form that matplotlib reads, as
[matplotlib's guide to colors](https://matplotlib.org/3.11.1/users/explain/colors/colors.html#color-formats)
lists them. Two common forms are a hex code, for example `'#3f90da'`, and a
name, for example `'tab:purple'`. Quote a hex code, because the shell reads
`#` as the start of a comment.
[design_decisions.md](design_decisions.md#a-models-color-can-be-given-with---colors-one-per-model)
says why the colors come as a list.

Each model also has its own marker, in `accuracy_by_k`, `mrr_by_size` and
`differences_from_reference`, by its place in `--model`. The lines of
`accuracy_by_k` also take one of four line styles.
