# Experiments

An experiment asks one question: does a change to a run's config help? Its
file gives the **base**, which is a run's config, and one or more **arms**.
Each arm is a version of the base that changes some settings.
`experiment train` trains the base and each arm at several seeds. So
`experiment compare` can tell a difference between the arms from the
difference between seeds. One arm at one seed is one [run](training.md), as
`train` trains it.

`experiment train` writes the runs, the arms' configs and the runs' logs to one
folder, the experiment's folder. `experiment compare` then compares each arm
with the base, and recommends an arm to build on. It writes its comparison to
the same folder. [What `experiment` writes](#what-experiment-writes) lists the
files. [Details](#details) describes:

- the experiment's file, in full
- the checks before training
- the GPUs, and clusters with SLURM
- how `experiment compare` compares the arms

Model development is a series of experiments. Each one builds on an arm of the
one before. You choose that arm.
[design_decisions.md](design_decisions.md#experiments) gives the reasons for
the rules below.

## Running `experiment`

```
python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml    # train the runs that are not done
python -m embedding_model experiment status --config embedding_model/configs/experiments/dim.yaml   # check which runs are done
python -m embedding_model experiment compare --config embedding_model/configs/experiments/dim.yaml  # compare the arms, once every run is done
```

### Writing an experiment's file

An experiment's file is a YAML file. The experiment takes its name from the
file, as a run takes its name from its config: `dim.yaml` is the experiment
`dim`. Keep your experiments' files in `embedding_model/configs/experiments/`,
apart from `inputs/` and `outputs/`, which the tools write. Git ignores all
three.

The file names the base, the seeds, and the arms, each with the settings that
it changes. This one varies `embedding_dim`, the length of each token's vector
in the table of a [static model](training.md#static-a-static-embedding-model).
It asks whether a smaller table loses
[validation MRR@10](training.md#manifestjson), or a larger one gains it:

```yaml
# dim.yaml: an experiment on the table's dimension, embedding_dim
question: Does a smaller table lose validation MRR@10, or a larger one gain it?
base:
  data:
    corpus: example
    tokenizer: example-8k
  training:
    dataloader_pin_memory: false
seeds: [0, 1, 2, 3, 4]
decision:
  scored_on: validation
  metric: mrr@10
  minimum_effect: 0.01
  alpha: 0.05
arms:
  dim128:
    test: non-inferiority
    model: {embedding_dim: 128}
  dim2048:
    test: superiority
    model: {embedding_dim: 2048}
```

Vary one setting per experiment, as `dim.yaml` varies `embedding_dim`. Then
each arm's difference from the base comes from that one setting.

[The experiment's file](#the-experiments-file) lists every field, and the two
tests that an arm can ask of the base.

### Training it

`experiment train` trains each run of the experiment that is not done. Each
run is the `train` command, `python -m embedding_model train`, with the arm's
config and the seed. Each run's output goes to the run's log, in `logs/`. A
run that fails does not stop the others. At the end, `experiment train` lists
the runs that failed, and exits with 1. If you train the same file again,
`experiment train` trains only the runs that are not done.

[Details](#details) says more:

- [What it checks before training](#what-it-checks-before-training): the file,
  the configs and the inputs
- [A changed file](#a-changed-file): after the first training, the
  `experiment` commands refuse a file whose fields changed
- [GPUs and CPUs](#gpus-and-cpus): one run per GPU at a time, or one at a time
  without a GPU
- [On a cluster with SLURM](#on-a-cluster-with-slurm): the whole experiment as
  one job

### Following it

For each arm, `experiment status` counts the runs that are done, started and
not started:

```
$ python -m embedding_model experiment status --config embedding_model/configs/experiments/dim.yaml
experiment dim: 11 of 15 runs done
  base     5 done
  dim128   3 done, 2 started
  dim2048  3 done, 2 not started
```

A run is done when its folder has its manifest. A run has started when a
staging folder sits beside its folder.
[Inputs and outputs](../README.md#inputs-and-outputs) describes staging
folders. A started run is training now, or it was killed before it finished,
for example by `scancel` or a time limit. A run that you stop with Ctrl-C
deletes its staging folder.

[design_decisions.md](design_decisions.md#an-experiments-status-comes-from-its-run-folders)
gives the reasons for these states.

### Comparing the arms

When every run is done, `experiment compare` compares each arm with the base,
by each run's score on the validation pairs of the [corpus](corpus.md). It
prints each arm's verdict, and recommends an arm to build on:

```
$ python -m embedding_model experiment compare --config embedding_model/configs/experiments/dim.yaml
experiment dim: validation MRR@10, the base and 2 arms at 5 seeds
  base     0.87200, 95% interval 0.86974 to 0.87426
  dim128   -0.00274, 97.5% interval -0.00678 to +0.00130, p 0.096: no more than 0.01 worse
  dim2048  +0.00178, 97.5% interval -0.00188 to +0.00544, p 0.21: not better by at least 0.01
  recommended: dim128, as no more than 0.01 worse
  To build on it, write in the next experiment's file:
    base: dim/dim128
  Its best run is dim128_s3, at validation MRR@10 0.8731. The comparison uses the validation pairs only. To score the run on the benchmarks: python -m embedding_model score --model /home/abcd/ai-harmonization/embedding_model/outputs/experiments/dim/runs/dim128_s3 --benchmark gdc topmed
  -> /home/abcd/ai-harmonization/embedding_model/outputs/experiments/dim/comparison
     figure: /home/abcd/ai-harmonization/embedding_model/outputs/experiments/dim/comparison/figures/arms_and_differences.png
     table: /home/abcd/ai-harmonization/embedding_model/outputs/experiments/dim/comparison/arms.csv
     table: /home/abcd/ai-harmonization/embedding_model/outputs/experiments/dim/comparison/differences.csv
```

The last lines give the figure and the tables by their full paths, as
[`compare`](comparing.md#running-compare) gives its own. In a terminal that
links paths, for example VS Code's, each path opens with a click.

To build on an arm, copy its `base:` line into the next experiment's file.
[Building on an earlier experiment](#building-on-an-earlier-experiment)
describes such a base. [How the arms are compared](#how-the-arms-are-compared)
explains each line and each verdict.
[The recommendation](#the-recommendation) says which arm `experiment compare`
recommends.

`experiment compare` stops, and says why, when:

- a run is not done. The message names the runs, and the command that trains
  them.
- a base run of an earlier experiment has started but not finished. The
  message says to wait for the run. If no job is training the run, the
  message says to delete its staging folder.
- the base names an earlier experiment that has not been trained in the
  outputs folder, or an arm that the earlier experiment does not have
- the file differs from the copy that its first training kept.
  [A changed file](#a-changed-file) says more.

## What `experiment` writes

`experiment train` writes to `embedding_model/outputs/experiments/<name>/`, or
under the folder that `--outputs` names. `train` writes each run whole, and
`experiment train` writes the arms' configs whole. A run that fails leaves no
folder. If you train the file again, `experiment train` trains the run again.
[Inputs and outputs](../README.md#inputs-and-outputs) describes how a step
writes a folder whole.

`experiment compare` writes `comparison/` in the same folder, whole too. So if
a comparison fails, the previous comparison stays as it was. A new comparison
replaces it.

```
experiments/<name>/
    experiment.yaml
    configs/
        base.yaml   <arm>.yaml ...
    runs/
        base_s0/   <arm>_s0/ ...
    logs/
        base_s0.log   <arm>_s0.log ...
    comparison/
        arms.csv
        differences.csv
        figures/
            arms_and_differences.eps   arms_and_differences.pdf   arms_and_differences.png
        manifest.json
```

### `experiment.yaml`

A copy of the experiment's file, from its first training.
[A changed file](#a-changed-file) says how the `experiment` commands use the
copy.

### `configs/`

Each arm's full config: the base's settings, with the arm's changes. The
base's config is `base.yaml`. `train` trains each run from its arm's config.
You can also give a config to `train` yourself, as a run's config. Then
`train` writes the run to `embedding_model/outputs/runs/dim128_s0/`, not into
the experiment's folder:

```
python -m embedding_model train --config embedding_model/outputs/experiments/dim/configs/dim128.yaml --seed 0
```

### `runs/`

Each run's folder, `<arm>_s<seed>/`, as `train` writes it.
[What `train` writes](training.md#what-train-writes) lists its files. A base
run that an [earlier experiment](#building-on-an-earlier-experiment) trained
stays in that experiment's folder.

### `logs/`

Each run's output, as `<arm>_s<seed>.log`: what `train` prints, and its
errors.

### `comparison/`

What `experiment compare` writes: two tables, a figure and a manifest.

#### `arms.csv`

One row per arm, the base's first:

| column | what it holds |
|---|---|
| `arm`, `test`, `metric` | the arm, its test (empty for the base), and the metric that decides |
| `seeds` | the number of seeds |
| `mean`, `sd` | the mean of the seeds' scores, and their spread from seed to seed |
| `low`, `high` | the mean's 95% interval |
| `s<seed>`, for example `s0` | each seed's score |

#### `differences.csv`

One row per arm:

| column | what it holds |
|---|---|
| `arm`, `test`, `metric` | the arm, its test, and the metric that decides |
| `difference` | the arm's mean minus the base's |
| `low`, `high` | Welch's interval of the difference |
| `confidence` | the interval's level, for example 0.975 with 2 arms |
| `df` | Welch's degrees of freedom |
| `p` | the p of Welch's t-test of no difference |
| `alpha` | the arm's alpha: the decision's, divided by the number of arms |
| `minimum_effect` | the decision's minimum effect |
| `passes` | whether the arm passes its test |
| `seeds_needed` | the seeds per arm that its test needs, by Welch's t-test and its power, as [How the arms are compared](#how-the-arms-are-compared) says |
| `underpowered` | whether the arm did not pass its test, with fewer seeds than the test needs, while its interval still reaches the margin that `minimum_effect` sets |
| `verdict` | the verdict, as printed |

#### `figures/`

`arms_and_differences`: each arm's seeds on the left, and each arm minus the
base on the right. [What it shows](#what-it-shows) says how to read it. The
figure comes in three files, as the [figures of `compare`](comparing.md#figures)
do: a `.png`, at 300 dpi, and a `.pdf` and an `.eps`.

#### `manifest.json`

The comparison's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `experiment`: the experiment's name
- `built`: the date of the comparison
- `sources`: the experiment's `experiment.yaml`, and each run's
  `manifest.json`, each with its SHA-256
- `settings`: the decision's fields, the arm's alpha, the intervals, the
  test, and the power that `seeds_needed` aims at, with the method that
  computes the power

It also has four fields of its own:

- `seeds`: the experiment's seeds
- `recommended`: the recommended arm, the base, or null when several arms
  pass
- `ai_harmonization`: the commit of the code, and whether it had uncommitted
  changes
- `libraries`: the versions of Python and of the libraries that compared the
  arms

## Details

### The experiment's file

| field | needed | what it holds |
|---|---|---|
| `question` | no | one sentence: what the experiment asks |
| `base` | yes | the base's settings, by the sections of a run's [config](training.md#the-config). Or an arm of an earlier experiment, as `<experiment>/<arm>`: see [Building on an earlier experiment](#building-on-an-earlier-experiment) |
| `seeds` | yes | the seeds, as a list of at least 2 different whole numbers, from 0 to 4,294,967,295. One seed has no seed-to-seed spread, so `experiment compare` could not compare the arms |
| `decision` | no | the rule that decides between the arms: see the next table |
| `arms` | yes | each arm, by its name: its `test`, and the settings that it changes, by section. An arm is not named `base`, and does not change `data.corpus` |

Neither the base nor an arm sets `training.seed`: each run's seed comes from
`seeds`.

The `decision` block has a default for each of its fields:

| field | what it holds | default |
|---|---|---|
| `scored_on` | the data whose scores decide: `validation`, the corpus's validation pairs, the only choice for now | `validation` |
| `metric` | the validation metric that decides: `mrr@10`, `accuracy@1`, `accuracy@3`, `accuracy@5` or `accuracy@10`, as each run's manifest records it | `mrr@10` |
| `minimum_effect` | the margin, on the metric's scale. For `superiority`, the smallest gain worth a change. For `non-inferiority`, the largest loss worth taking | `0.01` |
| `alpha` | the chance, over the whole experiment, that an arm passes by chance alone. `experiment compare` splits it over the arms | `0.05` |

Each arm names its `test`: the statistical test, the question that it asks of
the base:

- `superiority`: is the arm better than the base, by at least
  `minimum_effect`? Use it for an arm that costs the same as the base, or more,
  for example a larger table.
- `non-inferiority`: is the arm no worse than the base, by more than
  `minimum_effect`? Use it for an arm that is cheaper to train or to run, for
  example a smaller table.

An arm changes only the settings that it names. Each other setting is the
base's.

An arm cannot change `data.corpus`: the arms would then be scored on different
validation pairs.

[design_decisions.md](design_decisions.md#one-experiment-asks-one-question-in-one-file)
gives the reasons for one file per question, and for
[its checks](design_decisions.md#an-experiments-file-is-checked-with-pydantic).

### Building on an earlier experiment

An experiment's `base` can be an arm of an earlier experiment, by name:

```yaml
question: Does a learning rate other than the default help the smaller table?
base: dim/dim128
seeds: [0, 1, 2, 3, 4]
arms:
  lr0.1:
    test: superiority
    training: {learning_rate: 0.1}
  lr0.4:
    test: superiority
    training: {learning_rate: 0.4}
```

The base is then that arm's full config, from the earlier experiment's
`configs/`. The new experiment reuses the arm's runs at the same seeds, and
does not train them again. At a seed where the earlier experiment has no run,
`experiment train` trains the base here, as `base_s<seed>`. If the earlier
experiment is still training one of the arm's runs, `experiment train` stops,
so that two experiments do not train the run at once. But if the earlier
experiment has not started the arm's run at a seed yet, `experiment train`
trains the run here too. So train a later experiment once the earlier one is
done.

`base: dim/base` builds on the earlier experiment's own base. The new
experiment asks a new question of the same config, and reuses the base's runs.
When `dim`'s own base is an arm of another experiment, for example
`base: size/dim512`, the new experiment reuses the runs of that arm too.

The earlier experiment's folder must stay, because the new experiment reads
the earlier experiment's runs. To delete the folder, first copy the arm's config, for example
`embedding_model/outputs/experiments/dim/configs/dim128.yaml`. A later
experiment can give those settings in full, as its `base`. Its base runs then
train again.

[design_decisions.md](design_decisions.md#the-base-can-be-an-arm-of-an-earlier-experiment-whose-runs-are-reused)
gives the reasons.

### What it checks before training

Before any run trains, `experiment train` checks that:

- the file's fields can be used
- each arm's config is one that `train` reads, so a setting that does not
  exist stops the experiment
- the corpus and each tokenizer are built. If one is not, the message gives
  the command that builds the corpus or trains the tokenizer.
- no base run of an earlier experiment is still training

### A changed file

You write the experiment's file, for example
`embedding_model/configs/experiments/dim.yaml`. At the first training,
`experiment train` keeps a copy of the file in the experiment's folder,
`embedding_model/outputs/experiments/dim/`, as `experiment.yaml`. After that,
each `experiment` command refuses a `dim.yaml` whose fields differ from the
copy, for example one with another seed or another arm. A comment, another
order of the fields, or a default written out is no change. The message names
the fields that differ, and gives two next steps:

- to train the changed file from the start, delete the folder,
  `embedding_model/outputs/experiments/dim/`. This deletes the experiment's
  runs too.
- to keep the runs, and ask a new question, save the changed file under a new
  name, for example `dim-small.yaml`. Its base can reuse the earlier runs, as
  `base: dim/base`.

[design_decisions.md](design_decisions.md#a-changed-experiment-file-is-refused-rather-than-trained-again)
gives the reasons.

### GPUs and CPUs

The runs train on the machine where `experiment train` runs:

- with GPUs, one run per GPU at a time. Each run sees one GPU of its own,
  through `CUDA_VISIBLE_DEVICES`. When a run ends, the next run starts on the
  same GPU.
- without a CUDA GPU, one run at a time. `train` picks the device, as
  [training.md](training.md#gpus-and-cpus) says.

`CUDA_VISIBLE_DEVICES` chooses the GPUs on a machine that has several:

```
CUDA_VISIBLE_DEVICES=1,2 python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml
```

[design_decisions.md](design_decisions.md#an-experiment-queues-its-runs-itself-rather-than-submitting-each-as-a-job)
gives the reasons.

### On a cluster with SLURM

From the repository root, submit the experiment as one
[SLURM](https://slurm.schedmd.com/overview.html) job, which asks for the
GPUs that its runs share:

```
sbatch --nodes=1 --gpus=4 --cpus-per-task=20 --mem=128G --time=04:00:00 \
    --wrap "python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml"
```

- `--gpus=4`: four runs train at a time, one on each GPU. Always ask for the
  GPUs with `--gpus`, because a job that asks for none gets no GPU from SLURM.
  Some clusters do not hide the node's GPUs from such a job. Its runs would
  then use GPUs that another job may hold.
- `--cpus-per-task=20`: about 5 CPUs for each run that trains at a time. Each
  run reads its pairs with 4 worker processes.
- `--mem`: enough memory for the runs that train at a time.
- `--time`: a limit. It only stops a job that takes too long. The job ends,
  and frees its GPUs, when its last run ends.
- `--nodes=1`: the runs train on the GPUs of the node where the job starts.

Your cluster may need other options too, for example a partition, `-p`. The
cluster's own guide says which.
[SLURM's sbatch page](https://slurm.schedmd.com/sbatch.html) lists every
option. These SLURM commands show the cluster and your jobs, and stop a job:

- `sinfo -N -p <partition> -O "NodeList,Gres,GresUsed"` shows each node's
  GPUs, and how many are in use
- `squeue --me` shows your jobs
- `scancel <job id>` stops a job

The job keeps its GPUs until its last run ends. While the last runs train, the
GPUs of the runs that are done wait. The runs that are done stay. So if the
job stops, for example at its time limit, it loses only the runs that were
training. If you submit the job again, it trains those runs. Train an
experiment in one job at a time, because a second job would train again the
runs that the first job is training.

### How the arms are compared

Each run gives one number: its score on the validation pairs, by the
decision's `metric`, from the run's manifest. An arm's score is the mean over
its seeds, with its 95% t interval.

In the printout, the base's line gives its mean, with its interval. Each
arm's line gives its mean minus the base's, and the interval of that
difference. Then the line gives the p of a test of no difference, and the
verdict of the arm's test.

Each arm minus the base gets [Welch's interval and Welch's t-test](design_decisions.md#arms-are-compared-by-their-seeds-scores-with-welchs-interval-and-t-test),
unpaired. The interval excludes 0 exactly when the test's p is below the arm's
alpha. When neither arm's scores vary, for example when every run scores 1,
the difference is exact. The interval is then the difference itself. Its p is
1 when the difference is 0, and 0 otherwise.

The decision's `alpha` is [split over the arms](design_decisions.md#alpha-is-split-over-the-arms-by-bonferroni).
With 2 arms, `experiment compare` tests each arm at 0.05 / 2 = 0.025, and the
arm's interval is a 97.5% interval.

- The [`superiority` test passes](design_decisions.md#the-superiority-test-passes-on-a-significant-gain-of-at-least-the-minimum-effect)
  when both hold: the p is below the arm's alpha, and the arm's mean is at
  least `minimum_effect` above the base's. With the default `minimum_effect`
  of 0.01, a significant gain of 0.005 does not pass the test.
- The [`non-inferiority` test passes](design_decisions.md#the-non-inferiority-test-passes-when-the-interval-stays-above-minus-the-minimum-effect)
  when the interval's low end is above minus `minimum_effect`, for example
  above -0.01.

| test | when it passes | when it does not, and the interval still reaches the margin | when it does not, and the interval lies wholly past the margin |
|---|---|---|---|
| `superiority` | `better by at least 0.01` | `not shown better by at least 0.01` | `not better by at least 0.01` |
| `non-inferiority` | `no more than 0.01 worse` | `may be more than 0.01 worse` | `more than 0.01 worse` |

A verdict in the middle column says too little, either way. A verdict in the
last column settles the question: more seeds would most likely not make the
arm pass.

Each arm's row also gives [`seeds_needed`](design_decisions.md#an-arm-that-does-not-pass-its-test-is-checked-for-enough-seeds-at-the-minimum-effect), the seeds per arm
that its test needs, at the two arms' seed-to-seed spreads:

- for `superiority`, the fewest seeds at which a mean gain of exactly
  `minimum_effect` would be significant, by Welch's t-test
- for `non-inferiority`, the fewest seeds that give an 80% chance to pass,
  when the arm and the base do not differ. This is the power of Welch's
  t-test, from the noncentral t distribution, scipy's `stats.nct`.

scipy's `optimize.elementwise` finds both counts. `bracket_root` finds a count
that is too few and a count that is enough. Then `find_root` finds the exact
count between them, and `experiment compare` rounds that count up.

`experiment compare` also marks an arm `underpowered: <k> seeds needed` when
all of these hold:

- the arm does not pass its test
- the arm's interval still reaches the margin
- the arm had fewer seeds than its test needs

To settle the arm's question, ask it again in a new experiment, with the same
number of arms, at k seeds or more. Build the new experiment on the same base.
Then the new experiment reuses the base's runs at the seeds already trained.
[Building on an earlier experiment](#building-on-an-earlier-experiment) says
how.

When `experiment compare` gives the verdict, it reads the difference and the
interval as `differences.csv` shows them, to 5 decimals. So a gain shown as
0.01 is a gain of at least 0.01.

### The recommendation

`experiment compare` [recommends an arm](design_decisions.md#the-comparison-recommends-one-arm-or-none-when-several-pass), and you choose:

- one arm passes its test: `experiment compare` recommends that arm
- no arm passes: it recommends the base
- several arms pass: it recommends none, and prints the `base:` line of each
  arm that passes

When `experiment compare` recommends the base, the base's line names this
experiment's base, for example `base: dim/base`. The next experiment then
reuses the base's runs, wherever they were trained.

The comparison uses the validation pairs only, never the benchmarks. So
`experiment compare` prints the recommended arm's best run, with the command
that scores the run on the benchmarks. When the run's corpus was checked
against no benchmark, the line says what to do instead. It says to build the
benchmarks, then build the corpus again, then train again. The command names the benchmarks that the run's corpus was
checked against for [leakage](leakage.md), because `score` refuses any other
benchmark.
[Scoring a run](training.md#scoring-a-run) says how `score` checks them. Then
[`compare`](comparing.md) compares the run with other models.

### What it shows

The figure, `arms_and_differences`, has two panels. Each row is one arm, the
base's first, and reads across both panels.

| in the figure or table | what it shows | how it is computed |
|---|---|---|
| the title | the experiment, the metric, and the number of seeds | from the experiment's file |
| a row | one arm, the base's first, named on the left | |
| a dot, in the left panel, just above its row | one seed's score | the run's validation score, from its manifest |
| a bar with an error bar, in the left panel, on its row | the arm's mean, with its 95% interval | the t interval of the mean of the seeds' scores, from scipy's `ttest_1samp` |
| the gray band, in the left panel | the base's mean, minus and plus `minimum_effect` | from the base's mean and the decision |
| a diamond with an error bar, in the right panel | the arm minus the base, with Welch's interval at the confidence of the arm's test. That is 95%, or the Bonferroni level with several arms, as the axis says. A filled diamond: the arm passes its test. A hollow diamond: it does not | scipy's `ttest_ind`, with `equal_var=False`: the rows of `differences.csv` |
| the solid line at 0, in the right panel | no difference from the base | |
| the gray band and its dashed edges, in the right panel | minus to plus `minimum_effect` | from the decision |
| a color | blue for the base, dark orange for a non-inferiority arm, purple for a superiority arm | three colors of matplotlib's `petroff10`, which stay apart under each color vision deficiency |
| the legend | the colors, as swatches, then the right panel's filled and hollow diamonds, and the bands | |
| `arms.csv` | each arm's mean, sd and interval, and each seed's score | as the left panel |
| `differences.csv` | each arm minus the base: the difference with its interval, the p of a test of no difference, the verdict, and the seeds that the arm's test needs | Welch's interval and t-test: scipy's `ttest_ind`, with `equal_var=False`. The seeds needed: by Welch's t-test, and for non-inferiority by its power, from scipy's noncentral t, `stats.nct`. Each count: scipy's `optimize.elementwise`, `bracket_root` then `find_root` |

In the right panel, 0 is at the center. So the side where an arm is worse and
the side where it is better have the same length. A `non-inferiority` arm
passes when its interval stays right of the band's left edge. A `superiority`
arm passes when its interval stays right of 0, and its diamond is on or right
of the band's right edge.

The left panel's band only helps you read the scores. The verdicts come from
the right panel's intervals. The printout and `differences.csv` give each
verdict in words.
