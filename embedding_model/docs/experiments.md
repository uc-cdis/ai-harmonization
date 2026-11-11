# Experiments

An experiment asks one question: does a change to a run's config help? It
trains the config, the **base**, and one or more **arms**, each a version of
the base that changes some settings. Each is trained at several seeds, so that
a difference between the arms can be told apart from the difference between
seeds. One arm at one seed is one [run](training.md), as `train` trains it.

`experiment train` writes the runs, the arms' configs and the runs' logs to one
folder, the experiment's folder. `experiment compare` then compares each arm
with the base, and recommends an arm to build on. It writes its comparison to
the same folder. [What `experiment` writes](#what-experiment-writes) lists the
files. [Details](#details) describes the experiment's file in full, the
checks before training, the GPUs and clusters, and how the arms are compared.

Model development is a series of experiments. Each one builds on an arm of the
one before, which the person running them selects.
[design_decisions.md](design_decisions.md#experiments) gives the reasons for
the rules below.

## Running `experiment`

```
python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml    # train the runs that are not done
python -m embedding_model experiment status --config embedding_model/configs/experiments/dim.yaml   # check which runs are done
python -m embedding_model experiment compare --config embedding_model/configs/experiments/dim.yaml  # compare the arms, once every run is done
```

### Writing an experiment's file

An experiment's file is a YAML file. The experiment is named after it, as a run
is named after its config: `dim.yaml` is the experiment `dim`. Keep your
experiments' files in `embedding_model/configs/experiments/`, apart from
`inputs/` and `outputs/`, which the tools write. Git ignores all three.

The file names the base, the seeds, and the arms, each with the settings that
it changes. This one varies the table's dimension, `embedding_dim`: the length
of each token's vector. It asks whether a smaller table loses validation
MRR@10, or a larger one gains it:

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
run is the command that trains it on its own, `python -m embedding_model
train`, with the arm's config and the seed. Its output goes to its log, in
`logs/`. A run that fails does not stop the others. At the end,
`experiment train` lists the runs that failed, and exits with 1. Training the
same file again trains only the runs that are not done.

Details says more:

- [What it checks before training](#what-it-checks-before-training): the file,
  the configs and the inputs;
- [A changed file](#a-changed-file): once trained, a file whose fields change
  is refused;
- [GPUs and CPUs](#gpus-and-cpus): one run per GPU at a time, or one at a time
  without a GPU;
- [On a cluster with SLURM](#on-a-cluster-with-slurm): the whole experiment as
  one job.

### Following it

`experiment status` says which runs are done, which have started, and which
have not started, for each arm:

```
$ python -m embedding_model experiment status --config embedding_model/configs/experiments/dim.yaml
experiment dim: 11 of 15 runs done
  base     5 done
  dim128   3 done, 2 started
  dim2048  3 done, 2 not started
```

A run is done when its folder has its manifest. It has started when a staging
folder sits beside its folder: it is training now, or it was stopped before it
finished.

[design_decisions.md](design_decisions.md#an-experiments-status-comes-from-its-run-folders)
gives the reasons for these states.

### Comparing the arms

When every run is done, `experiment compare` compares each arm with the base,
by each run's score on the validation pairs. It prints each arm's verdict,
and recommends an arm to build on:

```
$ python -m embedding_model experiment compare --config embedding_model/configs/experiments/dim.yaml
experiment dim: validation MRR@10, the base and 2 arms at 5 seeds
  base     0.87198, 95% interval 0.86491 to 0.87905
  dim128   -0.00412, 97.5% interval -0.00845 to +0.00021, p 0.059: no more than 0.01 worse
  dim2048  +0.00178, 97.5% interval -0.00676 to +0.01032, p 0.56: not shown better by at least 0.01
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
links paths, such as VS Code's, each opens with a click.

To build on an arm, copy its `base:` line into the next experiment's file;
see [Building on an earlier experiment](#building-on-an-earlier-experiment).
[How the arms are compared](#how-the-arms-are-compared) explains each line and
verdict, and [The recommendation](#the-recommendation) which arm is
recommended.

`experiment compare` stops, and says why, when:

- a run is not done. The message names the runs, and the command that trains
  them;
- the experiment has one seed, so its seed-to-seed spread is unknown;
- the file differs from the copy that its first training kept; see
  [A changed file](#a-changed-file).

## What `experiment` writes

`experiment train` writes to `embedding_model/outputs/experiments/<name>/`, or
under the folder that `--outputs` names. It writes each run whole, as `train`
does, and the arms' configs whole. A run that fails leaves no folder, and the
next training of the file trains it again; see
[Inputs and outputs](../README.md#inputs-and-outputs). `experiment compare`
writes `comparison/` in the same folder, whole too. So a comparison that fails
leaves the previous one as it was, and comparing again replaces it.

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

A copy of the experiment's file, as it was first trained.
[A changed file](#a-changed-file) says what it is compared with.

### `configs/`

Each arm's full config, the base's as `base.yaml`: the base's settings, with
the arm's changes. `train` trains each run from its arm's config. A config is
also a run's config that `train` takes on its own:

```
python -m embedding_model train --config embedding_model/outputs/experiments/dim/configs/dim128.yaml --seed 0
```

### `runs/`

Each run's folder, `<arm>_s<seed>/`, as `train` writes it; see
[What `train` writes](training.md#what-train-writes). A base run that an
[earlier experiment](#building-on-an-earlier-experiment) trained stays in that
experiment's folder.

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
| `s<seed>`, such as `s0` | each seed's score |

#### `differences.csv`

One row per arm:

| column | what it holds |
|---|---|
| `arm`, `test`, `metric` | the arm, its test, and the metric that decides |
| `difference` | the arm's mean minus the base's |
| `low`, `high` | Welch's interval of the difference |
| `confidence` | the interval's level, such as 0.975 with 2 arms |
| `df` | Welch's degrees of freedom |
| `p` | Welch's t-test of no difference |
| `alpha` | the arm's alpha: the decision's, divided by the number of arms |
| `minimum_effect` | the decision's minimum effect |
| `passes` | whether the arm passes its test |
| `seeds_needed` | the seeds per arm that its test needs, by Welch's t-test and its power; see [How the arms are compared](#how-the-arms-are-compared) |
| `underpowered` | whether the arm did not pass its test, with fewer seeds than the test needs, while its interval still reaches the margin |
| `verdict` | the verdict, as printed |

#### `figures/`

`arms_and_differences`: each arm's seeds on the left, and each arm minus the
base on the right. [What it shows](#what-it-shows) says how to read it. It
comes in three files, as the [figures of `compare`](comparing.md#figures)
do: a `.png`, at 300 dpi, and a `.pdf` and an `.eps`.

#### `manifest.json`

The comparison's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `sources`: the experiment's `experiment.yaml`, and each run's
  `manifest.json`, each with its SHA-256;
- `settings`: the decision's fields, the arm's alpha, the intervals, the
  test, and the power that `seeds_needed` aims at, with how it is computed.

It also has four fields of its own:

- `seeds`: the experiment's seeds;
- `recommended`: the recommended arm, the base, or null when several arms
  pass;
- `ai_harmonization`: the commit of the code, and whether it had uncommitted
  changes;
- `libraries`: the versions of Python and of the libraries that compared the
  arms.

## Details

### The experiment's file

| field | needed | what it holds |
|---|---|---|
| `question` | no | one sentence: what the experiment asks |
| `base` | yes | the base's settings, by the sections of a run's [config](training.md#the-config). Or an arm of an earlier experiment, as `<experiment>/<arm>`: see [Building on an earlier experiment](#building-on-an-earlier-experiment) |
| `seeds` | yes | the seeds, as a list of different whole numbers, 0 or more |
| `decision` | no | the rule that decides between the arms: see the next table |
| `arms` | yes | each arm, by its name: its `test`, and the settings that it changes, by section |

The `decision` block has a default for each of its fields:

| field | what it holds | default |
|---|---|---|
| `scored_on` | the data whose scores decide: `validation`, the corpus's validation pairs, the only choice for now | `validation` |
| `metric` | the validation metric that decides: `mrr@10`, `accuracy@1`, `accuracy@3`, `accuracy@5` or `accuracy@10`, as each run's manifest records it | `mrr@10` |
| `minimum_effect` | the smallest difference worth deciding, on the metric's scale | `0.01` |
| `alpha` | the share of false findings allowed, split over the arms | `0.05` |

Each arm names its `test`: the statistical test, the question that it asks of
the base:

- `superiority`: is the arm better than the base, by at least
  `minimum_effect`? Use it for an arm that costs the same as the base, or more,
  such as a larger table.
- `non-inferiority`: is the arm no worse than the base, by more than
  `minimum_effect`? Use it for an arm that is cheaper to train or to run, such
  as a smaller table.

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
`configs/`. The arm's runs at the same seeds are reused, not trained again. A
seed that the earlier experiment has not trained is trained here, as
`base_s<seed>`. A base run that the earlier experiment is still training stops
`experiment train`, so that no run is trained twice.

`base: dim/base` builds on the earlier experiment's own base. It asks a new
question of the same config, and reuses its runs.

The earlier experiment's folder must stay, since the new one reads its runs.
To delete it, first copy the arm's config, such as
`embedding_model/outputs/experiments/dim/configs/dim128.yaml`. A later
experiment can give those settings in full, as its `base`. Its base runs then
train again.

[design_decisions.md](design_decisions.md#the-base-can-be-an-arm-of-an-earlier-experiment-whose-runs-are-reused)
gives the reasons.

### What it checks before training

Before any run trains, `experiment train` checks that:

- the file's fields can be used;
- each arm's config is one that `train` reads, so a setting that does not
  exist stops the experiment;
- the corpus and each tokenizer are built. If one is not, the message gives
  the command that builds it;
- no base run of an earlier experiment is still training.

### A changed file

You write the experiment's file, such as
`embedding_model/configs/experiments/dim.yaml`. Its first training keeps a copy
of it in the experiment's folder, `embedding_model/outputs/experiments/dim/`,
as `experiment.yaml`. Once the experiment has been trained, a `dim.yaml` whose
fields differ from that copy is refused, such as one with another seed or
another arm. A comment, another order of the fields, or a default written out
is no change. The message names the fields that differ, and gives two next
steps:

- to train the changed file from the start, delete the folder,
  `embedding_model/outputs/experiments/dim/`. Its runs go with it;
- to keep the runs, and ask a new question, save the changed file under a new
  name, such as `dim-small.yaml`. Its base can reuse the earlier runs, as
  `base: dim/base`.

[design_decisions.md](design_decisions.md#a-changed-experiment-file-is-refused-rather-than-trained-again)
gives the reasons.

### GPUs and CPUs

The runs train on the machine where `experiment train` runs:

- with GPUs, one run per GPU at a time. Each run sees one GPU of its own,
  through `CUDA_VISIBLE_DEVICES`. When a run ends, the next one starts on its
  GPU;
- without a CUDA GPU, one run at a time. `train` picks the device, as
  [training.md](training.md#gpus-and-cpus) says.

`CUDA_VISIBLE_DEVICES` chooses the GPUs on a machine that has several:

```
CUDA_VISIBLE_DEVICES=1,2 python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml
```

[design_decisions.md](design_decisions.md#an-experiment-queues-its-runs-itself-rather-than-submitting-each-as-a-job)
gives the reasons.

### On a cluster with SLURM

Submit the experiment as one job, which asks for the GPUs that its runs share,
from the repository root:

```
sbatch --nodes=1 --gpus=4 --cpus-per-task=20 --mem=128G --time=04:00:00 \
    --wrap "python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml"
```

- `--gpus=4`: four runs train at a time, one on each GPU. Always ask for the
  GPUs with `--gpus`. A job that asks for none gets no GPU from SLURM. Some
  clusters do not hide the other GPUs from a job. There, it can still see
  them, and its runs would use GPUs that another job may hold.
- `--cpus-per-task=20`: about 5 CPUs for each run that trains at a time. Each
  run reads its pairs with 4 worker processes.
- `--mem`: enough memory for the runs that train at a time.
- `--time`: a limit. It only stops a job that takes too long. The job ends,
  and frees its GPUs, when its last run ends.
- `--nodes=1`: the runs train on the GPUs of the node where the job starts.

Your cluster may need other options too, such as a partition, `-p`. Its own
guide says which. [SLURM's sbatch page](https://slurm.schedmd.com/sbatch.html)
lists every option. `sinfo -N -p <partition> -O "NodeList,Gres,GresUsed"`
shows each node's GPUs and how many are in use, `squeue --me` shows your jobs,
and `scancel <job id>` stops one.

The job keeps its GPUs until its last run ends. While the last runs train, the
GPUs of the runs that are done wait. The runs that are done stay, so a job
that is stopped, such as by its time limit, loses only the runs that were
training. Submitting it again trains those. Train an experiment in one job at
a time: a second job would train again the runs that the first one is
training.

### How the arms are compared

Each run gives one number: its score on the validation pairs, by the
decision's `metric`, from the run's manifest. An arm's score is the mean over
its seeds, with its 95% t interval.

In the printout, the base's line gives its mean, with its interval. Each
arm's line gives its mean minus the base's, and the interval of that
difference. Then it gives the p of a test of no difference, and the verdict
of the arm's test.

Each arm minus the base gets [Welch's interval and Welch's t-test](design_decisions.md#arms-are-compared-by-their-seeds-scores-with-welchs-interval-and-t-test),
unpaired. The interval excludes 0 exactly when the test's p is below the arm's
alpha. When neither arm's scores vary, such as when every run scores 1, the
difference is exact. Its interval is the difference itself, and p is 1 when it
is 0, and 0 otherwise.

The decision's `alpha` is [split over the arms](design_decisions.md#alpha-is-split-over-the-arms-by-bonferroni). With 2 arms,
each is tested at 0.05 / 2 = 0.025, and its interval is a 97.5% interval.

- The [`superiority` test passes](design_decisions.md#the-superiority-test-passes-on-a-significant-gain-of-at-least-the-minimum-effect)
  when both hold: the p is below the arm's alpha, and the arm's mean is at
  least `minimum_effect` above the base's. A significant gain of 0.005 does
  not pass the test, with the default of 0.01.
- The [`non-inferiority` test passes](design_decisions.md#the-non-inferiority-test-passes-when-the-interval-stays-above-minus-the-minimum-effect)
  when the interval's low end is above minus `minimum_effect`, such as above
  -0.01.

| test | when it passes | when it does not, and the interval still reaches the margin | when it does not, and the interval lies wholly past the margin |
|---|---|---|---|
| `superiority` | `better by at least 0.01` | `not shown better by at least 0.01` | `not better by at least 0.01` |
| `non-inferiority` | `no more than 0.01 worse` | `may be more than 0.01 worse` | `more than 0.01 worse` |

The middle column says too little, either way. The last has settled the
question: more seeds would not make the arm pass.

Each arm's row also gives [`seeds_needed`](design_decisions.md#an-arm-that-does-not-pass-its-test-is-checked-for-enough-seeds-at-the-minimum-effect), the seeds per arm
that its test needs, at the two arms' seed-to-seed spreads:

- for `superiority`, the fewest seeds at which a mean gain of exactly
  `minimum_effect` would be significant, by Welch's t-test;
- for `non-inferiority`, the fewest seeds that give an 80% chance to pass,
  when the arm and the base do not differ. This is the power of Welch's
  t-test, from the noncentral t distribution, scipy's `stats.nct`.

scipy's `optimize.elementwise` finds both counts: `bracket_root` finds a count too few and a count enough, and `find_root` finds the exact count between them, rounded up.

An arm whose interval still reaches the margin, and that had fewer seeds
than it needs, is also marked `underpowered: <k> seeds needed`. To settle its
question, ask it again in a new experiment, with the same number of arms, at
k seeds or more. Build it on the same base. Then the base's runs
at the seeds already trained are reused; see
[Building on an earlier experiment](#building-on-an-earlier-experiment).

The verdict reads the difference and the interval as `differences.csv` shows
them, to 5 decimals. So a gain shown as 0.01 is a gain of at least 0.01.

### The recommendation

`experiment compare` [recommends an arm](design_decisions.md#the-comparison-recommends-one-arm-or-none-when-several-pass), and you choose:

- one arm passes its test: that arm is recommended;
- no arm passes: the base is recommended;
- several arms pass: none is recommended, and each one's line is printed.

When the base is recommended, and it is itself an arm of an earlier
experiment, its line names that arm, such as `base: dim/dim128`. The next
experiment then reuses its runs.

The comparison uses the validation pairs only, never the benchmarks. So the
recommended arm's best run comes with the command that scores it on the
benchmarks. It names the benchmarks that the run's corpus was checked against
for leakage, as `score` requires; see [Scoring a run](training.md#scoring-a-run).
Then [`compare`](comparing.md) compares the run with other models.

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
| a diamond with an error bar, in the right panel | the arm minus the base, with Welch's interval, at the confidence of the arm's test: 95%, or with several arms, its Bonferroni level, as the axis says. A filled diamond: the arm passes its test. A hollow one: it does not | scipy's `ttest_ind`, with `equal_var=False`: the rows of `differences.csv` |
| the solid line at 0, in the right panel | no difference from the base | |
| the gray band and its dashed edges, in the right panel | minus to plus `minimum_effect` | from the decision |
| a color | blue for the base, dark orange for a non-inferiority arm, purple for a superiority arm | three colors of matplotlib's `petroff10`, which stay apart under each color vision deficiency |
| the legend | the colors, as swatches; the right panel's filled and hollow diamonds; and the bands | |
| `arms.csv` | each arm's mean, sd and interval, and each seed's score | as the left panel |
| `differences.csv` | each arm minus the base: the difference with its interval, the p of a test of no difference, the verdict, and the seeds that the arm's test needs | Welch's interval and t-test, from scipy's `ttest_ind` with `equal_var=False`. The seeds needed by Welch's t-test, and for non-inferiority by its power, from scipy's noncentral t, `stats.nct`. Each count from scipy's `optimize.elementwise`: `bracket_root`, then `find_root` |

In the right panel, 0 is at the center. So the side where an arm is worse and
the side where it is better have the same length. A `non-inferiority` arm
passes when its interval stays right of the band's left edge. A `superiority`
arm passes when its interval stays right of 0, and its diamond is on or right
of the band's right edge.

The left panel's band only helps to read the scores: the verdicts come from
the right panel's intervals. Each verdict's words are in the printout and in
`differences.csv`.
