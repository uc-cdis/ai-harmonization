# Experiments

An experiment asks one question: does a change to a run's config help? Its
file gives the **base**, which is a run's config, and one or more **arms**.
Each arm is a version of the base that changes some settings.
`experiment train` trains the base and each arm at several seeds. The base or
one arm, at one seed, is one [run](training.md), as `train` trains it.
`experiment compare` then compares each arm with the base, and recommends an
arm to build on.

Model development is a series of experiments. Each one builds on an arm of the
one before. You choose that arm.

## Running `experiment`

```
python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml    # train the runs that are not done
python -m embedding_model experiment status --config embedding_model/configs/experiments/dim.yaml   # check which runs are done
python -m embedding_model experiment compare --config embedding_model/configs/experiments/dim.yaml  # compare the arms, once every run is done
```

- **Give the same `--outputs` to all three commands.** `experiment status` and
  `experiment compare` read the folder that `experiment train` wrote.

### Writing an experiment's file

An experiment's file is a YAML file. The experiment takes its name from the
file: `dim.yaml` is the experiment `dim`. Keep your experiments' files in
`embedding_model/configs/experiments/`, apart from `inputs/` and `outputs/`,
which the tools write. Git ignores all three.

The file names the base, the seeds and the arms. Each arm names the settings
that it changes. This one varies `embedding_dim`, the length of each token's vector
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
    # The example is small enough for a CPU, where pinned memory has no use.
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

Vary one setting per experiment, as `dim.yaml` varies `embedding_dim`.
[The experiment's file](#the-experiments-file) lists every field and its
default.

### Training it

`experiment train` trains each run of the experiment that is not done:

- Each run is the `train` command, `python -m embedding_model train`, with
  the arm's config and the seed.
- Each run's output goes to the run's log, `logs/<run>.log` in the
  experiment's folder.
- A run that fails does not stop the others, and leaves no folder. At the
  end, `experiment train` lists the runs that failed, and exits with status
  1.
- If you train the same file again, `experiment train` trains only the runs
  that are not done, the failed ones included.
- After you build the corpus again, the runs that are done stay as they are.
  To train them on the new build:
  1. Train each tokenizer that the experiment uses again, with the same
     [`tokenizer train`](tokenizer.md#running-tokenizer-train) command as
     before: the same `--corpus`, `--vocab-size`, `--min-frequency` and
     `--name`. The tokenizer's `manifest.json` holds its settings.
  2. Delete the experiment's folder.
  3. Train the experiment again.

  Do the same for each experiment that builds on it, the earlier experiment
  first.

### Following it

For the base and each arm, `experiment status` counts the runs that are done, started and
not started:

```
$ python -m embedding_model experiment status --config embedding_model/configs/experiments/dim.yaml
experiment dim: 11 of 15 runs done
  base     5 done
  dim128   3 done, 2 started
  dim2048  3 done, 2 not started
```

A run is done when its folder has its manifest. A run has started when its
hidden staging folder, `.<run>~<random>/`, sits beside its folder. A started
run is training now, or it was killed before it finished, for example by
`scancel` or a time limit.

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

To build on an arm, copy its `base:` line into the next experiment's file,
as [Building on an earlier experiment](#building-on-an-earlier-experiment)
says. [How the arms are compared](#how-the-arms-are-compared) gives the verdicts,
and [The recommendation](#the-recommendation) says which arm
`experiment compare` recommends.


## What `experiment` writes

`experiment train` writes to `embedding_model/outputs/experiments/<name>/`.
`experiment compare` writes `comparison/` in the same folder.

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
base's config is `base.yaml`. You can also give an arm's config to `train`
yourself, as a run's config. `train` then writes the run to
`embedding_model/outputs/runs/`, not into the experiment's folder.

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

One row for the base, then one row per arm:

| column | what it holds |
|---|---|
| `arm`, `test`, `metric` | the arm's name (`base` for the base), its test (empty for the base), and the metric that decides |
| `seeds` | the number of seeds |
| `mean`, `sd` | the mean of the seeds' scores, and their spread from seed to seed |
| `low`, `high` | the mean's 95% interval |
| `s<seed>`, for example `s0` | each seed's score |

#### `differences.csv`

One row per arm. The base has no row:

| column | what it holds |
|---|---|
| `arm`, `test`, `metric` | the arm, its test, and the metric that decides |
| `difference` | the arm's mean minus the base's |
| `low`, `high` | Welch's interval of the difference |
| `confidence` | the interval's level, for example 0.975 with 2 arms |
| `df` | Welch's degrees of freedom |
| `p` | the p-value of Welch's t-test of no difference |
| `alpha` | the arm's alpha: the decision's, divided by the number of arms |
| `minimum_effect` | the decision's minimum effect |
| `passes` | whether the arm passes its test |
| `seeds_needed` | the seeds per arm that its test needs |
| `underpowered` | whether the arm's question is still open, with fewer seeds than `seeds_needed`, as [How the arms are compared](#how-the-arms-are-compared) says |
| `verdict` | the verdict, as printed |

#### `figures/`

`arms_and_differences`: each arm's seeds on the left, and each arm minus the
base on the right. [Reading the figure](#reading-the-figure) says how to read
it.

#### `manifest.json`

The comparison's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `experiment`: the experiment's name
- `built`: the date of the comparison
- `sources`: the experiment's `experiment.yaml`, and each run's
  `manifest.json`, each with its SHA-256
- `settings`: the decision's fields, each arm's alpha and its correction, the
  intervals and the test. It also holds `target_power`, the power that
  `seeds_needed` aims at for non-inferiority, and `power`, the method that
  computes it.

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
| `seeds` | yes | the seeds, as a list of at least 2 different whole numbers, from 0 to 4,294,967,295 |
| `decision` | no | the rule that decides between the arms: see the next table |
| `arms` | yes | each arm, by its name: its `test`, and the settings that it changes, by section. An arm is not named `base`, and does not change `data.corpus`. |

Neither the base nor an arm sets `training.seed`: each run's seed comes from
`seeds`. An arm changes only the settings that it names. Each other setting
is the base's.

The `decision` block has a default for each of its fields:

| field | what it holds | default |
|---|---|---|
| `scored_on` | the data whose scores decide: `validation`, the corpus's validation pairs, the only choice for now | `validation` |
| `metric` | the validation metric that decides: `mrr@10`, `accuracy@1`, `accuracy@3`, `accuracy@5` or `accuracy@10` | `mrr@10` |
| `minimum_effect` | the margin, on the metric's scale, greater than 0 and less than 1. For `superiority`, the smallest gain worth a change. For `non-inferiority`, the largest loss worth taking. | `0.01` |
| `alpha` | the chance, over the whole experiment, that an arm passes by chance alone, greater than 0 and less than 1. `experiment compare` splits it over the arms. | `0.05` |

Each arm names its `test`, the question that it asks of the base:

- `superiority`: is the arm better than the base, by at least
  `minimum_effect`? Use it for an arm that costs the same as the base, or
  more, for example a larger table.
- `non-inferiority`: is the arm no worse than the base, by more than
  `minimum_effect`? Use it for an arm that is cheaper to train or to run, for
  example a smaller table.

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

- The base is that arm's full config, from the earlier experiment's
  `configs/`. The new experiment reuses the arm's runs at the same seeds. At
  a seed where the earlier experiment has no run, `experiment train` trains
  the base in the new experiment's folder, as `base_s<seed>`.
- Train a later experiment only when the earlier one is done. Otherwise both
  can train the same base run.
- `base: dim/base` builds on the earlier experiment's own base, and reuses
  its runs.
- The earlier experiment's folder must stay.
- You may delete an earlier experiment and train it again with other
  settings. Then delete the folder of each experiment that builds on it, and
  train that experiment again. No command checks that you did this. Without
  it, the later experiment compares runs of two different bases.

To delete the earlier experiment's folder:

1. Copy the arm's config, for example
   `embedding_model/outputs/experiments/dim/configs/dim128.yaml`.
2. Paste that config's sections under `base:` in the later experiment's
   file.
3. If the later experiment has trained, delete its folder, or save its file
   under a new name. `experiment` refuses a trained file whose base changed,
   as [A changed file](#a-changed-file) says.

The later experiment then trains all its runs again.

### A changed file

After the first training, each `experiment` command refuses a file whose
fields differ from its copy, [`experiment.yaml`](#experimentyaml). An
example is a file with another seed or another arm. A comment, another order of the
fields, or a default written out is no change. The message names the fields
that differ, and gives two next steps:

- to train the changed file from the start, delete the experiment's folder,
  for example `embedding_model/outputs/experiments/dim/`. This deletes the
  experiment's runs too.
- to keep the runs, and ask a new question, save the changed file under a new
  name, for example `dim-small.yaml`. Its base can reuse the earlier runs, as
  `base: dim/base`.

### GPUs and CPUs

The runs train on the machine where `experiment train` runs:

- with GPUs, one run per GPU at a time. Each run sees one GPU of its own,
  through `CUDA_VISIBLE_DEVICES`.
- without a CUDA GPU, one run at a time. `train` picks the device, as
  [training.md](training.md#gpus-and-cpus) says.

`CUDA_VISIBLE_DEVICES` chooses the GPUs on a machine that has several:

```
CUDA_VISIBLE_DEVICES=1,2 python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml
```

### On a cluster with SLURM

From the repository root, submit the experiment as one
[SLURM](https://slurm.schedmd.com/overview.html) job, which asks for the
GPUs that its runs share:

```
sbatch --nodes=1 --gpus=4 --cpus-per-task=20 --mem=128G --time=04:00:00 \
    --wrap "python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml"
```

- `--gpus=4`: four runs train at a time, one on each GPU. Always ask for the
  GPUs with `--gpus`, as [training.md](training.md#on-a-cluster-with-slurm)
  says.
- `--cpus-per-task=20`: about 5 CPUs for each run that trains at a time.
- `--mem`: enough memory for the runs that train at a time.
- `--time`: a limit. The job ends, and frees its GPUs, when its last run
  ends.
- `--nodes=1`: the runs train on the GPUs of the node where the job starts.

Your cluster may need other options too, for example a partition, `-p`. The
cluster's own guide says which.
[SLURM's sbatch page](https://slurm.schedmd.com/sbatch.html) lists every
option. These SLURM commands show the cluster and your jobs, and stop a job:

- `sinfo -N -p <partition> -O "NodeList,Gres,GresUsed"` shows each node's
  GPUs, and how many are in use
- `squeue --me` shows your jobs
- `scancel <job id>` stops a job

If the job stops, for example at its time limit, it loses only the runs that
were training. Submit it again, and it trains those runs. Train an experiment
in one job at a time.

### How the arms are compared

Each run gives one number: its score on the validation pairs, by the
decision's `metric`. An arm's score is the mean over its seeds, with its 95%
t interval. For each arm, `experiment compare` takes the arm's mean minus the
base's mean. It gives this difference Welch's interval and Welch's t-test.
The decision's `alpha` is split over the arms. With 2 arms, each arm's test is
at 0.05 / 2 = 0.025, and its interval is a 97.5% interval.

- The `superiority` test passes when the p-value is below the arm's alpha,
  and the arm's mean is at least `minimum_effect` above the base's.
- The `non-inferiority` test passes when the interval's low end is above
  minus `minimum_effect`, for example above -0.01.

| test | passes | open | settled |
|---|---|---|---|
| `superiority` | `better by at least 0.01` | `not shown better by at least 0.01` | `not better by at least 0.01` |
| `non-inferiority` | `no more than 0.01 worse` | `may be more than 0.01 worse` | `more than 0.01 worse` |

- **passes:** the arm passes its test.
- **open:** the arm does not pass, but its interval still holds a difference
  that would pass. More seeds could change the verdict.
- **settled:** the arm does not pass, and its interval lies wholly past the
  margin.

`experiment compare` marks an arm `underpowered: <k> seeds needed` when all of
these hold:

- the arm does not pass its test
- the arm's interval still holds a difference that would pass
- the arm had fewer seeds than its test needs

To settle the arm's question, ask it again in a new experiment, with the same
number of arms, at k seeds or more. Write its base as `base: <experiment>/base`,
for example `base: dim/base`. Then it reuses the base's runs.

### The recommendation

`experiment compare` recommends an arm, and you choose:

- one arm passes its test: `experiment compare` recommends that arm
- no arm passes: it recommends the base
- several arms pass: it recommends none, and prints the `base:` line of each
  arm that passes

When `experiment compare` recommends the base, the printed `base:` line
names this experiment's base, for example `base: dim/base`.

The comparison uses the validation pairs only, never the benchmarks. So
`experiment compare` prints the recommended arm's best run: its run with the
highest score on the decision's `metric`. It also prints the command that
scores the run on the benchmarks. The command names the benchmarks that the
run's corpus was checked against for [leakage](leakage.md).
[Scoring a run](training.md#scoring-a-run) describes `score --model`. Then
[`compare`](comparing.md) compares the run with other models.

### Reading the figure

The figure, `arms_and_differences`, has two panels. Each row is the base or
one arm, the base first, and reads across both panels.

- **The left panel:** a dot is one seed's score. A bar is the arm's mean,
  with its 95% confidence interval. The gray band is the base's mean, minus
  and plus `minimum_effect`.
- **The right panel:** a diamond is the arm minus the base, with Welch's
  interval at the confidence of the arm's test. The gray band runs from
  minus to plus `minimum_effect`.
- **A filled diamond** passes its test, and a hollow one does not. A
  `non-inferiority` arm passes when its interval stays right of the band's
  left edge. A `superiority` arm passes when its interval stays right of 0,
  and its diamond is on or right of the band's right edge.

The verdicts come from the right panel's intervals, not from the left
panel's band. The printout and `differences.csv` give each verdict in words.
