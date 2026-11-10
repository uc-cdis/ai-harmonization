# Experiments

An experiment asks one question: does a change to a run's config help? It
trains the config, the **base**, and one or more **arms**, each a version of
the base that changes some settings. Each is trained at several seeds, so that
a difference between the arms can be told apart from the difference between
seeds. One arm at one seed is one [run](training.md), as `train` trains it.

`experiment train` writes the runs, the arms' configs and the runs' logs to one
folder, the experiment's folder.
[What `experiment train` writes](#what-experiment-train-writes) lists them.
[Details](#details) describes the experiment's file in full, what is checked
before training, and the GPUs and clusters that the runs train on.

Model development is a series of experiments. Each one builds on an arm of the
one before, which the person running them selects.
[design_decisions.md](design_decisions.md#experiments) gives the reasons for
the rules below.

## Running `experiment train`

```
python -m embedding_model experiment train --config embedding_model/configs/experiments/dim.yaml   # train the runs that are not done
python -m embedding_model experiment status --config embedding_model/configs/experiments/dim.yaml  # check which runs are done
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
  base        5 done
  dim128      3 done, 2 started
  dim2048     3 done, 2 not started
```

A run is done when its folder has its manifest. It has started when a staging
folder sits beside its folder: it is training now, or it was stopped before it
finished.

## What `experiment train` writes

`experiment train` writes to `embedding_model/outputs/experiments/<name>/`, or
under the folder that `--outputs` names. It writes each run whole, as `train`
does, and the arms' configs whole. A run that fails leaves no folder, and the
next training of the file trains it again; see
[Inputs and outputs](../README.md#inputs-and-outputs).

```
experiments/<name>/
    experiment.yaml
    configs/
        base.yaml   <arm>.yaml ...
    runs/
        base_s0/   <arm>_s0/ ...
    logs/
        base_s0.log   <arm>_s0.log ...
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

Each arm names its `test`, the question that it asks of the base:

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
