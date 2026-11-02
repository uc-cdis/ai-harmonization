# Training

`train` trains one model on a [corpus](corpus.md)'s training pairs, with the
trainer of [sentence-transformers](https://sbert.net). It reads its settings
from a config, a YAML file:

```
python -m embedding_model train --config embedding_model/examples/run_example.yaml --seed 0
python -m embedding_model train --config my_run.yaml --seed 1 --learning_rate 1e-2   # a flag overrides the config
```

Each training is a **run**: one config, trained with one seed. Its folder
holds the trained model, and the record of how it was trained. A run is named
after its config and its seed: `run_example.yaml` with `--seed 0` gives the run
`run_example_s0`. Two runs of one config with different seeds train two models
that differ only by their seed. A series compares many runs: each of its arms
is one config, trained at several seeds.

[design_decisions.md](design_decisions.md#training) gives the reasons for the
rules below.

## The config

A config has four sections. Each setting has a default, so a config names only
the corpus, the tokenizer and what it changes:

```yaml
data:
  corpus: example
  tokenizer: example-8k
model:
  embedding_dim: 512
training:
  num_train_epochs: 10
```

This config trains a smaller model than the default, for longer.

The first three sections hold the settings of this pipeline:

| section | setting | what it sets | default here | the library's default |
|---|---|---|---|---|
| `data` | `corpus` | the [corpus](corpus.md) to train on, by its name | none: required | — |
| `data` | `tokenizer` | the [tokenizer](tokenizer.md), by its name | none: required | — |
| `model` | `architecture` | the architecture | `static` | — |
| `model` | `embedding_dim` | the length of each embedding | 1024 | none: `StaticEmbedding` requires it |
| `loss` | `scale` | the factor that multiplies the similarities before the softmax of [MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) | 20 | 20 |

The `training` section takes any field of
[`SentenceTransformerTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html),
the trainer's own settings, under its own name. Some of their defaults are
changed, most of them to those of Hugging Face's
[recipe for static models](https://huggingface.co/blog/static-embeddings):

| setting | default here | the library's default |
|---|---|---|
| `num_train_epochs` | 5 | 3 |
| `per_device_train_batch_size` | 2048 | 8 |
| `per_device_eval_batch_size` | 2048 | 8 |
| `learning_rate` | 0.2 | 5e-5 |
| `warmup_steps` | 0.1, a share of the steps | 0 |
| `eval_strategy` | `epoch`: after each epoch | `no` |
| `logging_steps` | 50 | 500 |
| `save_strategy` | `no` | `steps` |
| `dataloader_num_workers` | 4 | 0 |

Every setting of every section is also a flag, named after it:
`--learning_rate 1e-2`, `--embedding_dim 128` or `--use_cpu`. A flag
overrides the config. `train --help` lists `train`'s own flags. A flag that
is not a setting is refused. A flag must be written in full, such as
`--learning_rate`, not `--learning`. A section or a setting that does not
exist in the config is refused too.

## What a run does

1. It reads the corpus and the tokenizer, each checked against its manifest.
2. It turns the texts of every training and validation pair into token ids,
   once, without special tokens. The ids stay in memory. They are the ids that
   the model would get from the texts itself, so the model trains the same.
3. It sets the seed, then builds the model. So the seed decides the model's
   starting weights and the order of the pairs.
4. It trains with MultipleNegativesRankingLoss: in each batch, an anchor's
   positive is ranked against the other pairs' positives.
5. After each epoch, it evaluates the model on the validation pairs. The
   loss is computed on them as on the training pairs. Each validation anchor
   also ranks every validation positive, which gives MRR@10 and accuracy@k.
   MRR@10 is MRR, as [The metrics](scoring.md#the-metrics) defines it, with
   every rank below 10 counted as 0.
6. It keeps the model at the last step. Its validation metrics are those of
   the last evaluation. When the last step was not evaluated, the run
   evaluates the model once more.

A note is printed when the tokenizer learned its vocabulary from another
corpus.

## GPUs and CPUs

A run trains on one GPU. When it sees several, it keeps the first one, for
its own process only, and says so:

```
GPUs 2, 3 are visible: this run trains on GPU 2 only. To choose another, set CUDA_VISIBLE_DEVICES, for example CUDA_VISIBLE_DEVICES=3
```

Without a GPU, a run trains on the CPU. `--use_cpu`, or `use_cpu: true` in the
config, trains on the CPU on a machine that has a GPU. A small corpus, such as
the README's example, trains on a CPU in a few minutes.

On a GPU that supports bf16, `bf16: true` in the config, or `--bf16`, trains
faster and with less memory, as Hugging Face's recipe does. It stays off by
default, as in the library, since a CPU and older GPUs cannot use it.

On a cluster with [SLURM](https://slurm.schedmd.com/), ask for one GPU and a
few CPUs, from the repository root:

```
srun --gpus=1 --cpus-per-task=8 --mem=64G python -m embedding_model train --config my_run.yaml --seed 0
```

The same options work in an `sbatch` script. SLURM gives the job its GPU
through `CUDA_VISIBLE_DEVICES`. The trainer reads the pairs with 4 worker
processes, so ask for at least 5 CPUs. A time limit, such as `--time=02:00:00`,
only stops a run that takes too long: the job ends, and frees its GPU, when
the run ends.

## Weights & Biases

`report_to: wandb`, in the config or as `--report_to wandb`, logs the run to
[Weights & Biases](https://docs.coreweave.com/products/wandb/quickstart). It
needs the `tracking` group, which `uv sync` leaves out:

```
uv sync --group tracking
```

Set up W&B as its [quickstart](https://docs.coreweave.com/products/wandb/quickstart)
says: an account, an API key, and `wandb login`. The trainer then logs to the
project that `WANDB_PROJECT` names, as
[W&B's Hugging Face page](https://docs.coreweave.com/guides/integrations/huggingface/)
shows, and to the project `huggingface` without it. W&B shows each run under
its name, such as `run_example_s0`.

## What a run writes

`train` writes to `embedding_model/outputs/runs/<run>/`, or under the folder
that `--outputs` names. It writes the folder whole, so a run that fails leaves
the previous run of that name as it was; see
[Inputs and outputs](../README.md#inputs-and-outputs).

### `model/`

The model at the last step, as sentence-transformers saves it.
`SentenceTransformer("<run>/model")` loads it. The table is in
`model.safetensors`, and the tokenizer in `tokenizer.json`. `README.md` is the
model card that sentence-transformers writes.

To compare two runs' models, compare `model.safetensors`. The model card and
`config_sentence_transformers.json` also record the training time and the
library versions. So they differ even between two runs of one config.

### `config.yaml`

The run's settings, in the config's sections. It records every setting of
`data`, `model` and `loss`. Of `training`, it records the settings that the
config or the flags gave, the changed defaults above, the seed and `bf16`.
The other settings are the library's defaults.

So `config.yaml` trains the same model again, with the same seed. A run is
named after its config's file, so copy it under a name of its own first:

```
cp embedding_model/outputs/runs/run_example_s0/config.yaml run_example_again.yaml
python -m embedding_model train --config run_example_again.yaml --seed 0
```

This trains the run `run_example_again_s0`, next to `run_example_s0`.

### `manifest.json`

The run's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: each file of `model/`, with its SHA-256;
- `sources`: the corpus and the tokenizer, with their checksums. The corpus's
  entry lists the benchmarks that the corpus was checked against for
  [leakage](leakage.md), with their versions;
- `settings`: the architecture, and `format`, the text format of the pairs,
  from the corpus's manifest;
- `counts`: the training and validation pairs, the steps, the epochs and the
  seconds that training took.

It also has four fields of its own:

- `validation`: the loss, MRR@10 and accuracy@k at the last step;
- `device`: the GPU's name, or the type of device that the trainer chose,
  such as `cpu`;
- `ai_harmonization`: the commit of the code when the run started, and
  whether it had uncommitted changes;
- `libraries`: the versions of Python and of the libraries that trained it.

### `trainer_state.json`

The trainer's state at the end of training, as the `transformers` Trainer
writes it. Its `log_history` holds one entry per log: the training loss every
`logging_steps`, and the validation metrics after each epoch. Each entry has
its `step` and `epoch`. The metrics keep the trainer's names, such as
`eval_loss` and `eval_validation_cosine_mrr@10`. pandas reads the log as a
table:

```python
pd.DataFrame(json.loads(Path("<run>/trainer_state.json").read_text())["log_history"])
```

## Scoring a run

`score --model` takes a run's folder, or its `model/` folder:

```
python -m embedding_model score --model embedding_model/outputs/runs/run_example_s0 --benchmark topmed
```

It scores the run in its training format, unless `--format` names another. It
refuses a benchmark, by name and version, that the run's corpus was not
checked against for leakage. The model may have learned that benchmark's
mappings. It also refuses a model whose files differ from the run's record.
The scores go to the run's own folder, as
`scores/<benchmark>/<format>/`. [scoring.md](scoring.md) describes the
scores.
