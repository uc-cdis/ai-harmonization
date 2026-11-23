# Training

`train` trains one model on a [corpus](corpus.md)'s training pairs, with the
trainer of [sentence-transformers](https://sbert.net). It reads its settings
from a config, a YAML file.

Each training is a **run**: one config, trained with one seed. `train` names
a run after its config and its seed: `run_example.yaml` with `--seed 0` gives
the run `run_example_s0`. An [experiment](experiments.md) trains many runs:
its base and each arm are configs, each trained at several seeds.

## Running `train`

```
python -m embedding_model train --config embedding_model/examples/run_example.yaml --seed 0
python -m embedding_model train --config embedding_model/configs/runs/my_run.yaml --seed 1 --learning_rate 1e-2   # a flag overrides the config
```

Keep your runs' configs in `embedding_model/configs/runs/`, apart from
`inputs/` and `outputs/`, which the tools write. Git ignores all three.

A config has four sections: `data`, `model`, `loss` and `training`. Every
setting but the corpus and the tokenizer has a default. So a config names
those two, and the settings that it changes. For example, `my_run.yaml`:

```yaml
data:
  corpus: example
  tokenizer: example-8k
model:
  architecture: static
  embedding_dim: 512
loss:
  scale: 20
training:
  learning_rate: 0.1
  num_train_epochs: 10
```

It changes three settings: `embedding_dim`, `learning_rate` and
`num_train_epochs`. `architecture: static` and `scale: 20` are the defaults.
The example holds them only to show where they go.
[The config](#the-config) lists the settings and their defaults.

Every setting is also a command-line flag, named after the setting:
`--learning_rate 1e-2`, `--embedding_dim 128` or `--use_cpu`. A flag
overrides the config, and the run keeps its name. `train --help` lists only
`train`'s own flags, for example `--config` and `--seed`.

- **After you build a corpus again**, train its tokenizer again, as
  [The workflow](../README.md#the-workflow) says. `train` refuses a
  tokenizer from another build of the corpus.
- **If a run of the same name exists**, `train` replaces it, and its scores,
  only if the same settings trained that run. Otherwise `train` stops before
  it trains.

## What `train` writes

`train` writes a run to `embedding_model/outputs/runs/<run>/`.

```
runs/<run>/
    model/                the trained model, as sentence-transformers saves it
    config.yaml           the run's settings
    manifest.json         the run's manifest
    trainer_state.json    the trainer's state
```

### `model/`

`model/` holds the model after the last training step.
`SentenceTransformer("<run>/model")` loads it. The
weights are in `model.safetensors`, and the tokenizer is in `tokenizer.json`.
`README.md` is the model card that sentence-transformers writes.

### `config.yaml`

`config.yaml` holds the run's settings. Give it to `train` to train the same
model again, with the same seed. `train` names a run after its config's file,
so copy `config.yaml` under a new name first:

```
cp embedding_model/outputs/runs/run_example_s0/config.yaml embedding_model/configs/runs/run_example_again.yaml
python -m embedding_model train --config embedding_model/configs/runs/run_example_again.yaml --seed 0
```

Train both runs on the same machine, with the same library versions. When
both train on the CPU, the new run's `model.safetensors` is the same as the
first run's, byte for byte. When both train on the same GPU,
`full_determinism: true` makes it the same too. A run on a CPU and a run on a
GPU give slightly different models.

### `manifest.json`

The run's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: each file of `model/`, with its SHA-256
- `sources`: the corpus and the tokenizer, with their checksums. The corpus's
  entry lists the benchmarks that `corpus build` checked the corpus against
  for [leakage](leakage.md).
- `settings`: the architecture, and `format`, the
  [format](scoring.md#text-formats) of the pairs' texts
- `counts`: the numbers of training and validation pairs, of steps and of
  epochs, and the seconds that training took

It also has four fields of its own:

- `validation`: the loss, [MRR@10 and accuracy@k](scoring.md#the-metrics) on
  the validation pairs, at the last step
- `device`: the GPU's name, or the type of device that `transformers` picked,
  for example `cpu`
- `ai_harmonization`: the commit of the code when the run started, and
  whether the code had uncommitted changes
- `libraries`: the versions of Python and of the libraries that trained the
  model

### `trainer_state.json`

The trainer's state at the end of training. Its `log_history` holds the
training loss every `logging_steps` steps, and the validation metrics at
each evaluation.

## Details

### The config

| section | its settings |
|---|---|
| `data` | `corpus`, the [corpus](corpus.md) to train on, and `tokenizer`, the [tokenizer](tokenizer.md), each by its name, for example `example` and `example-8k`. A run needs both. |
| `model` | `architecture`: `static`, the default, or `bert`. Then that architecture's settings: see [`static`](#static-a-static-embedding-model) and [`bert`](#bert-a-transformer-embedding-model-with-berts-architecture). |
| `loss` | `scale`: the factor that multiplies the similarities before the softmax of [MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss). The default is 20. |
| `training` | the trainer's settings, for example `learning_rate` and `num_train_epochs`: see [The `training` section](#the-training-section) |

#### The `training` section

The `training` section takes any setting of the trainer, under its own name.
Most are settings of transformers'
[`TrainingArguments`](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer#transformers.TrainingArguments).
sentence-transformers adds a few of its own, in
[`SentenceTransformerTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html).
Both pages give the library's defaults.

`train` refuses `prompts`, `router_mapping`, `load_best_model_at_end` and
`resume_from_checkpoint`. `save_strategy` must stay `no`.

These settings have defaults of their own here:

| setting | default |
|---|---|
| `num_train_epochs` | 5 |
| `per_device_train_batch_size` | 2048 |
| `per_device_eval_batch_size` | 2048 |
| `learning_rate` | the architecture's: 0.2 for `static`, 1e-4 for `bert` |
| `warmup_steps` | 0.1. A value below 1 is a share of all the steps. |
| `eval_strategy` | `epoch`: after each epoch |
| `logging_steps` | 50 |
| `save_strategy` | `no` |
| `dataloader_num_workers` | 4 |

### Architectures

`architecture`, in the `model` section, chooses one of two models.

#### `static`: a static embedding model

`static` is the default architecture. The model is a table that holds one
vector per token. A text's embedding is the average of its tokens' vectors.

| setting | what it sets | default |
|---|---|---|
| `embedding_dim` | the length of each token's vector, and of the embedding | 1024 |

#### `bert`: a transformer embedding model with BERT's architecture

The model is BERT's encoder, trained from scratch: it starts from random
weights, not from a pretrained BERT. Mean pooling over the token vectors
gives the text's embedding. For example, this config trains a BERT of 2
layers and 256 dimensions:

```yaml
data:
  corpus: example
  tokenizer: example-8k
model:
  architecture: bert
  hidden_size: 256
  num_hidden_layers: 2
  num_attention_heads: 4
  intermediate_size: 1024
```

With `architecture: bert`, the `model` section takes these settings of
[`BertConfig`](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/bert#transformers.BertConfig),
under its own names:

| setting | what it sets | default |
|---|---|---|
| `hidden_size` | the length of each token's vector, and of the embedding | 512 |
| `num_hidden_layers` | the number of transformer layers | 4 |
| `num_attention_heads` | the number of attention heads in each layer. The number must divide `hidden_size`. | 8 |
| `intermediate_size` | the size of each layer's feed-forward part | 2048 |
| `hidden_dropout_prob` | the dropout on the embeddings and on each layer's output | 0.1 |
| `attention_probs_dropout_prob` | the dropout on the attention weights | 0.1 |
| `max_position_embeddings` | the longest text, in tokens. A longer text is cut. | 128 |
| `initializer_range` | the standard deviation of the random starting weights | 0.02 |

### GPUs and CPUs

A run trains on one device, which `transformers` picks:

| where the run starts | the run trains on |
|---|---|
| a machine with one NVIDIA GPU | that GPU |
| a machine with several | GPU 0, or the first that `CUDA_VISIBLE_DEVICES` lists. `CUDA_VISIBLE_DEVICES=2` chooses another. |
| a Mac with Apple silicon | its GPU, through PyTorch's MPS backend |
| a machine with neither | the CPU |
| a SLURM job with `--gpus=1` | the GPU that SLURM gives the job |
| anywhere, with `use_cpu: true` in the config, or `--use_cpu` | the CPU |

If `CUDA_VISIBLE_DEVICES` lists several GPUs, `train` prints the GPU that the
run keeps.

A static model trains on a CPU in minutes, even on a corpus of all the
sources. A BERT also trains on a CPU, but far more slowly. For such a corpus,
use a GPU.

On a GPU that supports bf16, set `bf16: true` in the config, or give
`--bf16`. Training is then faster, and uses less memory. `bf16` is off by
default. `transformers` refuses
bf16 on a GPU without bf16, and on a CPU unless the config sets
`use_cpu: true`.

### On a cluster with SLURM

Submit the run as a [SLURM](https://slurm.schedmd.com/) job that asks for one
GPU and a few CPUs, from the repository root:

```
sbatch --gpus=1 --cpus-per-task=8 --mem=64G --time=02:00:00 --wrap "python -m embedding_model train --config embedding_model/configs/runs/my_run.yaml --seed 0"
```

- `--gpus=1`: the run's GPU. Always ask for it: a job that asks for none
  gets no GPU from SLURM. Some clusters do not hide the other GPUs from such
  a job. The run would then use a GPU that another job may hold.
- `--cpus-per-task`: at least 5, for the trainer's 4 worker processes.
- `--time`: a limit. It only stops a run that takes too long. The job ends,
  and frees its GPU, when the run ends.

The job's output goes to `slurm-<job id>.out`, or to the file that `--output`
names. [experiments.md](experiments.md#on-a-cluster-with-slurm) shows how to
train many runs in one job.

### Weights & Biases

`report_to: wandb`, in the config or as `--report_to wandb`, logs the run to
[Weights & Biases](https://docs.coreweave.com/products/wandb/quickstart). It
needs the `tracking` group, which `uv sync` does not install:

```
uv sync --group tracking
```

Configure W&B as its quickstart says: an account, an API key, and
`wandb login`. The trainer then logs to the project that `WANDB_PROJECT`
names, as
[W&B's Hugging Face page](https://docs.coreweave.com/products/wandb/integrations/huggingface_transformers#name-the-project)
shows. Without `WANDB_PROJECT`, the project is `sentence-transformers`. W&B
shows each run under its name, for example `run_example_s0`.

### Scoring a run

```
python -m embedding_model score --model embedding_model/outputs/runs/run_example_s0 --benchmark topmed
```

[scoring.md](scoring.md) says how `score` scores a run, and where it writes
the scores.
