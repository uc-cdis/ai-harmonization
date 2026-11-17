# Training

`train` trains one model on a [corpus](corpus.md)'s training pairs, with the
trainer of [sentence-transformers](https://sbert.net). It reads its settings
from a config, a YAML file.

Each training is a **run**: one config, trained with one seed. The run's
folder holds the trained model, and the record of how `train` trained it.
`train` names a run after its config and its seed: `run_example.yaml` with
`--seed 0` gives the run `run_example_s0`. When no flag overrides a setting,
runs of one config differ only in their seed. An [experiment](experiments.md) compares many
runs: each of its arms is one config, trained at several seeds.

A run's folder holds the model, the run's config, its
[manifest](manifests.md) and the trainer's log.
[What `train` writes](#what-train-writes) lists the files.
[Details](#details) describes the config's settings, the architectures, what
a run does, GPUs and CPUs, a cluster with SLURM, Weights & Biases, and
scoring a run.

[design_decisions.md](design_decisions.md#training) gives the reasons for the
rules below.

## Running `train`

```
python -m embedding_model train --config embedding_model/examples/run_example.yaml --seed 0
python -m embedding_model train --config embedding_model/configs/runs/my_run.yaml --seed 1 --learning_rate 1e-2   # a flag overrides the config
```

Keep your runs' configs in `embedding_model/configs/runs/`, apart from
`inputs/` and `outputs/`, which the tools write. Git ignores all three.

A config has four sections: `data`, `model`, `loss` and `training`. Every
setting but the corpus and the tokenizer has a default. So a config names
those two, and the settings that it changes. For example, this config,
`my_run.yaml`, shows every section:

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

It changes three settings. Each token's vector has length 512, in place of
1024. The learning rate is 0.1, in place of 0.2. Training runs for 10 epochs,
in place of 5. `architecture: static` and `scale: 20` are the defaults, and
the example holds them only to show where they go. [The config](#the-config)
says which section each setting goes in, and links to the lists of the
settings and their defaults.

Every setting is also a command-line flag, named after the setting:
`--learning_rate 1e-2`, `--embedding_dim 128` or `--use_cpu`. A flag
overrides the config. `train --help` lists only `train`'s own flags, for
example `--config` and `--seed`. [The config](#the-config) lists the
settings. `train` refuses:

- a flag that is not a setting
- a flag for a setting of the other architecture
- a shortened flag, for example `--learning` for `--learning_rate`
- a config with an unknown section or setting

`train` names a run after its config and its seed, so a flag keeps the run's
name. A run replaces an earlier run of the same name only when the same
settings trained it. Otherwise `train` stops before it trains, and names the
settings that differ. The message gives the run's full path:

```
ValueError: run my_run_s1 exists, trained with other settings: learning_rate 0.1, here 0.01. To train this run, delete .../embedding_model/outputs/runs/my_run_s1, or save the changed config under a new name
```

## What `train` writes

`train` writes a run to `embedding_model/outputs/runs/<run>/`, or under the
folder that `--outputs` names. It writes the folder whole. So if a run fails,
the previous run of that name stays as it was.
[Inputs and outputs](../README.md#inputs-and-outputs) describes how a step
writes a folder whole.

```
runs/<run>/
    model/                the trained model, as sentence-transformers saves it
    config.yaml           the run's settings
    manifest.json         the run's manifest
    trainer_state.json    the trainer's log
```

### `model/`

`model/` holds the model at the last step, as sentence-transformers saves it.
`SentenceTransformer("<run>/model")` loads it. The weights are in
`model.safetensors`: a static model's table, or a BERT's layers. The tokenizer
is in `tokenizer.json`. `README.md` is the
model card that sentence-transformers writes.

To compare two runs' models, compare `model.safetensors`. The model card also
holds the training time, so it differs even between two runs of one config.
`config_sentence_transformers.json` holds the library versions, so it
differs only between two environments.

### `config.yaml`

`config.yaml` holds the run's settings, in the config's sections. It holds
every setting of `data`, `model` and `loss`. In `model`, these are the
architecture and the settings of the run's architecture. Of `training`,
`config.yaml` holds these settings:

- the settings that the config or the flags gave
- the changed defaults of [The `training` section](#the-training-section)
- the learning rate, the seed and `bf16`

The other settings of `training` are the library's defaults.

So `config.yaml` trains the same model again, with the same seed. `train`
names a run after its config's file, so copy `config.yaml` under a name of
its own first:

```
cp embedding_model/outputs/runs/run_example_s0/config.yaml run_example_again.yaml
python -m embedding_model train --config run_example_again.yaml --seed 0
```

This trains the run `run_example_again_s0`, next to `run_example_s0`.

On a CPU, the two models are the same, byte for byte. On a GPU,
`full_determinism: true` makes them the same too. Without it, a GPU may add
its numbers in another order on each run. A model trained on a CPU and one
trained on a GPU differ slightly, for the same reason.

### `manifest.json`

`manifest.json` is the run's manifest. It has the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: each file of `model/`, with its SHA-256
- `sources`: the corpus and the tokenizer, with their checksums. The corpus's
  entry lists the benchmarks that `corpus build` checked the corpus against
  for [leakage](leakage.md), with their versions and their files' checksums.
- `settings`: the architecture, and `format`, the
  [format](scoring.md#text-formats) of the pairs' texts, from the corpus's
  manifest
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

`trainer_state.json` holds the trainer's state at the end of training, as
the `transformers` Trainer writes it. Its `log_history` holds one entry per
log: the training loss every `logging_steps`, and the validation metrics at
each evaluation, after each epoch by default. Each entry has its `step` and
`epoch`. The metrics keep the trainer's names, for example `eval_loss` and
`eval_validation_cosine_mrr@10`. pandas reads the log as a table:

```python
pd.DataFrame(json.loads(Path("<run>/trainer_state.json").read_text())["log_history"])
```

## Details

### The config

A config has four sections, and each decides one part of a run, as the table
shows. The trained model needs `embedding_dim` to be used, so `embedding_dim`
goes in `model`. A trained model does not need its learning rate, so
`learning_rate` goes in `training`.

| section | what it decides | its settings |
|---|---|---|
| `data` | what the model learns from | `corpus`, the [corpus](corpus.md) to train on, and `tokenizer`, the [tokenizer](tokenizer.md), each by its name, for example `example` and `example-8k`. A run needs both |
| `model` | what the model is | `architecture`: `static`, the default, or `bert`. Then that architecture's settings: see [`static`](#static-a-static-embedding-model) and [`bert`](#bert-a-transformer-embedding-model-with-berts-architecture) |
| `loss` | what the model learns to do | `scale`: the factor that multiplies the similarities before the softmax of [MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss). The default is 20, as in the library |
| `training` | how the model learns | the trainer's settings, for example `learning_rate` and `num_train_epochs`: see [The `training` section](#the-training-section) |

`train` refuses a setting in the wrong section, and the error names the
section where the setting belongs:
`the model section has no setting learning_rate. learning_rate belongs in the training section`.
The `model` section takes only its own architecture's settings. So `train`
refuses `hidden_size`, a setting of `bert`, for a static model, in the config
or as a flag.

#### The `training` section

The `training` section takes any setting of the trainer, under its own name.
Most are settings of transformers'
[`TrainingArguments`](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer#transformers.TrainingArguments),
for example `learning_rate` and `num_train_epochs`. sentence-transformers
adds a few of its own, in
[`SentenceTransformerTrainingArguments`](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html).
Both pages give the library's defaults.

`train` refuses four of these settings. It refuses
[`prompts` and `router_mapping`](design_decisions.md#a-run-refuses-the-trainers-prompts-and-router_mapping).
It also refuses `load_best_model_at_end` and `resume_from_checkpoint`,
because
[the model kept is the model at the last step](design_decisions.md#the-model-kept-is-the-model-at-the-last-step)
and a run saves no checkpoint. For the same reason, `save_strategy` must stay
`no`.

This pipeline changes some of the defaults. Some come from Hugging Face's
[recipe for static models](https://huggingface.co/blog/static-embeddings), and
the others are this pipeline's own:

| setting | default here | from | the library's default |
|---|---|---|---|
| `num_train_epochs` | 5 | here. The recipe trains 1 epoch | 3 |
| `per_device_train_batch_size` | 2048 | the recipe | 8 |
| `per_device_eval_batch_size` | 2048 | the recipe | 8 |
| `learning_rate` | the architecture's: 0.2 for `static`, 1e-4 for `bert` | the recipe for `static`, [BERT's paper](#bert-a-transformer-embedding-model-with-berts-architecture) for `bert` | 5e-5 |
| `warmup_steps` | 0.1, a share of the steps | the recipe | 0 |
| `eval_strategy` | `epoch`: after each epoch | here. The recipe evaluates every 250 steps | `no` |
| `logging_steps` | 50 | here. The recipe logs every 250 steps | 500 |
| `save_strategy` | `no` | here. The recipe saves every 250 steps | `steps` |
| `dataloader_num_workers` | 4 | here | 0 |

### Architectures

`architecture`, in the `model` section, chooses one of two models. Both are
sentence-transformers models. Both train with the same loss, the same trainer
and the same defaults, except the learning rate.

#### `static`: a static embedding model

`static` is the default architecture. The model is a table that holds one
vector per token. A text's embedding is the average of its tokens' vectors,
as sentence-transformers' `StaticEmbedding` computes it. A static model's
default learning rate is 0.2, the rate of Hugging Face's
[recipe for static models](https://huggingface.co/blog/static-embeddings).

With `architecture: static`, the `model` section takes one setting:

| setting | what it sets | default here | the library's default |
|---|---|---|---|
| `embedding_dim` | the length of each token's vector, and of the embedding | 1024 | none: `StaticEmbedding` requires it |

#### `bert`: a transformer embedding model with BERT's architecture

The model is BERT's encoder, as `BertModel` of
[transformers](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/bert#transformers.BertModel) builds it, without
BERT's pretraining heads. `train` trains it from scratch: the model starts
from random weights, not from a pretrained BERT. Its attention layers let
each token's vector depend on the other tokens of the text. Mean pooling over
the token vectors gives the text's embedding. A BERT's default learning rate
is 1e-4, the rate at which the BERT paper
([Devlin et al., 2019](https://doi.org/10.18653/v1/N19-1423)) pretrains.

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

This config trains a BERT of 2 layers and 256 dimensions.

With `architecture: bert`, the `model` section takes these settings of
[`BertConfig`](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/bert#transformers.BertConfig),
under its own names. The first four set the model's shape. Their defaults
are BERT-Small's, from Google's release of smaller BERT models
([Turc et al., 2019](https://doi.org/10.48550/ARXIV.1908.08962)): 4 layers of
512, with 8 heads. A text is cut at 128 tokens. The other defaults are
`BertConfig`'s:

| setting | what it sets | default here | `BertConfig`'s default |
|---|---|---|---|
| `hidden_size` | the length of each token's vector, and of the embedding | 512 | 768 |
| `num_hidden_layers` | the number of transformer layers | 4 | 12 |
| `num_attention_heads` | the number of attention heads in each layer. The number must divide `hidden_size` | 8 | 12 |
| `intermediate_size` | the size of each layer's feed-forward part | 2048 | 3072 |
| `hidden_dropout_prob` | the dropout on the embeddings and on each layer's output | 0.1 | 0.1 |
| `attention_probs_dropout_prob` | the dropout on the attention weights | 0.1 | 0.1 |
| `max_position_embeddings` | the longest text, in tokens. A longer text is cut. The setting is also the model's `max_seq_length` | 128 | 512 |
| `initializer_range` | the standard deviation of the random starting weights | 0.02 | 0.02 |

The vocabulary is the tokenizer's. A BERT reads a text without special
tokens, as a static model does: no `[CLS]` and no `[SEP]`.
[design_decisions.md](design_decisions.md#a-second-architecture-bert-trained-from-scratch)
says why.

### What a run does

1. It reads the corpus and the tokenizer, each checked against its manifest.
2. It sets the seed, then builds the model. So the seed decides the model's
   starting weights and the order of the pairs.
3. For a static model, it turns the texts of every training and validation
   pair into token ids, once, without special tokens. The ids stay in memory.
   They are the ids that the model would get from the texts itself, so the
   model trains the same. A BERT trains on the texts: the trainer tokenizes
   each batch.
4. It trains with MultipleNegativesRankingLoss: in each batch, an anchor's
   positive is ranked against the other pairs' positives.
5. It evaluates the model on the validation pairs, after each epoch by
   default. The trainer computes the loss on them as on the training pairs.
   For each validation anchor, the evaluator, sentence-transformers'
   `InformationRetrievalEvaluator`, also ranks every validation positive,
   which gives MRR@10 and accuracy@k. Two texts that the tokenizer normalises
   to one text count as one. [The metrics](scoring.md#the-metrics) defines
   MRR@10. `score` reports it on the benchmarks too.
6. It keeps the model at the last step. The model's validation metrics are
   those of the last evaluation. If the trainer did not evaluate the last
   step, the run evaluates the model once more.

A tokenizer that learned its vocabulary from another corpus, or from another
build of the run's corpus, could hold the run's validation texts. So
`train` refuses such a tokenizer. Two builds are the same when their pairs
have the same number of rows and the same content checksum. The error gives
the command that trains a tokenizer on the current build of the run's
corpus, with the same vocabulary size and minimum frequency.

### GPUs and CPUs

A run trains on one device, which `transformers` picks:

| where the run starts | the run trains on |
|---|---|
| a machine with one NVIDIA GPU | that GPU |
| a machine with several | GPU 0, or the first that `CUDA_VISIBLE_DEVICES` lists. `CUDA_VISIBLE_DEVICES=2` chooses another |
| a Mac with Apple silicon | its GPU, through PyTorch's MPS backend, which `transformers` picks by itself |
| a machine with neither | the CPU |
| a SLURM job with `--gpus=1` | the GPU that SLURM gives the job |
| anywhere, with `use_cpu: true` in the config, or `--use_cpu` | the CPU |

If `CUDA_VISIBLE_DEVICES` lists several GPUs, `train` prints the GPU that the
run keeps:

```
GPUs 2, 3 are listed: this run trains on GPU 2 only. To choose another, set CUDA_VISIBLE_DEVICES, for example CUDA_VISIBLE_DEVICES=3
```

A static model trains on a CPU in minutes, even on a corpus of all the
sources. A BERT trains far more slowly, and needs a GPU for such a corpus.

On a GPU that supports bf16, `bf16: true` in the config, or `--bf16`, trains
faster and with less memory, as Hugging Face's recipe does. `bf16` is off by
default, as in the library. `transformers` refuses bf16 on a GPU without
bf16, and on a CPU unless the config sets `use_cpu: true`.

### On a cluster with SLURM

Submit the run as a [SLURM](https://slurm.schedmd.com/) job that asks for one
GPU and a few CPUs, from the repository root:

```
sbatch --gpus=1 --cpus-per-task=8 --mem=64G --time=02:00:00 --wrap "python -m embedding_model train --config embedding_model/configs/runs/my_run.yaml --seed 0"
```

- `--gpus=1`: the run's GPU. Always ask for it: a job that asks for none
  gets no GPU from SLURM. Some clusters do not hide the other GPUs from such
  a job. The run would then use a GPU that another job may hold.
- `--cpus-per-task`: at least 5. The trainer reads the pairs with 4 worker
  processes.
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
shows. Without `WANDB_PROJECT`, sentence-transformers' trainer sets the
project `sentence-transformers`. W&B shows each run under its name, for
example `run_example_s0`.

### Scoring a run

`score --model` takes a run's folder, or its `model/` folder:

```
python -m embedding_model score --model embedding_model/outputs/runs/run_example_s0 --benchmark topmed
```

`score` scores the run in its training [format](scoring.md#text-formats),
unless `--format` names another format. `score` refuses a benchmark that the
leakage check of the run's corpus did not cover, because the model may have
learned that benchmark's mappings. `score` looks for the benchmark in the
run's manifest, which lists the benchmarks of the leakage check. It matches
the benchmark by its name and its files' checksums. So a new version whose files did
not change passes. The error gives the `corpus build` command that
checks the corpus again.

`score` also refuses a model whose files differ from the run's record. It
refuses a record that cannot be read, or that lacks a field that scoring
reads. `score` writes the scores to the run's own folder, as
`scores/<benchmark>/<format>/`. [scoring.md](scoring.md) describes the
scores.
