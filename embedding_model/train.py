"""Train one model: ``train --config <config.yaml> --seed <seed>``.

Each training is a run: one config, trained with one seed. A run trains one
model on a corpus's training pairs, with sentence-transformers' trainer, and
evaluates it on the validation pairs. It takes four steps:

1. It reads the config: a YAML file in five sections, ``data``, ``model``,
   the architecture's own section, ``loss`` and ``training``.
   ``training`` holds sentence-transformers' training arguments, under
   their own names. Any setting can also be given as
   a flag, such as ``--learning_rate 1.0e-2``, which overrides the config.
2. It reads the corpus and the tokenizer, each checked against its manifest.
   For a static model, it turns every text of the pairs into token ids once,
   before training, so the training steps never wait for the tokenizer. A
   BERT trains on the texts, and the trainer tokenizes each batch.
3. It trains with MultipleNegativesRankingLoss. After each epoch, it
   evaluates the model on the validation pairs: their loss, and how it ranks
   them, as MRR@10 and accuracy@k.
4. It writes the run's folder, ``<outputs>/runs/<config>_s<seed>/``,
   whole: the model at the last step, its settings, its record and the
   trainer's log.
"""

import os
import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path

import yaml

from embedding_model import manifests
from embedding_model.corpus import TRAIN, VALIDATION, build_command, read_corpus
from embedding_model.folders import FOLDER_NAME, FOLDER_NAME_RULE, staged
from embedding_model.manifests import file_sha256
from embedding_model.normalization import normalised
from embedding_model.architectures import ARCHITECTURES
from embedding_model.provenance import code_commit, library_versions
from embedding_model.scoring import ACCURACY_AT_K, MRR_CUTOFF, ranking_evaluator
from embedding_model.texts import recorded_format
from embedding_model.tokenizer import BATCH_SIZE, read_tokenizer

# Where the runs trained on their own are written, under the outputs folder.
RUNS = "runs"
# The training settings whose defaults here differ from the library's. The
# batch sizes and the warmup are those of Hugging Face's recipe for training a
# static model from scratch. The others are this pipeline's own, as
# design_decisions.md says. These defaults are shared by every architecture,
# since none of them depends on the model. Each architecture sets its own
# learning rate. The config overrides them.
TRAINING_DEFAULTS = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 2048,
    "per_device_eval_batch_size": 2048,
    "warmup_steps": 0.1,
    "eval_strategy": "epoch",
    "logging_steps": 50,
    "save_strategy": "no",
    "dataloader_num_workers": 4,
}
# The training settings that a run refuses for now, with the reason. The
# approaches of ai-harmonization embed both sides with the same encode call, in
# embed_query and embed_documents. It adds only the model's default prompt and
# takes a Router's default route, so a prompt or a route for each side would
# be lost in use. A run keeps the model at the last step, so it loads no
# checkpoint.
REFUSED_TRAINING = {
    "prompts": (
        "the approaches embed both sides with the same encode call, which adds"
        " no prompt for each side"
    ),
    "router_mapping": "it needs a model built with a Router module, not added yet",
    "load_best_model_at_end": (
        "a run keeps the model at the last step, and saves no checkpoint"
    ),
}


@dataclass
class DataSettings:
    """The data section: the corpus and the tokenizer that a run trains on."""

    corpus: str | None = field(
        default=None, metadata={"help": "the corpus to train on, by its name"}
    )
    tokenizer: str | None = field(
        default=None, metadata={"help": "the tokenizer, by its name"}
    )


@dataclass
class ModelSettings:
    """The model section: the architecture, whose own section holds its settings."""

    architecture: str = field(
        default="static",
        metadata={"help": "the architecture", "choices": list(ARCHITECTURES)},
    )


@dataclass
class LossSettings:
    """The loss section: the settings of MultipleNegativesRankingLoss."""

    scale: float = field(
        default=20.0,
        metadata={
            "help": "the factor that multiplies the similarities: the inverse"
            " of the softmax's temperature"
        },
    )


def sections():
    """The config's sections, in order, each with the dataclass of its settings.

    Each architecture has a section of its own, named after it. A run reads
    only its own architecture's section.

    Returns:
        dict[str, type]: Each section's dataclass, by the section's name.
    """
    from sentence_transformers import SentenceTransformerTrainingArguments

    return {
        "data": DataSettings,
        "model": ModelSettings,
        **{name: module.Settings for name, module in ARCHITECTURES.items()},
        "loss": LossSettings,
        "training": SentenceTransformerTrainingArguments,
    }


def flag_names(overrides, setting_names):
    """The settings that a list of flags gives, such as ``learning_rate``.

    HfArgumentParser names each flag after its setting, as
    ``--learning_rate``, and writes ``--no_<setting>`` to turn off a setting
    that is on by default.

    Args:
        overrides (list[str]): The flags and their values.
        setting_names (set[str]): Every setting's name.

    Returns:
        set[str]: The names of the settings that the flags give.
    """
    given_names = set()
    for argument in overrides:
        if not argument.startswith("--"):
            continue
        name = argument[2:].split("=")[0].replace("-", "_")
        if name not in setting_names and name.startswith("no_"):
            name = name[3:]
        given_names.add(name)
    return given_names


def read_config(path, seed, overrides=(), run_name=None):
    """A run's config, read: TRAINING_DEFAULTS, then the config, then the flags.

    It only reads the values and checks them. apply_config then builds the
    settings from them.

    HfArgumentParser has a flag for every setting of every section. Each
    section's settings become the parser's defaults, and the flags given then
    override them. argparse reads a default that is text by the setting's
    type. So ``1e-3``, which YAML reads as text, becomes a number. A flag must
    be written in full: an abbreviation would change a setting that
    config.yaml then does not record. argparse checks a setting's choices
    only for a flag, so the architecture is checked here too. A setting of
    another architecture, in its section or as a flag, is refused, and so is
    a setting of REFUSED_TRAINING. A learning rate that neither the config nor
    a flag gives is the architecture's own.

    Args:
        path (Path): The config, a YAML file.
        seed (int): The seed. It overrides the config's.
        overrides (list[str]): Flags, such as ``["--learning_rate", "1e-2"]``.
        run_name (str | None): The run's name, the trainer's ``run_name`` by
            default. Weights & Biases names the run after it.

    Returns:
        tuple[dict, set[str]]: Every setting's value, by the setting's name,
            and the names of the training settings that the config or the
            flags give.

    Raises:
        ValueError: If the config cannot be read, names a section, a
            setting or an architecture that does not exist, names no
            corpus or no tokenizer, or gives a setting of another
            architecture or of REFUSED_TRAINING, or if a flag is not a
            setting.
        argparse.ArgumentError: If a value does not fit its setting's type.
    """
    from transformers import HfArgumentParser

    section_classes = sections()
    setting_names = {
        section: {setting.name for setting in fields(settings_class) if setting.init}
        for section, settings_class in section_classes.items()
    }
    try:
        config_sections = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception as error:
        raise ValueError(
            f"{path} cannot be read ({type(error).__name__}: {error})"
        ) from error
    if not isinstance(config_sections, dict):
        raise ValueError(f"{path} is not a config: it holds no sections")
    unknown = sorted(set(config_sections) - set(section_classes))
    if unknown:
        raise ValueError(
            f"{path} has no section {', '.join(unknown)}: its sections are"
            f" {', '.join(section_classes)}"
        )
    config_values = {}
    for section, section_values in config_sections.items():
        section_values = section_values or {}
        if not isinstance(section_values, dict):
            raise ValueError(f"{path}: the {section} section holds no settings")
        unknown = sorted(set(section_values) - setting_names[section])
        if unknown:
            homes = [
                f"{name} belongs in the {other} section"
                for name in unknown
                for other, names in setting_names.items()
                if name in names
            ]
            raise ValueError(
                f"{path}: the {section} section has no setting {', '.join(unknown)}"
                + "".join(f"; {home}" for home in homes)
            )
        config_values.update(section_values)
    flags = flag_names(overrides, set().union(*setting_names.values()))
    training_given = (
        set(config_sections.get("training") or {}) | flags
    ) & setting_names["training"]
    refused = sorted(training_given & set(REFUSED_TRAINING))
    if refused:
        raise ValueError(
            f"{path}: the {refused[0]} setting cannot be used:"
            f" {REFUSED_TRAINING[refused[0]]}"
        )

    parser = HfArgumentParser(
        list(section_classes.values()),
        prog="python -m embedding_model train",
        allow_abbrev=False,
        exit_on_error=False,
    )
    parser.set_defaults(**{**TRAINING_DEFAULTS, "run_name": run_name, **config_values})
    parsed, unused = parser.parse_known_args([*overrides, "--seed", str(seed)])
    if unused:
        raise ValueError(f"not a setting: {' '.join(unused)}")
    values = vars(parsed)
    if values["corpus"] is None or values["tokenizer"] is None:
        raise ValueError(
            f"{path}: the data section must name the corpus and the tokenizer"
        )
    architecture = values["architecture"]
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"{path}: there is no architecture {architecture!r}: the"
            f" architectures are {', '.join(ARCHITECTURES)}"
        )
    for other in set(ARCHITECTURES) - {architecture}:
        other_flags = sorted(flags & setting_names[other])
        if other in config_sections or other_flags:
            where = f"--{other_flags[0]}" if other_flags else f"the {other} section"
            raise ValueError(
                f"{path}: {where} is for the {other} architecture, and this"
                f" run's architecture is {architecture}"
            )
    if "learning_rate" not in training_given:
        values["learning_rate"] = ARCHITECTURES[architecture].LEARNING_RATE
    return values, training_given


def apply_config(values):
    """Build each section's settings from the config's values.

    Building the training settings sets up the device: transformers picks
    the GPU, or the CPU when ``use_cpu`` is set or no GPU is visible. The run
    has kept its process to one GPU before, with first_gpu_only.

    Args:
        values (dict): Every setting's value, from read_config.

    Returns:
        dict: Each section's settings, by the section's name: the run's own
            architecture's section, and not the others'.

    Raises:
        ValueError: If transformers refuses a setting, such as bf16 where no
            GPU supports it.
    """
    return {
        section: settings_class(
            **{
                setting.name: values[setting.name]
                for setting in fields(settings_class)
                if setting.init
            }
        )
        for section, settings_class in sections().items()
        if section not in ARCHITECTURES or section == values["architecture"]
    }


def first_gpu_only():
    """Keep this process to one GPU: the first that CUDA_VISIBLE_DEVICES lists, or GPU 0.

    With several GPUs, the trainer would wrap the model in ``nn.DataParallel``.
    That multiplies the batch, and so the loss's in-batch negatives, by the
    number of GPUs. It also cuts each input tensor apart on its own, and
    StaticEmbedding's ``offsets`` then point at tokens on another GPU. So the
    run sets ``CUDA_VISIBLE_DEVICES`` to one GPU, for its own process only.
    CUDA reads the variable when it starts, so this comes first, before torch
    is imported: the command line imports it only when a run reads its
    config. Where the variable is unset, GPU 0 is the first GPU, and no GPU on
    a machine without one.

    Returns:
        list[str]: The GPUs that CUDA_VISIBLE_DEVICES listed, when it listed
            several, and otherwise an empty list.
    """
    listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [part.strip() for part in (listed or "").split(",") if part.strip()]
    if listed is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return []
    if len(ids) <= 1:
        return []
    os.environ["CUDA_VISIBLE_DEVICES"] = ids[0]
    print(
        f"  GPUs {', '.join(ids)} are listed: this run trains on GPU {ids[0]}"
        " only. To choose another, set CUDA_VISIBLE_DEVICES, for example"
        f" CUDA_VISIBLE_DEVICES={ids[1]}"
    )
    return ids


def device_name(training):
    """The device that the run trains on: the GPU's name, or the device's type.

    The type is the one that transformers chose, such as ``cpu``, or ``mps``
    for an Apple GPU.

    Args:
        training (SentenceTransformerTrainingArguments): The training settings.

    Returns:
        str: The device, as the run's manifest holds it.
    """
    if training.device.type != "cuda":
        return training.device.type
    import torch

    return torch.cuda.get_device_name(0)


def tokenizer_note(corpus_name, tokenizer_name, tokenizer_manifest):
    """A note when the tokenizer learned its vocabulary from another corpus.

    Args:
        corpus_name (str): The corpus that the run trains on.
        tokenizer_name (str): The tokenizer's name.
        tokenizer_manifest (dict): The tokenizer's manifest.

    Returns:
        str | None: The note, or None when the tokenizer learned from this
            corpus, or its manifest names no corpus.
    """
    learned_from = {
        source.get("name")
        for source in tokenizer_manifest.get("sources", [])
        if source.get("role") == "corpus"
    }
    if not learned_from or corpus_name in learned_from:
        return None
    return (
        f"  note: tokenizer {tokenizer_name} learned its vocabulary from"
        f" {', '.join(sorted(learned_from))}, not from corpus {corpus_name}"
    )


def read_splits(inputs, corpus_name):
    """A corpus's training and validation pairs, each checked to be there.

    Args:
        inputs (Path): The inputs folder.
        corpus_name (str): The corpus's name.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: The training pairs, the
            validation pairs and the corpus's manifest.

    Raises:
        FileNotFoundError: If no corpus of that name was built.
        ValueError: If the corpus cannot be read, holds no text format of
            FORMATS for its pairs, or has no pairs of a split.
    """
    pairs, corpus_manifest = read_corpus(inputs, corpus_name)
    folder = Path(inputs) / "corpora" / corpus_name
    recorded_format(
        corpus_manifest, folder / "manifest.json", f"run: {build_command(folder)}"
    )
    parts = {split: pairs[pairs.split == split] for split in (TRAIN, VALIDATION)}
    for split, part in parts.items():
        if part.empty:
            raise ValueError(
                f"corpus {corpus_name!r} has no {split} pairs, so no model can be"
                " trained on it"
            )
    return parts[TRAIN], parts[VALIDATION], corpus_manifest


def pairs_dataset(pairs):
    """The pairs' anchors and positives, as a dataset for the trainer.

    Args:
        pairs (pd.DataFrame): The pairs, with ``anchor`` and ``positive``.

    Returns:
        Dataset: ``anchor`` and ``positive``, each text as it is.
    """
    from datasets import Dataset

    return Dataset.from_pandas(pairs[["anchor", "positive"]], preserve_index=False)


def token_ids(pairs, tokenizer):
    """The pairs' anchors and positives as token ids, in a dataset for the trainer.

    By default, StaticEmbedding tokenizes each batch in each training step.
    So every epoch would encode the same texts again, in the dataloader's
    workers, on one thread each. Here each text is encoded once, before
    training, on all the CPU's cores. The workers then only gather the ids.
    The ids have no special tokens, as StaticEmbedding encodes a text. The
    dataset stays in memory.

    Args:
        pairs (pd.DataFrame): The pairs, with ``anchor`` and ``positive``.
        tokenizer (Tokenizer): The tokenizer.

    Returns:
        Dataset: ``anchor`` and ``positive``, each text's token ids.
    """

    def encode(batch):
        """One batch of texts, as token ids."""
        return {
            column: [
                encoding.ids
                for encoding in tokenizer.encode_batch(
                    batch[column], add_special_tokens=False
                )
            ]
            for column in ("anchor", "positive")
        }

    return pairs_dataset(pairs).map(
        encode,
        batched=True,
        batch_size=BATCH_SIZE,
        keep_in_memory=True,
        desc="encoding",
    )


def validation_texts(pairs):
    """The validation pairs' texts, as InformationRetrievalEvaluator takes them.

    InformationRetrievalEvaluator ranks a corpus for each query. It takes
    three dicts: the queries, the corpus and each query's relevant documents,
    by id. The anchors are the queries, and the positives are the corpus. An
    anchor's relevant documents are the positives that it is paired with.

    A text's id is the text as the tokenizer normalises it, so ``BMI`` and
    ``bmi`` are one query, as they are one text to the model. Its first
    spelling is kept. A text that repeats in the pairs is one query, or one
    text of the corpus, and the query's relevant documents are those of every
    spelling.

    Args:
        pairs (pd.DataFrame): The validation pairs, with ``anchor`` and
            ``positive``.

    Returns:
        tuple[dict, dict, dict]: The queries, the corpus and the relevant
            documents.
    """
    keyed = pairs.assign(
        anchor_key=normalised(pairs.anchor), positive_key=normalised(pairs.positive)
    )
    anchors = keyed.drop_duplicates("anchor_key").set_index("anchor_key").anchor
    positives = keyed.drop_duplicates("positive_key").set_index("positive_key")
    relevant = keyed.groupby("anchor_key").positive_key.agg(set).to_dict()
    return anchors.to_dict(), positives.positive.to_dict(), relevant


def fit(settings, tokenizer, training_pairs, validation_pairs):
    """Build the model, train it, and evaluate it at the last step.

    An architecture that reads token ids gets the pairs as ids, made once.
    Another gets the texts, and its model's own ``preprocess`` tokenizes
    each batch.

    Args:
        settings (dict): Each section's settings, from read_config.
        tokenizer (Tokenizer): The tokenizer.
        training_pairs (pd.DataFrame): The training pairs.
        validation_pairs (pd.DataFrame): The validation pairs.

    Returns:
        tuple[SentenceTransformer, SentenceTransformerTrainer, dict, float]:
            The model at the last step, the trainer, its validation metrics
            and the seconds that training took. When the trainer evaluated
            the last step, those metrics are used again, and the model is not
            evaluated twice.
    """
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.sentence_transformer.data_collator import (
        SentenceTransformerDataCollator,
    )
    from sentence_transformers.sentence_transformer.losses import (
        MultipleNegativesRankingLoss,
    )
    from transformers import set_seed

    training = settings["training"]
    architecture_name = settings["model"].architecture
    architecture = ARCHITECTURES[architecture_name]
    # The trainer sets the seed itself, but only after the model is built.
    # Setting it here first makes the seed decide the starting weights too.
    set_seed(training.seed)
    model = architecture.build(settings[architecture_name], tokenizer)
    evaluator = ranking_evaluator(
        model,
        *validation_texts(validation_pairs),
        batch_size=training.per_device_eval_batch_size,
        name="validation",
        cutoff=MRR_CUTOFF,
    )
    # The variable is set on each path, not by default: an earlier run in the
    # same process, or the shell, may have set it otherwise.
    features = getattr(architecture, "features", None)
    if features is None:
        # The dataloader's workers tokenize, in forked processes. The
        # tokenizer's threads do not survive a fork, so a worker that uses
        # them can hang.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        datasets = [
            pairs_dataset(pairs) for pairs in (training_pairs, validation_pairs)
        ]
        collator = None
    else:
        # The ids are made on all the tokenizer's threads. The dataloader's
        # workers then only gather them, so the threads are safe there, and
        # the tokenizer gives no warning in each forked worker.
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        datasets = [
            token_ids(pairs, tokenizer) for pairs in (training_pairs, validation_pairs)
        ]
        collator = SentenceTransformerDataCollator(preprocess_fn=features)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training,
        train_dataset=datasets[0],
        eval_dataset=datasets[1],
        loss=MultipleNegativesRankingLoss(model, scale=settings["loss"].scale),
        evaluator=evaluator,
        data_collator=collator,
    )
    seconds = trainer.train().metrics["train_runtime"]
    evaluated = next(
        (
            entry
            for entry in reversed(trainer.state.log_history)
            if "eval_loss" in entry
        ),
        None,
    )
    if evaluated is None or evaluated["step"] != trainer.state.global_step:
        evaluated = trainer.evaluate()
    return model, trainer, evaluated, seconds


def recorded_config(settings, training_given):
    """The settings that the run's config.yaml holds, by section.

    Every setting of data, model, the run's architecture and loss is
    recorded. Of training's settings, those of TRAINING_DEFAULTS, those that
    the config or the flags gave, the learning rate, the seed and bf16 are
    recorded. The others are the library's defaults, for
    the version that the run's record names. The trainer's ``output_dir`` is
    the run's own, so it is left out. The training settings are written as
    ``TrainingArguments.to_dict`` gives them: an enum as its value, and a
    token hidden.

    Args:
        settings (dict): Each section's settings, from read_config.
        training_given (set[str]): The training settings that the config or
            the flags gave.

    Returns:
        dict: The config, in the sections that ``--config`` reads.
    """
    training_values = settings["training"].to_dict()
    names = (
        set(TRAINING_DEFAULTS) | training_given | {"learning_rate", "seed", "bf16"}
    ) - {"output_dir"}
    return {
        **{
            section: asdict(section_settings)
            for section, section_settings in settings.items()
            if section != "training"
        },
        "training": {name: training_values[name] for name in sorted(names)},
    }


def validation_metrics(evaluated):
    """The validation metrics of the model at the last step.

    Args:
        evaluated (dict): What ``trainer.evaluate()`` returns.

    Returns:
        dict: ``loss``, ``mrr@10`` and accuracy@k, rounded.
    """
    prefix = "eval_validation_cosine_"
    return {
        "loss": round(evaluated["eval_loss"], 4),
        f"mrr@{MRR_CUTOFF}": round(evaluated[f"{prefix}mrr@{MRR_CUTOFF}"], 4),
        **{
            f"accuracy@{k}": round(evaluated[f"{prefix}accuracy@{k}"], 4)
            for k in ACCURACY_AT_K
            if k <= MRR_CUTOFF
        },
    }


def run_manifest(
    *,
    run_name,
    settings,
    corpus_manifest,
    tokenizer_manifest,
    model_folder,
    counts,
    validation,
    device,
    commit,
):
    """The run's record, for ``manifest.json``.

    It names the corpus and the tokenizer with their checksums, and the
    benchmarks that the corpus was checked against for leakage. ``score``
    reads them. It also holds the text format of the pairs, from the
    corpus's manifest, the device, the code's commit and the libraries'
    versions.

    Args:
        run_name (str): The run's name.
        settings (dict): Each section's settings, from read_config.
        corpus_manifest (dict): The corpus's manifest.
        tokenizer_manifest (dict): The tokenizer's manifest.
        model_folder (Path): The saved model.
        counts (dict): The pairs, steps, epochs and seconds.
        validation (dict): The validation metrics at the last step.
        device (str): ``cpu``, or the GPU's name.
        commit (dict): The commit of the code that trained the model, from
            ``code_commit``, read when the run started.

    Returns:
        dict: The manifest.
    """
    return {
        "run": run_name,
        "built": date.today().isoformat(),
        "files": {
            f"model/{path.relative_to(model_folder).as_posix()}": {
                "sha256": file_sha256(path)
            }
            for path in sorted(model_folder.rglob("*"))
            if path.is_file()
        },
        "sources": [
            {
                "role": "corpus",
                "name": settings["data"].corpus,
                "files": {"pairs.parquet": corpus_manifest["files"]["pairs.parquet"]},
                "benchmarks": [
                    {
                        "name": source["name"],
                        "version": source["version"],
                        "files": source.get("files", {}),
                    }
                    for source in corpus_manifest.get("sources", [])
                    if source.get("role") == "benchmark"
                ],
            },
            {
                "role": "tokenizer",
                "name": settings["data"].tokenizer,
                "files": {
                    "tokenizer.json": tokenizer_manifest["files"]["tokenizer.json"]
                },
            },
        ],
        "settings": {
            "architecture": settings["model"].architecture,
            "format": corpus_manifest["settings"]["format"],
        },
        "counts": counts,
        "validation": validation,
        "device": device,
        "ai_harmonization": commit,
        "libraries": library_versions(),
    }


def train(config, seed, overrides, inputs, outputs):
    """Train one model, and write its run's folder.

    The run is written to a staging folder first. It replaces the previous run
    of that name only once it is complete.

    Args:
        config (Path): The config, a YAML file.
        seed (int): The seed.
        overrides (list[str]): Flags that override the config's settings.
        inputs (Path): The inputs folder, with the corpus and the tokenizer.
        outputs (Path): The outputs folder. The run goes to its
            ``runs/<config>_s<seed>/``.

    Returns:
        int: 0 if the run was written, 1 if the seed, the config, the
            corpus, the tokenizer or the GPUs stop it.

    Raises:
        Exception: If training fails or the run cannot be written. The
            previous run of that name is then kept.
    """
    run_name = f"{Path(config).stem}_s{seed}"
    if not FOLDER_NAME.fullmatch(run_name):
        print(
            f"{run_name!r} is not a run name: use {FOLDER_NAME_RULE}", file=sys.stderr
        )
        return 1
    if seed < 0:
        print(f"--seed must be 0 or more, not {seed}", file=sys.stderr)
        return 1
    first_gpu_only()
    try:
        values, training_given = read_config(config, seed, overrides, run_name)
        training_pairs, validation_pairs, corpus_manifest = read_splits(
            inputs, values["corpus"]
        )
        tokenizer, tokenizer_manifest = read_tokenizer(inputs, values["tokenizer"])
        settings = apply_config(values)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    data, training = settings["data"], settings["training"]
    device = device_name(training)
    print(
        f"  run {run_name}: {settings['model'].architecture} model, corpus"
        f" {data.corpus}, tokenizer {data.tokenizer}, on {device}\n"
        f"  {len(training_pairs):,} training pairs,"
        f" {len(validation_pairs):,} validation pairs"
    )
    note = tokenizer_note(data.corpus, data.tokenizer, tokenizer_manifest)
    if note:
        print(note)

    # Read before training, so that a change to the checkout during a long run
    # does not change the commit that the run writes.
    commit = code_commit()
    destination = Path(outputs) / RUNS / run_name
    with staged(destination) as built:
        training.output_dir = str(built / "trainer")
        model, trainer, evaluated, seconds = fit(
            settings, tokenizer, training_pairs, validation_pairs
        )
        shutil.rmtree(built / "trainer", ignore_errors=True)
        model.save_pretrained(str(built / "model"))
        trainer.state.save_to_json(str(built / "trainer_state.json"))
        (built / "config.yaml").write_text(
            yaml.safe_dump(recorded_config(settings, training_given), sort_keys=False),
            encoding="utf-8",
        )
        validation = validation_metrics(evaluated)
        counts = {
            "training_pairs": len(training_pairs),
            "validation_pairs": len(validation_pairs),
            "steps": trainer.state.global_step,
            "epochs": round(trainer.state.epoch, 3),
            "training_seconds": round(seconds, 1),
        }
        manifests.write(
            built / "manifest.json",
            run_manifest(
                run_name=run_name,
                settings=settings,
                corpus_manifest=corpus_manifest,
                tokenizer_manifest=tokenizer_manifest,
                model_folder=built / "model",
                counts=counts,
                validation=validation,
                device=device,
                commit=commit,
            ),
        )
    print(
        f"  {counts['steps']:,} steps in {seconds:.0f} s -> {destination}\n"
        "  on the validation pairs: "
        + ", ".join(f"{metric} {value}" for metric, value in validation.items())
    )
    return 0
