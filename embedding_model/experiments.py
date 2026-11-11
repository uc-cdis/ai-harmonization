"""Experiments: a base config and its arms, each trained at several seeds.

An experiment asks one question: does a change to the base help? Its file
gives the base, the arms, the seeds and the rule that decides. ``experiment
train`` trains each run that the file declares, on the machine where it
runs: one run per visible GPU at a time, or one at a time on the CPU. Each
run is the public command ``python -m embedding_model train``, in a process
of its own. ``experiment status`` says which runs are done.
"""

import os
import queue
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from embedding_model import manifests
from embedding_model.corpus import build_command
from embedding_model.folders import (
    FOLDER_NAME,
    FOLDER_NAME_RULE,
    STAGING_MARK,
    staged,
    staging_folders,
)
from embedding_model.tokenizer import read_tokenizer
from embedding_model.train import RUNS, listed_gpus, read_config

# Where experiments are written, under the outputs folder.
EXPERIMENTS = "experiments"
# The copy of the experiment's file that its first training keeps, and the
# folders beside it: each arm's full config, and each run's log.
EXPERIMENT_FILE = "experiment.yaml"
CONFIGS = "configs"
LOGS = "logs"
# The arm that each other arm is measured against. No arm of a file takes its
# name.
BASE = "base"
# What follows the base or an arm that sets training.seed.
SEED_SET = (
    "sets training.seed. Remove it: each run's seed comes from the experiment's"
    " seeds field"
)
# The states of a run, from its folder.
DONE = "done"
STARTED = "started"
NOT_STARTED = "not started"


@dataclass(frozen=True)
class Run:
    """One run of an experiment: an arm at a seed.

    Attributes:
        arm (str): The arm, or BASE.
        seed (int): The seed.
        folder (Path): The run's folder. A base run that an earlier experiment
            trained keeps its folder there.
        config (Path): The arm's full config, which trains the run.
    """

    arm: str
    seed: int
    folder: Path
    config: Path


class Decision(BaseModel):
    """The rule that decides between the arms, with its defaults.

    The validation pairs decide, by MRR@10. A difference under 0.01 is not
    worth deciding on, and 0.05 is split over the arms. The metric can be any
    of the validation metrics that a run's manifest records.
    """

    model_config = ConfigDict(extra="forbid")

    scored_on: Literal["validation"] = "validation"
    metric: Literal[
        "mrr@10", "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10"
    ] = "mrr@10"
    minimum_effect: float = Field(0.01, gt=0, lt=1)
    alpha: float = Field(0.05, gt=0, lt=1)


class Arm(BaseModel):
    """An arm: its test, and the settings that it changes, by section.

    The test asks whether the arm is better than the base, or no worse than
    it by more than the minimum effect.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, dict[str, Any] | None]

    test: Literal["superiority", "non-inferiority"]

    @model_validator(mode="after")
    def check_settings(self):
        """Refuse an arm that changes the corpus or sets the seed.

        Returns:
            Arm: The arm.

        Raises:
            ValueError: If the arm sets ``data.corpus`` or ``training.seed``.
        """
        sections = self.model_extra
        if "corpus" in (sections.get("data") or {}):
            raise ValueError(
                "the arm changes data.corpus, so it would be scored on other"
                " validation pairs than the base. Remove it, or start another"
                " experiment on that corpus"
            )
        if "seed" in (sections.get("training") or {}):
            raise ValueError(f"the arm {SEED_SET}")
        return self


class Experiment(BaseModel):
    """An experiment's file: its question, base, seeds, decision rule and arms."""

    model_config = ConfigDict(extra="forbid")

    question: str | None = None
    base: dict[str, dict[str, Any] | None] | str
    seeds: list[Annotated[StrictInt, Field(ge=0)]] = Field(min_length=1)
    decision: Decision = Field(default_factory=Decision)
    arms: dict[str, Arm] = Field(min_length=1)

    @field_validator("decision", mode="before")
    @classmethod
    def empty_decision_is_the_default(cls, decision):
        """Read an empty decision, which YAML reads as None, as the default rule.

        train reads an empty section of a run's config as empty in the same way.

        Returns:
            The decision, or ``{}`` when it is empty.
        """
        return {} if decision is None else decision

    @field_validator("base", mode="before")
    @classmethod
    def check_base(cls, base):
        """Check the base's form: the settings by section, or an earlier experiment's arm.

        Returns:
            dict | str: The base.

        Raises:
            ValueError: If the base is neither form, or sets ``training.seed``.
        """
        if isinstance(base, str):
            parts = base.split("/")
            if len(parts) != 2 or not all(
                FOLDER_NAME.fullmatch(part) for part in parts
            ):
                raise ValueError(
                    f"{base!r} is not an arm of an earlier experiment, which is"
                    " written <experiment>/<arm>, such as dim/dim128"
                )
        elif not isinstance(base, dict) or not all(
            section is None or isinstance(section, dict) for section in base.values()
        ):
            raise ValueError(
                "it is neither the base's settings, by section, such as data: and"
                " training:, nor an arm of an earlier experiment, written"
                " <experiment>/<arm>"
            )
        elif "seed" in (base.get("training") or {}):
            raise ValueError(f"the base {SEED_SET}")
        return base

    @field_validator("seeds")
    @classmethod
    def check_seeds(cls, seeds):
        """Check that the seeds all differ.

        Returns:
            list[int]: The seeds.

        Raises:
            ValueError: If two seeds are the same.
        """
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"{seeds} lists a seed twice. List each seed once, such as"
                " [0, 1, 2, 3, 4]"
            )
        return seeds

    @field_validator("arms")
    @classmethod
    def check_arm_names(cls, arms):
        """Check that each arm's name is a folder's name, and not the base's.

        Returns:
            dict[str, Arm]: The arms.

        Raises:
            ValueError: If a name cannot be an arm's.
        """
        for arm in arms:
            if not FOLDER_NAME.fullmatch(arm) or arm == BASE:
                raise ValueError(
                    f"{arm!r} is not an arm's name: use {FOLDER_NAME_RULE}, and not"
                    f" {BASE}, which is the base's"
                )
        return arms


def problems(error):
    """The problems that pydantic found in a file, each with its location.

    Args:
        error (ValidationError): pydantic's error.

    Returns:
        list[str]: One line per problem, as ``<location>: <what is wrong>``.
            The location is pydantic's path to the value in the file, such as
            ``seeds.1`` for the second seed, or ``arms.dim8.test``.
    """
    # The model at each location whose fields are fixed: the file itself, and
    # its decision. An arm takes any section, and train checks the sections.
    models = {(): Experiment, ("decision",): Decision}
    lines = []
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"])
        if problem["type"] == "value_error":
            lines.append(f"{location}: {problem['ctx']['error']}")
        elif problem["type"] == "missing":
            lines.append(f"{location} is missing")
        elif problem["type"] == "extra_forbidden":
            *parent, unknown_field = problem["loc"]
            field_names = ", ".join(models[tuple(parent)].model_fields)
            lines.append(
                f"{'.'.join(parent) or 'the file'} has no field {unknown_field}: its fields"
                f" are {field_names}"
            )
        elif problem["type"] in ("too_short", "too_long"):
            lines.append(f"{location}: {problem['msg']}")
        else:
            lines.append(f"{location}: {problem['msg']}, not {problem['input']!r}")
    return lines


def read_experiment(path):
    """An experiment's file, read and checked, with the decision rule's defaults.

    pydantic checks the file against Experiment: each field, its type and its
    values. Experiment's validators add the rules of an experiment.

    Args:
        path (Path): The experiment's file, a YAML file. The experiment is
            named after it, as a run is named after its config.

    Returns:
        dict: The experiment: its ``name``, its ``question`` (None when the
            file has none), and the file's ``base``, ``seeds``, ``decision``,
            with every field, and ``arms``.

    Raises:
        ValueError: If the file cannot be read, its name cannot be an
            experiment's, or a field is missing, unknown or holds a value
            that cannot be used.
    """
    path = Path(path)
    if not FOLDER_NAME.fullmatch(path.stem):
        raise ValueError(
            f"{path.stem!r} is not an experiment name, from the file's name:"
            f" use {FOLDER_NAME_RULE}"
        )
    try:
        fields = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(
            f"{path} cannot be read ({type(error).__name__}: {error})"
        ) from error
    if not isinstance(fields, dict):
        raise ValueError(
            f"{path} holds no fields. An experiment's file has at least base,"
            " seeds and arms"
        )
    try:
        experiment = Experiment.model_validate(fields)
    except ValidationError as error:
        raise ValueError(f"{path}: {'; '.join(problems(error))}") from error
    return {"name": path.stem, **experiment.model_dump()}


def checked_experiment(path, outputs):
    """An experiment's file, read, and checked against the copy that its first training kept.

    The copy holds the file as its runs were trained from. A file of the same
    name that reads otherwise, in any field, would mix the runs of two
    questions in one folder, so it is refused. Both are compared as
    read_experiment reads them, with the decision rule's defaults. So a
    comment, the order of the fields, or a default written out changes
    nothing.

    Args:
        path (Path): The experiment's file.
        outputs (Path): The outputs folder.

    Returns:
        tuple[dict, Path]: The experiment, from read_experiment, and its
            folder, ``outputs/experiments/<name>/``.

    Raises:
        ValueError: If the file cannot be used, as read_experiment says, or
            it differs from the copy.
    """
    experiment = read_experiment(path)
    name = experiment["name"]
    folder = Path(outputs) / EXPERIMENTS / name
    copy = folder / EXPERIMENT_FILE
    if not copy.exists():
        return experiment, folder
    next_steps = (
        f"To start it again, delete {folder}. To ask a new question, give the"
        f" file a new name; its base can reuse these runs, as base: {name}/{BASE}"
    )
    try:
        trained_experiment = read_experiment(copy)
    except Exception as error:
        raise ValueError(
            f"{copy}, the file that experiment {name} was trained from, cannot"
            f" be read ({error}). {next_steps}"
        ) from error
    changed_fields = [
        field
        for field in experiment
        if field != "name" and experiment[field] != trained_experiment[field]
    ]
    if changed_fields:
        raise ValueError(
            f"{path} differs from {copy}, the file that experiment {name} was"
            f" trained from, in its {', '.join(changed_fields)}. {next_steps}"
        )
    return experiment, folder


def base_config(base, outputs):
    """The base's config, by section, and the earlier experiment's arm that it comes from.

    Args:
        base (dict | str): The file's base: the settings themselves, or an
            earlier experiment's arm, as ``<experiment>/<arm>``.
        outputs (Path): The outputs folder.

    Returns:
        tuple[dict, tuple[Path, str] | None]: The base's sections; and the
            earlier experiment's folder and arm, or None for a base that the
            file gives itself.

    Raises:
        ValueError: If the earlier experiment has not been trained, or has no
            such arm.
    """
    if isinstance(base, dict):
        return base, None
    earlier_name, arm = base.split("/")
    earlier_folder = Path(outputs) / EXPERIMENTS / earlier_name
    config_path = earlier_folder / CONFIGS / f"{arm}.yaml"
    if not config_path.exists():
        earlier_arms = sorted(
            path.stem for path in (earlier_folder / CONFIGS).glob("*.yaml")
        )
        if not earlier_arms:
            raise ValueError(
                f"base: no experiment {earlier_name} has been trained in {earlier_folder.parent}"
            )
        raise ValueError(
            f"base: experiment {earlier_name} has no arm {arm}; its arms are"
            f" {', '.join(earlier_arms)}"
        )
    return yaml.safe_load(config_path.read_text(encoding="utf-8")), (
        earlier_folder,
        arm,
    )


def arm_configs(experiment, base_sections):
    """Each arm's full config, by section: the base's, with the arm's changes.

    Args:
        experiment (dict): The experiment, from read_experiment.
        base_sections (dict): The base's config, by section.

    Returns:
        dict[str, dict]: Each config, by arm, the base's first, under BASE.
    """
    configs = {BASE: base_sections}
    for arm, arm_fields in experiment["arms"].items():
        sections = {
            section: dict(values or {}) for section, values in base_sections.items()
        }
        for section, values in arm_fields.items():
            if section != "test":
                sections.setdefault(section, {}).update(values or {})
        configs[arm] = sections
    return configs


def written_configs(configs, folder, seed):
    """Write each arm's full config, and check it as ``train`` reads it.

    The configs are written whole, to ``configs/`` in the experiment's
    folder. Each one is then read by ``train``'s own parser, so a setting
    that ``train`` would refuse stops the experiment before any run trains.

    Args:
        configs (dict[str, dict]): Each arm's full config, from arm_configs.
        folder (Path): The experiment's folder.
        seed (int): A seed of the experiment, which the parser needs.

    Returns:
        dict[str, dict]: Each arm's setting values, as ``train``'s read_config
            reads them, by arm.

    Raises:
        ValueError: If ``train`` refuses a config.
    """
    with staged(folder / CONFIGS) as built:
        for arm, sections in configs.items():
            (built / f"{arm}.yaml").write_text(
                yaml.safe_dump(sections, sort_keys=False), encoding="utf-8"
            )
    return {
        arm: read_config(folder / CONFIGS / f"{arm}.yaml", seed)[0] for arm in configs
    }


def check_inputs(arm_values, inputs):
    """Check that the corpus and each tokenizer that the arms name are built.

    Args:
        arm_values (dict[str, dict]): Each arm's setting values, from
            written_configs.
        inputs (Path): The inputs folder.

    Raises:
        FileNotFoundError: If the corpus or a tokenizer is not built. The
            message gives the command that builds it.
        ValueError: If its manifest cannot be read, or the tokenizer differs
            from its manifest.
    """
    corpus_name = arm_values[BASE]["corpus"]
    corpus_folder = Path(inputs) / "corpora" / corpus_name
    manifests.read(
        corpus_folder / "manifest.json",
        build_command(corpus_folder),
        "corpus",
        corpus_name,
    )
    for tokenizer_name in sorted(
        {values["tokenizer"] for values in arm_values.values()}
    ):
        read_tokenizer(inputs, tokenizer_name)


def run_state(run_folder):
    """Whether a run is done, has started, or has not started.

    A run that ``train`` finished has its ``manifest.json``. One that is
    training has a staging folder beside its folder, and so does one that
    was stopped before it finished.

    Args:
        run_folder (Path): The run's folder.

    Returns:
        str: DONE, STARTED or NOT_STARTED.
    """
    if (run_folder / "manifest.json").exists():
        return DONE
    if staging_folders(run_folder):
        return STARTED
    return NOT_STARTED


def experiment_runs(experiment, folder, earlier_arm):
    """Every run that the experiment needs, the base's first, each arm's in seed order.

    A base run comes from the earlier experiment when that experiment has
    it, done or started, and this experiment has not trained it itself.

    Args:
        experiment (dict): The experiment, from read_experiment.
        folder (Path): The experiment's folder.
        earlier_arm (tuple[Path, str] | None): The earlier experiment's arm
            that the base is, as that experiment's folder and the arm's name,
            from base_config, or None.

    Returns:
        list[Run]: The runs.
    """
    runs = []
    for arm in [BASE, *experiment["arms"]]:
        for seed in experiment["seeds"]:
            run_folder = folder / RUNS / f"{arm}_s{seed}"
            if arm == BASE and earlier_arm and run_state(run_folder) == NOT_STARTED:
                earlier_folder, earlier_arm_name = earlier_arm
                earlier_run = earlier_folder / RUNS / f"{earlier_arm_name}_s{seed}"
                if run_state(earlier_run) != NOT_STARTED:
                    run_folder = earlier_run
            runs.append(Run(arm, seed, run_folder, folder / CONFIGS / f"{arm}.yaml"))
    return runs


def gpu_ids():
    """The CUDA GPUs that the runs can train on, by the ids that CUDA_VISIBLE_DEVICES takes.

    CUDA_VISIBLE_DEVICES, when it is set, lists them, as SLURM sets it for a
    job's GPUs. Otherwise every GPU that torch counts is one.

    Returns:
        list[str]: Each GPU's id, or an empty list when there is none.
    """
    listed = listed_gpus()
    if listed is not None:
        return listed
    import torch

    return [str(index) for index in range(torch.cuda.device_count())]


def train_command(run, inputs, folder):
    """The public command that trains one run, as a list of arguments.

    Args:
        run (Run): The run.
        inputs (Path): The inputs folder.
        folder (Path): The experiment's folder, which holds the runs.

    Returns:
        list[str]: The command, with the Python that runs this one.
    """
    return [
        sys.executable,
        "-m",
        "embedding_model",
        "train",
        "--config",
        str(run.config),
        "--seed",
        str(run.seed),
        "--inputs",
        str(inputs),
        "--outputs",
        str(folder),
    ]


def train_runs(runs, gpus, inputs, folder):
    """Train each run in a process of its own, one per GPU at a time, or one at a time without a GPU.

    Each process sees one GPU, through CUDA_VISIBLE_DEVICES. Without a CUDA
    GPU, its environment is left as it is, and ``train`` picks the device, as
    it does on its own. Each process writes its output to
    ``logs/<run>.log``. A run that fails does not stop the others.

    Args:
        runs (list[Run]): The runs to train.
        gpus (list[str]): The GPUs' ids, from gpu_ids, or an empty list.
        inputs (Path): The inputs folder.
        folder (Path): The experiment's folder.

    Returns:
        list[Run]: The runs that failed.
    """
    # The ids of the GPUs that no run is training on. Without a GPU, one empty
    # id lets one run train at a time.
    free_gpus = queue.Queue()
    for gpu in gpus or [""]:
        free_gpus.put(gpu)
    (folder / LOGS).mkdir(parents=True, exist_ok=True)

    def train_one(run):
        """Train one run on a free GPU, or without one, and say whether it succeeded."""
        gpu = free_gpus.get()
        run_name = run.folder.name
        log_path = folder / LOGS / f"{run_name}.log"
        try:
            on_gpu = f" on GPU {gpu}" if gpu else ""
            print(f"  {run_name}: training{on_gpu} -> {log_path}", flush=True)
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu} if gpu else None
            with open(log_path, "w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    train_command(run, inputs, folder),
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            succeeded = completed.returncode == 0
            outcome = "done" if succeeded else f"failed; see {log_path}"
            print(f"  {run_name}: {outcome}", flush=True)
            return succeeded
        finally:
            free_gpus.put(gpu)

    executor = ThreadPoolExecutor(max_workers=max(len(gpus), 1))
    try:
        successes = list(executor.map(train_one, runs))
    except KeyboardInterrupt:
        # Ctrl-C stops the runs that are training, which share the terminal.
        # The runs still waiting are cancelled, so none starts after it.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    return [run for run, success in zip(runs, successes) if not success]


def keep_copy(path, folder):
    """Copy the experiment's file to experiment.yaml in the experiment's folder, at its first training.

    The experiment's folder is ``outputs/experiments/<name>/``. A later
    training compares the file with this copy, in checked_experiment. The
    copy is written in a staging folder first and then moved, so it is never
    half written, as staged writes a folder. shutil.copyfile creates it as
    any new file, so it is as readable as the run folders beside it.

    Args:
        path (Path): The experiment's file.
        folder (Path): The experiment's folder.
    """
    copy = folder / EXPERIMENT_FILE
    if copy.exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=folder, prefix=f".{EXPERIMENT_FILE}{STAGING_MARK}"
    ) as staging:
        staged_copy = Path(staging) / EXPERIMENT_FILE
        shutil.copyfile(path, staged_copy)
        os.replace(staged_copy, copy)


def checked_runs(path, inputs, outputs):
    """An experiment's runs, after every check that comes before training.

    Args:
        path (Path): The experiment's file.
        inputs (Path): The inputs folder.
        outputs (Path): The outputs folder.

    Returns:
        tuple[dict, Path, list[Run]]: The experiment, its folder and its runs.

    Raises:
        ValueError: If the file, the base or a config cannot be used, or a
            base run of an earlier experiment has started but not finished.
        FileNotFoundError: If the corpus or a tokenizer is not built.
    """
    experiment, folder = checked_experiment(path, outputs)
    base_sections, earlier_arm = base_config(experiment["base"], outputs)
    arm_values = written_configs(
        arm_configs(experiment, base_sections), folder, experiment["seeds"][0]
    )
    check_inputs(arm_values, inputs)
    runs = experiment_runs(experiment, folder, earlier_arm)
    for run in runs:
        if run.folder.parent != folder / RUNS and run_state(run.folder) == STARTED:
            raise ValueError(
                f"the base run {run.folder} of experiment {run.folder.parent.parent.name}"
                " has started but not finished. Wait for it to finish. If no job is"
                f" training it, delete its staging folder,"
                f" {staging_folders(run.folder)[0]}"
            )
    return experiment, folder, runs


def train(path, inputs, outputs):
    """Train each run of an experiment that is not done, and keep its file.

    Every check comes before the first run trains: the file, the base, each
    arm's config, the corpus and the tokenizers.

    Args:
        path (Path): The experiment's file.
        inputs (Path): The inputs folder.
        outputs (Path): The outputs folder. The experiment goes to its
            ``experiments/<name>/``.

    Returns:
        int: 0 if every run is done, 1 if a run failed.

    Raises:
        ValueError: If a check fails, as checked_runs says.
        FileNotFoundError: If the corpus or a tokenizer is not built.
    """
    experiment, folder, runs = checked_runs(path, inputs, outputs)
    keep_copy(path, folder)
    to_train = [run for run in runs if run_state(run.folder) != DONE]
    gpus = gpu_ids()
    arm_count = len(experiment["arms"])
    seed_count = len(experiment["seeds"])
    print(
        f"experiment {experiment['name']}: the base and {arm_count} arm"
        f"{'s' if arm_count > 1 else ''} at {seed_count} seed{'s' if seed_count > 1 else ''},"
        f" {len(runs)} runs: {len(runs) - len(to_train)} done,"
        f" {len(to_train)} to train"
    )
    if experiment["question"]:
        print(f"  {experiment['question']}")
    if not to_train:
        print(f"  every run is done -> {folder}")
        return 0
    print(
        f"  {len(gpus)} GPU{'s' if len(gpus) > 1 else ''}:"
        f" {len(gpus)} run{'s' if len(gpus) > 1 else ''} at a time"
        if gpus
        else "  no CUDA GPU: one run at a time"
    )
    failed = train_runs(to_train, gpus, inputs, folder)
    if failed:
        print(
            f"  {len(failed)} of {len(to_train)} runs failed: "
            + ", ".join(run.folder.name for run in failed)
            + f". Their logs are in {folder / LOGS}. Train the experiment again"
            " to retry them.",
            file=sys.stderr,
        )
        return 1
    print(f"  {len(to_train)} runs trained -> {folder}")
    return 0


def status(path, outputs):
    """Print which of an experiment's runs are done, started or not started.

    Args:
        path (Path): The experiment's file.
        outputs (Path): The outputs folder.

    Returns:
        int: 0.

    Raises:
        ValueError: If the file or the base cannot be used, or the file
            differs from the copy that its first training kept.
    """
    experiment, folder = checked_experiment(path, outputs)
    _, earlier_arm = base_config(experiment["base"], outputs)
    runs = experiment_runs(experiment, folder, earlier_arm)
    states = {run: run_state(run.folder) for run in runs}
    done_count = sum(state == DONE for state in states.values())
    print(f"experiment {experiment['name']}: {done_count} of {len(runs)} runs done")
    arm_names = [BASE, *experiment["arms"]]
    # Each name is padded to the longest, so that the counts line up.
    name_width = max(len(arm) for arm in arm_names) + 2
    for arm in arm_names:
        state_counts = [
            f"{count} {state}"
            for state in (DONE, STARTED, NOT_STARTED)
            if (count := sum(run.arm == arm and states[run] == state for run in runs))
        ]
        print(f"  {arm:<{name_width}}{', '.join(state_counts)}")
    return 0
