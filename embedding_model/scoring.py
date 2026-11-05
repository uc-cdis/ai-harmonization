"""Score a model on a benchmark: where it ranks each variable's expected targets.

For each source variable, the model ranks every target of the benchmark by
cosine similarity. sentence-transformers' ``InformationRetrievalEvaluator`` encodes
the texts, ranks the targets and computes MRR and accuracy@k.

A variable with several expected targets takes the rank of the best placed of
them. accuracy@k is the share of variables with an expected target among the
first k. It is what a reviewer of the top k suggestions sees.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ai_harmonization.formatters import VALUE_SEPARATOR
from embedding_model.benchmarks import TARGET_SEPARATOR, read_benchmark
from embedding_model.corpus import build_command
from embedding_model.folders import staged
from embedding_model.manifests import differences, file_sha256
from embedding_model.provenance import code_commit, library_versions, model_revision
from embedding_model.texts import (
    DEFAULT_FORMAT,
    FORMATS,
    property_text,
    recorded_format,
)

ACCURACY_AT_K = (1, 3, 5, 10)
# The cutoff of MRR@10, the evaluator's default. Training reports MRR@10 on
# the validation pairs, and score reports it beside the full MRR, so the two
# compare.
MRR_CUTOFF = 10
METRIC_COLUMNS = [
    "mrr",
    f"mrr@{MRR_CUTOFF}",
    *[f"accuracy@{k}" for k in ACCURACY_AT_K],
    "median_rank",
]
# sentence-transformers' own default, in encode() and in its
# InformationRetrievalEvaluator.
BATCH_SIZE = 32
# The fields of a run's record that scoring reads: the run's name, its model's
# checksums, its corpus's leakage check, and its text format.
RUN_FIELDS = ("run", "files", "sources", "settings")


def split_values(cell):
    """A benchmark cell's values as a list, or None when it has none."""
    return [value for value in cell.split(VALUE_SEPARATOR) if value] or None


def benchmark_texts(targets, mappings, text_format):
    """A benchmark's texts, in the three dicts InformationRetrievalEvaluator takes.

    Args:
        targets (pd.DataFrame): The benchmark's targets.
        mappings (pd.DataFrame): The benchmark's mappings.
        text_format (str): A key of FORMATS.

    Returns:
        tuple[dict, dict, dict]: The evaluator's queries, corpus and relevant
        documents:

        - source_texts: each source variable's text, by its row number in the
          mappings;
        - target_texts: each target property's text, by its ``node.property``
          key;
        - expected: each source variable's expected ``node.property`` keys, by
          its row number.
    """
    formatter = FORMATS[text_format]
    target_texts = {}
    for row in targets.itertuples():
        node, _, name = row.target_property.rpartition(".")
        target_texts[row.target_property] = property_text(
            formatter,
            node,
            name,
            row.target_type,
            row.target_description,
            split_values(row.target_values),
        )
    source_texts, expected = {}, {}
    for number, row in enumerate(mappings.itertuples()):
        source_texts[str(number)] = property_text(
            formatter,
            "",
            row.source_variable_name,
            row.source_type,
            row.source_description,
            split_values(row.source_values),
        )
        expected[str(number)] = set(row.expected_target.split(TARGET_SEPARATOR))
    return source_texts, target_texts, expected


def ranking_evaluator(
    model,
    queries,
    corpus,
    relevant,
    batch_size=BATCH_SIZE,
    name="",
    write_predictions=False,
    cutoff=None,
):
    """An InformationRetrievalEvaluator that ranks as the approaches rank.

    Every text of the corpus is ranked for each query, by cosine similarity.
    Scoring ranks a benchmark's targets with it. MRR is then computed over the
    whole ranking, and MRR@10 beside it. Training ranks the validation pairs'
    positives, with a cutoff: the evaluator keeps each query's hits in Python
    lists, so a whole ranking of tens of thousands of positives for each of
    tens of thousands of queries would not fit.

    Args:
        model (SentenceTransformer): The model, for its default prompt.
        queries (dict[str, str]): The query texts, by id.
        corpus (dict[str, str]): The texts to rank, by id.
        relevant (dict[str, set[str]]): Each query's relevant corpus ids.
        batch_size (int): Texts encoded at a time.
        name (str): The evaluator's name, which starts the names of its
            metrics, such as ``validation_cosine_mrr@<corpus size>``.
        write_predictions (bool): Whether the evaluator writes each query's
            full ranking when it is given a folder.
        cutoff (int | None): The last rank that the metrics count: MRR@cutoff,
            and the evaluator's other metrics at most at the cutoff. None
            counts the whole ranking, for MRR and MRR@10.

    Returns:
        InformationRetrievalEvaluator: The evaluator.
    """
    # The approaches embed both sides with model.encode(), which adds only the
    # model's default prompt, if its config names one. Left alone, the
    # evaluator would add the model's "query" and "document" prompts instead.
    #
    # The approaches rank by cosine similarity, even when a model's config
    # names another similarity function. So the evaluator ranks by cosine too.
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )
    from sentence_transformers.util import cos_sim

    prompt = model.prompts.get(model.default_prompt_name, "")
    # The evaluator also computes precision, recall, nDCG and MAP, at its own
    # default cutoffs. With a cutoff, none of them looks further.
    limits = {}
    if cutoff is not None:
        limits = {
            "precision_recall_at_k": [cutoff],
            "ndcg_at_k": [cutoff],
            "map_at_k": [cutoff],
        }
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        query_prompt=prompt,
        corpus_prompt=prompt,
        score_functions={"cosine": cos_sim},
        mrr_at_k=[cutoff] if cutoff else sorted({MRR_CUTOFF, len(corpus)}),
        accuracy_at_k=[k for k in ACCURACY_AT_K if cutoff is None or k <= cutoff],
        **limits,
        batch_size=batch_size,
        name=name,
        show_progress_bar=False,
        write_csv=False,
        write_predictions=write_predictions,
    )


def rank_targets(model, source_texts, target_texts, expected, batch_size=BATCH_SIZE):
    """Rank every target for each source, and score the ranks.

    Targets are ranked by cosine similarity. The evaluator gives the results
    two ways. It returns the averages in memory: MRR, MRR@10 and accuracy@k
    over all source variables. Its predictions file, written to a temporary folder,
    holds each source variable's full ranking. Each source variable's rank and
    top-1 target are read from that file, and the folder is deleted.

    Args:
        model (SentenceTransformer): The model to score.
        source_texts (dict[str, str]): The source variables' texts, by id.
        target_texts (dict[str, str]): The target properties' texts, by key.
        expected (dict[str, set[str]]): Each source variable's expected
            target keys.
        batch_size (int): Texts encoded at a time.

    Returns:
        tuple[dict, dict]: The metrics (METRIC_COLUMNS), and for each source
            id its rank, its top-1 target and that target's similarity.
    """
    evaluator = ranking_evaluator(
        model, source_texts, target_texts, expected, batch_size, write_predictions=True
    )
    with tempfile.TemporaryDirectory() as folder:
        scores = evaluator(model, output_path=folder)
        # The evaluator writes one predictions file per score function.
        files = list(Path(folder).glob("*_predictions_cosine.jsonl"))
        if len(files) != 1:
            raise RuntimeError(f"the evaluator wrote {len(files)} predictions files")
        lines = files[0].read_text(encoding="utf-8").splitlines()
        ranked = {}
        for entry in map(json.loads, lines):
            query, results = entry["query_id"], entry["results"]
            for position, hit in enumerate(results):
                if hit["corpus_id"] in expected[query]:
                    break
            else:
                raise RuntimeError(
                    f"no expected target in the ranking of source {query}"
                )
            ranked[query] = {
                "rank": position + 1,
                "top1_target": results[0]["corpus_id"],
                "top1_similarity": round(results[0]["score"], 4),
            }

    # The evaluator names each metric after its score function, as in
    # ``cosine_mrr@<number of targets>``.
    metrics = {
        "mrr": round(float(scores[f"cosine_mrr@{len(target_texts)}"]), 4),
        f"mrr@{MRR_CUTOFF}": round(float(scores[f"cosine_mrr@{MRR_CUTOFF}"]), 4),
        **{
            f"accuracy@{k}": round(float(scores[f"cosine_accuracy@{k}"]), 4)
            for k in ACCURACY_AT_K
        },
        "median_rank": float(np.median([r["rank"] for r in ranked.values()])),
    }
    return metrics, ranked


def scores_folder(outputs, model, benchmark, text_format):
    """Where a model's scores on a benchmark, in one format, are written.

    ``--model`` is either a Hugging Face model id or a path to a model folder,
    and the scores folder is named after it:

    - a Hugging Face model keeps its owner and name, so two models of the same
      name from different owners stay apart:
      ``scores/uc-ctds/StdElement-Mapper/topmed/stdelement``;
    - a model folder on disk gives only its own name, whether the path is full
      or relative: ``/home/me/models/abcd-model`` gives
      ``scores/abcd-model/topmed/compact``.

    It tells them apart as SentenceTransformer does: an existing folder is a
    model on disk, and anything else is a Hugging Face id. A downloaded Hugging
    Face model sits in the Hugging Face cache under another name
    (``models--uc-ctds--StdElement-Mapper``), so its id stays an id.

    The format has a folder of its own, so a model scored in two formats keeps
    both scores.

    Args:
        outputs (Path): The outputs folder.
        model (str): The Hugging Face model id or folder that was scored.
        benchmark (str): The benchmark.
        text_format (str): The key of FORMATS that the texts were written in.

    Returns:
        Path: ``<outputs>/scores/<model>/<benchmark>/<format>``.
    """
    local = Path(model)
    name = local.resolve().name if local.is_dir() else model
    return Path(outputs) / "scores" / name / benchmark / text_format


def write_scores(folder, row, mappings, ranked, config):
    """Write one scoring's files: metrics.csv, per_variable.csv and config.yaml.

    The folder is written whole, through ``folders.staged``. So a scoring that
    fails while it writes leaves the previous scores as they were, and the
    files in a folder always come from one scoring.

    Args:
        folder (Path): The scores folder, from scores_folder.
        row (dict): The metrics row.
        mappings (pd.DataFrame): The benchmark's mappings, in ranking order.
        ranked (dict): Each source variable's rank and top-1 target, by row
            number, from rank_targets.
        config (dict): What config.yaml holds.
    """
    per_variable = mappings[
        [
            "source_study_id",
            "source_table_id",
            "source_variable_id",
            "source_variable_name",
            "expected_target",
        ]
    ].copy()
    for column in ("rank", "top1_target", "top1_similarity"):
        per_variable[column] = [ranked[str(i)][column] for i in range(len(mappings))]
    with staged(folder) as built:
        pd.DataFrame([row]).to_csv(built / "metrics.csv", index=False)
        per_variable.to_csv(built / "per_variable.csv", index=False)
        (built / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )


def scored_model(folder):
    """The model whose scores a scores folder holds, from its config.yaml.

    Args:
        folder (Path): The scores folder, from scores_folder.

    Returns:
        str | None: The model, or None when the folder has no config.yaml
            that can be read.
    """
    try:
        config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        return config.get("model")
    except Exception:
        return None


def run_record(model):
    """The record of a run trained here, when ``model`` is its folder or its model.

    ``train`` writes a run's folder with its record, ``manifest.json``, beside
    the model's folder, ``model/``. Either folder finds the record. The path is
    resolved first, so ``.`` inside ``model/`` finds it too. A folder that
    holds both is a run, so its record must hold every one of RUN_FIELDS.

    Args:
        model (str): A Hugging Face model id, a model folder or a run folder.

    Returns:
        tuple[Path, dict] | None: The run's folder and its record, or None
            when ``model`` is not a run of ``train``.

    Raises:
        ValueError: If the folder holds a record that cannot be read, or that
            lacks one of RUN_FIELDS. Scoring then stops, rather than scoring
            the run without its leakage check.
    """
    folder = Path(model)
    if not folder.is_dir():
        return None
    folder = folder.resolve()
    run = folder.parent if folder.name == "model" else folder
    path = run / "manifest.json"
    if not path.exists() or not (run / "model").is_dir():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(
            f"{path} cannot be read ({type(error).__name__}: {error}), so the"
            " run's leakage check cannot be read either"
        ) from error
    missing = [
        field
        for field in RUN_FIELDS
        if not isinstance(record, dict) or field not in record
    ]
    if missing:
        raise ValueError(
            f"{path} is not a run's record: it has no {', '.join(missing)}. So the"
            " run's leakage check cannot be read. Train the run again from its"
            " config.yaml"
        )
    return run, record


def check_run_files(run_folder, record):
    """Check that a run's model is the one that its record names.

    Args:
        run_folder (Path): The run's folder.
        record (dict): The run's record, whose ``files`` holds the SHA-256 of
            each file of ``model/``.

    Raises:
        ValueError: If a file is missing or differs from the record.
    """
    for name, entry in record["files"].items():
        path = run_folder / name
        if not path.exists() or file_sha256(path) != entry.get("sha256"):
            raise ValueError(
                f"{path} differs from the run's record, so it is not the model"
                f" that run {record['run']} trained"
            )


def checked_benchmarks(record):
    """The benchmarks that a run's corpus was checked against, by name.

    Args:
        record (dict): The run's record.

    Returns:
        dict[str, dict]: Each benchmark's entry: its version and its files,
            each with its checksums.
    """
    return {
        benchmark["name"]: benchmark
        for source in record.get("sources", [])
        if source.get("role") == "corpus"
        for benchmark in source.get("benchmarks", [])
    }


def score(
    model,
    benchmarks,
    inputs,
    outputs,
    text_format=None,
    device=None,
    batch_size=BATCH_SIZE,
):
    """Score a model on built benchmarks and write the results.

    Every benchmark is read and checked first. So a missing or changed one
    stops scoring before the model is loaded. The model is then loaded once,
    and scored on each benchmark in turn. A model on disk is loaded without
    asking Hugging Face, and recorded by its full path. A Hugging Face model
    is recorded by its id.

    A run of ``train`` is scored from its record. Its texts take the run's
    training format, unless ``text_format`` names another. A benchmark that
    the run's corpus was not checked against for leakage, by name and
    version, is refused: the model may have learned its mappings. The run's
    model must have the checksums that its record holds. The scores go into
    the run's folder. Any other model has no record, so its
    leakage is unknown, and scoring says so.

    For each benchmark, it writes:

    - ``metrics.csv``: one row of metrics;
    - ``per_variable.csv``: every source variable's rank and top-1 target;
    - ``config.yaml``: the settings, the model's revision, the
      ai-harmonization commit, and the benchmark's version and checksums.

    Args:
        model (str): A Hugging Face model id or a model folder on disk.
        benchmarks (list[str]): The benchmarks, such as ``["topmed"]``.
        inputs (Path): The inputs folder that the benchmarks were built into.
        outputs (Path): The outputs folder.
        text_format (str | None): A key of FORMATS: the shape of the texts.
            None takes a run's training format, and DEFAULT_FORMAT for any
            other model.
        device (str | None): ``cpu`` or ``cuda``. None uses a GPU if there is
            one, and the CPU otherwise.
        batch_size (int): Texts encoded at a time.

    Returns:
        list[dict]: The metrics rows, one per benchmark.

    Raises:
        FileNotFoundError: If a benchmark has not been built.
        ValueError: If a benchmark cannot be read, differs from its manifest
            or lists an expected target that is not among its targets, if a
            run's corpus was not checked against it, if a run's record cannot
            be read or its model differs from it, or if the model cannot be
            loaded.
    """
    from sentence_transformers import SentenceTransformer

    built = {name: read_benchmark(inputs, name) for name in benchmarks}
    run = run_record(model)
    if run is None:
        text_format = text_format or DEFAULT_FORMAT
        print(
            f"note: {model} was not trained here, so whether its training data"
            " leaks a benchmark is unknown"
        )
        model_path = model
    else:
        run_folder, record = run
        trained_format = recorded_format(
            record,
            run_folder / "manifest.json",
            "train the run again from its config.yaml",
        )
        text_format = text_format or trained_format
        checked = checked_benchmarks(record)
        corpus_name = next(
            source["name"]
            for source in record["sources"]
            if source.get("role") == "corpus"
        )
        rebuild = build_command(Path(inputs) / "corpora" / corpus_name)
        for name, (_, _, manifest) in built.items():
            if name not in checked:
                unchecked = f"benchmark {name}"
            else:
                changed = differences(
                    {"files": checked[name].get("files", {})},
                    {"files": manifest["files"]},
                )
                if not changed:
                    continue
                unchecked = f"this build of benchmark {name} ({'; '.join(changed)})"
            raise ValueError(
                f"the corpus of run {record['run']} was not checked against"
                f" {unchecked} for leakage, so the model may have learned its"
                f" mappings. Build the corpus again, after the benchmark:"
                f" {rebuild}. Then train again"
            )
        check_run_files(run_folder, record)
        model, model_path = str(run_folder), str(run_folder / "model")
    texts = {
        name: benchmark_texts(targets, mappings, text_format)
        for name, (targets, mappings, _) in built.items()
    }
    try:
        encoder = SentenceTransformer(
            model_path, device=device, local_files_only=Path(model_path).is_dir()
        )
    except Exception as error:
        raise ValueError(
            f"the model {model} cannot be loaded ({type(error).__name__}:"
            f" {error}); give a model folder on disk or a Hugging Face model id"
        ) from error
    scored = str(Path(model).resolve()) if Path(model).is_dir() else model
    commit = code_commit()
    revision = model_revision(model)
    rows = []
    for name, (_, mappings, manifest) in built.items():
        source_texts, target_texts, expected = texts[name]
        print(
            f"{name} version {manifest['version']}: {len(source_texts):,} source "
            f"variables, {len(target_texts):,} targets; format {text_format}"
        )
        metrics, ranked = rank_targets(
            encoder, source_texts, target_texts, expected, batch_size
        )
        row = {
            "model": scored,
            "benchmark": name,
            "benchmark_version": manifest["version"],
            "format": text_format,
            "n": len(source_texts),
            "targets": len(target_texts),
            **metrics,
        }
        config = {
            "model": scored,
            "model_revision": revision,
            "benchmark": name,
            "format": text_format,
            "device": str(encoder.device),
            "batch_size": batch_size,
            "benchmark_manifest": {
                key: manifest[key]
                for key in ("version", "built", "files")
                if key in manifest
            },
            "ai_harmonization": commit,
            "libraries": library_versions(),
            "scored": datetime.now().isoformat(timespec="seconds"),
        }
        if run is None:
            folder = scores_folder(outputs, model, name, text_format)
        else:
            folder = Path(model) / "scores" / name / text_format
        previous = scored_model(folder)
        if previous and previous != scored:
            print(
                f"warning: {folder} held the scores of {previous}. They are "
                f"replaced by those of {scored}.",
                file=sys.stderr,
            )
        write_scores(folder, row, mappings, ranked, config)
        print(
            "  "
            + ", ".join(f"{metric} {metrics[metric]}" for metric in METRIC_COLUMNS)
            + f"\n  -> {folder}"
        )
        rows.append(row)
    return rows
