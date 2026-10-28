"""Score a model on a benchmark: where it ranks each variable's expected targets.

For each source variable, the model ranks every target of the benchmark by
cosine similarity. sentence-transformers' ``InformationRetrievalEvaluator`` encodes
the texts, ranks the targets and computes MRR and accuracy@k.

A variable with several expected targets takes the rank of the best placed of
them. accuracy@k is the share of variables with an expected target among the
first k. It is what a reviewer of the top k suggestions sees.
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ai_harmonization.formatters import VALUE_SEPARATOR
from embedding_model.benchmarks import TARGET_SEPARATOR, read_benchmark
from embedding_model.folders import staged
from embedding_model.texts import DEFAULT_FORMAT, FORMATS, property_text

ACCURACY_AT_K = (1, 5, 10)
METRIC_COLUMNS = ["mrr", *[f"accuracy@{k}" for k in ACCURACY_AT_K], "median_rank"]
BATCH_SIZE = 128


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

    Raises:
        ValueError: If an expected target is not among the benchmark's targets.
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
    source_texts, expected, unknown = {}, {}, []
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
        unknown += sorted(expected[str(number)] - target_texts.keys())
    if unknown:
        raise ValueError(
            f"{len(unknown)} expected targets are not in the benchmark's targets: "
            + ", ".join(unknown[:5])
        )
    return source_texts, target_texts, expected


def evaluator_class():
    """The InformationRetrievalEvaluator of the installed sentence-transformers.

    Version 6 and later 5.x releases, such as 5.4, keep it in
    ``sentence_transformers.sentence_transformer.evaluation``. Earlier 5.x
    releases, such as 5.1, keep it in ``sentence_transformers.evaluation``.
    Version 6 still accepts that path, with a deprecation warning.
    """
    try:
        from sentence_transformers.sentence_transformer.evaluation import (
            InformationRetrievalEvaluator,
        )
    except ImportError:
        from sentence_transformers.evaluation import InformationRetrievalEvaluator
    return InformationRetrievalEvaluator


def rank_targets(model, source_texts, target_texts, expected, batch_size=BATCH_SIZE):
    """Rank every target for each source, and score the ranks.

    Targets are ranked by cosine similarity. The evaluator gives the results
    two ways. It returns the averages in memory: MRR and accuracy@k over all
    source variables. Its predictions file, written to a temporary folder,
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
    # The approaches embed both sides with model.encode(), which adds only the
    # model's default prompt, if its config names one. Left alone, the
    # evaluator would add the model's "query" and "document" prompts instead.
    # A model with a Router module still has its two sides routed apart.
    #
    # The approaches rank by cosine similarity, even when a model's config
    # names another similarity function. So the evaluator ranks by cosine too.
    from sentence_transformers.util import cos_sim

    prompt = model.prompts.get(model.default_prompt_name, "")
    evaluator = evaluator_class()(
        queries=source_texts,
        corpus=target_texts,
        relevant_docs=expected,
        query_prompt=prompt,
        corpus_prompt=prompt,
        score_functions={"cosine": cos_sim},
        mrr_at_k=[len(target_texts)],
        accuracy_at_k=list(ACCURACY_AT_K),
        batch_size=batch_size,
        show_progress_bar=False,
        write_csv=False,
        write_predictions=True,
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
        **{
            f"accuracy@{k}": round(float(scores[f"cosine_accuracy@{k}"]), 4)
            for k in ACCURACY_AT_K
        },
        "median_rank": float(np.median([r["rank"] for r in ranked.values()])),
    }
    return metrics, ranked


def code_commit():
    """The ai-harmonization commit of the code that wrote the texts and ranked them.

    Returns:
        dict: ``commit``, the commit at ``HEAD``, and ``uncommitted_changes``,
            whether ``ai_harmonization/`` or ``embedding_model/`` differ from
            it. Both are None when git cannot be asked.
    """
    git = ["git", "-C", str(Path(__file__).resolve().parent)]
    paths = ["--", ":/ai_harmonization", ":/embedding_model"]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            [*git, "status", "--porcelain", *paths],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:
        return {"commit": None, "uncommitted_changes": None}
    return {"commit": commit, "uncommitted_changes": bool(status.strip())}


def model_revision(model):
    """The Hugging Face commit of a model loaded by its id, read from the cache.

    Loading a model by id downloads it into the Hugging Face cache, under its
    commit. The commit is read from there, with no network, in the cache that
    SentenceTransformer uses: ``SENTENCE_TRANSFORMERS_HOME``, where set.

    The id is looked up as given. SentenceTransformer reads an id with no
    owner, such as ``all-MiniLM-L6-v2``, as a ``sentence-transformers/``
    model, so such an id gives None.

    Args:
        model (str): A Hugging Face model id or a model folder on disk.

    Returns:
        str | None: The commit, or None for a model folder on disk or an id
            that the cache does not hold.
    """
    if Path(model).is_dir():
        return None
    from huggingface_hub import snapshot_download

    try:
        folder = snapshot_download(
            model,
            local_files_only=True,
            cache_dir=os.getenv("SENTENCE_TRANSFORMERS_HOME"),
        )
    except Exception:
        return None
    return Path(folder).name


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


def score(
    model,
    benchmarks,
    inputs,
    outputs,
    text_format=DEFAULT_FORMAT,
    device=None,
    batch_size=BATCH_SIZE,
):
    """Score a model on built benchmarks and write the results.

    Every benchmark is read and checked first. So a missing or changed one
    stops scoring before the model is loaded. The model is then loaded once,
    and scored on each benchmark in turn. A model on disk is loaded without
    asking Hugging Face, and recorded by its full path. A Hugging Face model
    is recorded by its id.

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
        text_format (str): A key of FORMATS: the shape of the texts.
        device (str | None): ``cpu`` or ``cuda``. None uses a GPU if there is
            one, and the CPU otherwise.
        batch_size (int): Texts encoded at a time.

    Returns:
        list[dict]: The metrics rows, one per benchmark.

    Raises:
        FileNotFoundError: If a benchmark has not been built.
        ValueError: If a benchmark cannot be read, differs from its manifest
            or lists an expected target that is not among its targets, or if
            the model cannot be loaded.
    """
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    built = {name: read_benchmark(inputs, name) for name in benchmarks}
    texts = {
        name: benchmark_texts(targets, mappings, text_format)
        for name, (targets, mappings, _) in built.items()
    }
    try:
        encoder = SentenceTransformer(
            model, device=device, local_files_only=Path(model).is_dir()
        )
    except Exception as error:
        raise ValueError(
            f"the model {model} cannot be loaded ({type(error).__name__}:"
            f" {error}); give a model folder on disk or a Hugging Face model id"
        ) from error
    scored = str(Path(model).resolve()) if Path(model).is_dir() else model
    code = code_commit()
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
            "ai_harmonization": code,
            "python": platform.python_version(),
            "sentence_transformers": sentence_transformers.__version__,
            "scored": datetime.now().isoformat(timespec="seconds"),
        }
        folder = scores_folder(outputs, model, name, text_format)
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
            + f"\n  written to {folder}"
        )
        rows.append(row)
    return rows
