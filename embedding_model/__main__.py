"""The command line: ``python -m embedding_model <step> ...``."""

import argparse
import sys
from pathlib import Path

from embedding_model import (
    benchmarks,
    corpus,
    pairs,
    scoring,
    sources,
    texts,
    tokenizer,
)

PACKAGE = Path(__file__).resolve().parent


def main(argv=None):
    """Parse the command line and run the step it names.

    Args:
        argv (list[str] | None): The arguments; ``sys.argv[1:]`` when None.

    Returns:
        int: The exit status, 0 on success.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--inputs",
        type=Path,
        default=PACKAGE / "inputs",
        help="where prepared data is written and read (default: %(default)s)",
    )
    writes_outputs = argparse.ArgumentParser(add_help=False)
    writes_outputs.add_argument(
        "--outputs",
        type=Path,
        default=PACKAGE / "outputs",
        help="where models and scores are written (default: %(default)s)",
    )

    parser = argparse.ArgumentParser(
        prog="python -m embedding_model",
        description="Build, train and evaluate a small embedding model.",
    )
    steps = parser.add_subparsers(dest="step", required=True)

    benchmark_step = steps.add_parser("benchmarks", help="build the benchmarks")
    benchmark_actions = benchmark_step.add_subparsers(dest="action", required=True)
    build = benchmark_actions.add_parser(
        "build",
        parents=[common],
        help="download a benchmark's inputs and write its mappings and targets",
    )
    for name in benchmarks.BUILDERS:
        build.add_argument(f"--{name}", action="store_true", help=f"build {name}")
    build.add_argument(
        "--update-manifest",
        action="store_true",
        help=(
            "accept a build that differs from its committed manifest, under a "
            "version not yet committed in git"
        ),
    )

    source_step = steps.add_parser("sources", help="fetch the training sources")
    source_actions = source_step.add_subparsers(dest="action", required=True)
    fetch = source_actions.add_parser(
        "fetch",
        parents=[common],
        help="download a source's release and write its tables",
    )
    fetch.add_argument(
        "--source",
        nargs="+",
        choices=list(sources.FETCHERS),
        default=list(sources.FETCHERS),
        help="the sources to fetch (default: all of them)",
    )
    fetch.add_argument(
        "--update-manifest",
        action="store_true",
        help=(
            "accept a fetch that differs from its committed manifest, under a "
            "version not yet committed in git"
        ),
    )

    pair_step = steps.add_parser("pairs", help="make the training pairs")
    pair_actions = pair_step.add_subparsers(dest="action", required=True)
    prepare = pair_actions.add_parser(
        "prepare",
        parents=[common],
        help="make each fetched source's pairs of texts that mean the same thing",
    )
    prepare.add_argument(
        "--source",
        nargs="+",
        choices=list(pairs.READERS),
        default=list(pairs.READERS),
        help="the sources to prepare (default: all of them)",
    )

    corpus_step = steps.add_parser("corpus", help="build a training corpus")
    corpus_actions = corpus_step.add_subparsers(dest="action", required=True)
    corpus_build = corpus_actions.add_parser(
        "build",
        parents=[common],
        help="combine the sources' pairs, drop what leaks a benchmark, and split",
    )
    corpus_build.add_argument("--name", required=True, help="the corpus's name")
    corpus_build.add_argument(
        "--source",
        nargs="+",
        choices=list(pairs.READERS),
        default=list(pairs.READERS),
        help="the sources to combine (default: all of them)",
    )

    tokenizer_step = steps.add_parser("tokenizer", help="train a tokenizer")
    tokenizer_actions = tokenizer_step.add_subparsers(dest="action", required=True)
    tokenizer_train = tokenizer_actions.add_parser(
        "train",
        parents=[common],
        help="learn the vocabulary from a corpus's training pairs",
    )
    tokenizer_train.add_argument(
        "--corpus", required=True, help="the corpus to learn from"
    )
    tokenizer_train.add_argument(
        "--name",
        help="the tokenizer's name (default: the corpus and the vocabulary size,"
        " such as example-16k)",
    )
    tokenizer_train.add_argument(
        "--vocab-size",
        type=int,
        default=tokenizer.VOCAB_SIZE,
        help=f"the vocabulary's size (default: {tokenizer.VOCAB_SIZE:,})",
    )
    tokenizer_train.add_argument(
        "--min-frequency",
        type=int,
        default=tokenizer.MIN_FREQUENCY,
        help="how often a merge must occur to become a token"
        f" (default: {tokenizer.MIN_FREQUENCY})",
    )

    score = steps.add_parser(
        "score",
        parents=[common, writes_outputs],
        help="score a model on benchmarks: each source variable's rank, MRR and accuracy@k",
    )
    score.add_argument(
        "--model",
        required=True,
        help="a Hugging Face model id or a model folder on disk",
    )
    score.add_argument(
        "--benchmark",
        required=True,
        nargs="+",
        choices=sorted(benchmarks.BUILDERS),
        help="one or more benchmarks. The model is loaded once for all of them.",
    )
    score.add_argument(
        "--format",
        default=texts.DEFAULT_FORMAT,
        choices=sorted(texts.FORMATS),
        help="the shape of the source and target texts (default: %(default)s)",
    )
    score.add_argument(
        "--device",
        help="cpu or cuda (default: a GPU if there is one, else the CPU)",
    )
    score.add_argument(
        "--batch-size",
        type=int,
        default=scoring.BATCH_SIZE,
        help="texts encoded at a time (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    if args.step == "score":
        try:
            scoring.score(
                args.model,
                args.benchmark,
                args.inputs,
                args.outputs,
                args.format,
                args.device,
                args.batch_size,
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"score: {error}", file=sys.stderr)
            return 1
        return 0
    if args.step == "benchmarks" and args.action == "build":
        names = [name for name in benchmarks.BUILDERS if getattr(args, name)]
        if not names:
            build.error(
                "choose at least one of "
                + ", ".join(f"--{name}" for name in benchmarks.BUILDERS)
            )
        return benchmarks.build(names, args.inputs, args.update_manifest)
    if args.step == "sources" and args.action == "fetch":
        return sources.fetch(args.source, args.inputs, args.update_manifest)
    if args.step == "pairs" and args.action == "prepare":
        return pairs.prepare(args.source, args.inputs)
    if args.step == "corpus" and args.action == "build":
        return corpus.build(args.name, args.source, args.inputs)
    if args.step == "tokenizer" and args.action == "train":
        return tokenizer.train(
            args.corpus, args.name, args.vocab_size, args.min_frequency, args.inputs
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
