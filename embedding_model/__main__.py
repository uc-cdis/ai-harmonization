"""The command line: ``python -m embedding_model <step> ...``."""

import argparse
import sys
from pathlib import Path

from embedding_model import benchmarks

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

    args = parser.parse_args(argv)
    if args.step == "benchmarks" and args.action == "build":
        names = [name for name in benchmarks.BUILDERS if getattr(args, name)]
        if not names:
            build.error(
                "choose at least one of "
                + ", ".join(f"--{name}" for name in benchmarks.BUILDERS)
            )
        return benchmarks.build(names, args.inputs, args.update_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
