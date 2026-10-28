"""Folders that a step writes whole, such as a corpus or a tokenizer.

Each is one folder under the inputs folder, named by the user. A step writes it
inside ``staged``, which moves it into place only once the step has written it
all. So a failed step leaves the previous folder of that name as it was.

A run that is killed, such as by a time limit, cannot clean up. It leaves its
hidden staging folder, ``.<name>.<random>/``, next to the folder that it
writes. When it was killed while it swapped the two folders, the staging folder
holds them both. The next run of the step warns about the staging folder that
is left, and prints its path, so it can be checked and deleted.
"""

import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

# A name becomes a folder's name, so it holds only these characters. It never
# starts with a dot, which ``staged`` and ``replace`` rely on.
FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# FOLDER_NAME in words, for the messages that refuse a name.
FOLDER_NAME_RULE = (
    "letters, digits, '.', '_' and '-', starting with a letter or a digit"
)


def replace(built, destination, staging):
    """Move a complete folder into place, over the previous one, if there is one.

    The previous folder, from an earlier build of the same name, is first
    renamed out of the way: into ``staging``, as ``.previous``. No FOLDER_NAME
    starts with a dot, so the two names cannot clash. Then the new folder
    takes its place. If that fails, the previous folder is renamed back. If
    even that fails, it stays in ``staging``, which ``staged`` then keeps.

    Args:
        built (Path): The new folder.
        destination (Path): Where the folder goes.
        staging (Path): A folder on the same file system, deleted afterwards.
    """
    previous = staging / ".previous"
    try:
        if destination.exists():
            destination.rename(previous)
        built.rename(destination)
    except BaseException:
        if previous.exists() and not destination.exists():
            previous.rename(destination)
        raise


@contextmanager
def staged(destination):
    """A new, empty folder to write ``destination`` in.

    The folder sits in a staging folder beside the destination, so the move
    stays on one file system. When the block ends without an error, the new
    folder replaces the previous one. When the block raises, the new folder
    is deleted and the previous one is kept. If the previous folder cannot be
    put back after a failed move, the staging folder is kept, and its path is
    printed.

    Args:
        destination (Path): The folder to write. Its name is a FOLDER_NAME.

    Yields:
        Path: The new folder, to write in.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for left in sorted(destination.parent.glob(f".{destination.name}.*")):
        print(
            f"warning: {left} is left from a run that was stopped; check it,"
            " then delete it",
            file=sys.stderr,
        )
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    built = staging / destination.name
    try:
        built.mkdir()
        yield built
        replace(built, destination, staging)
    finally:
        previous = staging / ".previous"
        if previous.exists() and not destination.exists():
            print(
                f"warning: the previous {destination.name} could not be put"
                f" back; it is kept in {previous}",
                file=sys.stderr,
            )
        else:
            shutil.rmtree(staging, ignore_errors=True)
