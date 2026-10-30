"""Folders that a step writes whole, such as a corpus or a tokenizer.

Each is one folder under the inputs folder, named by the user. A step writes it
inside ``staged``, which moves it into place only once the step has written it
all. So a failed step leaves the previous folder of that name as it was.

A run that is stopped, by its user or by a time limit, cannot clean up. It
leaves its hidden staging folder, ``.<name>.<random>/``, next to the folder
that it writes. When it was stopped while it swapped the two folders, the
staging folder holds them both, and the previous folder is its ``.previous``.
The next run of the step warns about each staging folder that is left, prints
its path and says what to do with it.
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
# The name of the previous folder inside a staging folder. It starts with a
# dot, so no FOLDER_NAME is the same.
PREVIOUS = ".previous"


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
    previous = staging / PREVIOUS
    try:
        if destination.exists():
            destination.rename(previous)
        built.rename(destination)
    except BaseException:
        if previous.exists() and not destination.exists():
            previous.rename(destination)
        raise


def left_warning(left, destination):
    """The warning about a staging folder that a stopped run left.

    The staging folder holds the previous folder, as ``.previous``, when the
    run was stopped between the two renames of ``replace``. The destination is
    then missing, and ``.previous`` is the only copy of it. Otherwise nothing
    in the staging folder is needed.

    Args:
        left (Path): The staging folder.
        destination (Path): The folder that the stopped run wrote.

    Returns:
        str: The warning, which says what to do with the staging folder.
    """
    warning = f"warning: {left} is left from a run that was stopped."
    previous = left / PREVIOUS
    if previous.exists() and not destination.exists():
        return (
            f"{warning} It holds the previous {destination.name}, in {previous}."
            f" If this run fails, move it back to {destination}. Otherwise"
            f" delete {left}."
        )
    return f"{warning} Nothing in it is needed, so delete it."


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
    prefix = f".{destination.name}."
    # The random part that mkdtemp adds after the prefix has no dot. The staging
    # folder of a longer name, such as .abcd.v2.x1y2 for abcd.v2, has a dot
    # after the prefix, so it is skipped.
    for left in sorted(destination.parent.glob(f"{prefix}*")):
        if "." not in left.name[len(prefix) :]:
            print(left_warning(left, destination), file=sys.stderr)
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=prefix))
    built = staging / destination.name
    try:
        built.mkdir()
        yield built
        replace(built, destination, staging)
    finally:
        previous = staging / PREVIOUS
        if previous.exists() and not destination.exists():
            print(
                f"warning: the previous {destination.name} could not be put"
                f" back. It is kept in {previous}. Move it back to"
                f" {destination}.",
                file=sys.stderr,
            )
        else:
            shutil.rmtree(staging, ignore_errors=True)
