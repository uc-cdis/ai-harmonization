"""Folders that a step writes whole, such as a corpus or a tokenizer.

Each is one folder under the inputs folder, named by the user. A step writes it
inside ``staged``, which moves it into place only once the step has written it
all. So a failed step leaves the previous folder of that name as it was.

A run that is stopped with Ctrl-C deletes its staging folder. A run that is
killed, such as by ``scancel`` or a time limit, cannot clean up. It leaves its
hidden staging folder, ``.<name>~<random>/``, next to the folder that it
writes. When it was killed while it swapped the two folders, the staging
folder holds them both, and the previous folder is its ``.previous``.
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
# The mark between a name and the random part of its staging folder's name.
# No FOLDER_NAME holds it, so a staging folder belongs to one name only.
STAGING_MARK = "~"


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
    """The warning about a staging folder that a killed run left, or that another run writes now.

    A run that writes the same folder at the same time, such as the same
    ``train`` submitted twice, has a staging folder too, and nothing tells the
    two apart. So the warning says to delete the folder only when no such run
    is writing. The staging folder holds the previous folder, as
    ``.previous``, when the run was killed between the two renames of
    ``replace``. The destination is then missing, and ``.previous`` is the
    only copy of it. Otherwise nothing in the staging folder is needed.

    Args:
        left (Path): The staging folder.
        destination (Path): The folder that the killed run wrote.

    Returns:
        str: The warning, which says what to do with the staging folder.
    """
    warning = (
        f"warning: {left} is left from a run that was killed, or another run"
        f" is writing {destination.name} now."
    )
    previous = left / PREVIOUS
    if previous.exists() and not destination.exists():
        return (
            f"{warning} It holds the previous {destination.name}, in {previous}."
            f" If this run fails, move it back to {destination}. Otherwise,"
            f" once no other run writes it, delete {left}."
        )
    return f"{warning} If no other run writes it, nothing in it is needed: delete it."


def staging_folders(destination):
    """The staging folders beside ``destination``: one that a run writes now, or ones that killed runs left.

    A staging folder is ``.<name>~<random>/``. No FOLDER_NAME holds a ``~``,
    so another name's staging folder never matches, such as ``.abcd.v2~x1y2``
    for ``abcd.v2``.

    Args:
        destination (Path): The folder that the staging folders write.

    Returns:
        list[Path]: The staging folders, sorted.
    """
    return sorted(destination.parent.glob(f".{destination.name}{STAGING_MARK}*"))


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
    for left in staging_folders(destination):
        print(left_warning(left, destination), file=sys.stderr)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent, prefix=f".{destination.name}{STAGING_MARK}"
        )
    )
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
