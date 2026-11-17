"""Folders that a step writes whole, for example a corpus or a tokenizer.

Each is one folder under the inputs or the outputs folder. A step writes it
inside ``staged``, which moves the folder into place only after the step writes
all of it. So a failed step leaves the previous folder of that name as it was.

A run that is stopped with Ctrl-C deletes its staging folder. A run that is
killed, for example by ``scancel`` or a time limit, cannot delete it. The
killed run leaves its hidden staging folder, ``.<name>~<random>/``, next to the
folder that it writes. If the run was killed while it swapped the two folders,
the staging folder holds both, with the previous folder as ``.previous``. The
next run of the step warns about each staging folder that is left. The warning
gives the folder's path and says what to do with it.
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

    replace first renames the previous folder, from an earlier build of the
    same name, into ``staging``, as ``.previous``. No FOLDER_NAME starts with a
    dot, so the two names cannot clash. Then the new folder takes the previous
    folder's place. If that rename fails, replace renames the previous folder
    back. If even that fails, the previous folder stays in ``staging``, which
    ``staged`` then keeps.

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

    Another run can write the same folder at the same time, for example the
    same ``train`` submitted twice. That run has a staging folder too, and
    nothing tells the two apart. So the warning says to delete the folder only
    when no such run is writing. The staging folder holds the previous folder,
    as ``.previous``, when the run was killed between the two renames of
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
            f"{warning} The staging folder holds the previous {destination.name},"
            f" in {previous}. If this run fails, move it back to {destination}."
            f" Otherwise, once no other run writes {destination.name}, delete"
            f" {left}."
        )
    return (
        f"{warning} If no other run writes {destination.name}, nothing in the"
        " staging folder is needed: delete it."
    )


def staging_folders(destination):
    """The staging folders beside ``destination``, from a running write or from killed runs.

    A staging folder is ``.<name>~<random>/``. No FOLDER_NAME holds a ``~``.
    So the staging folders of ``abcd`` never include ``.abcd.v2~x1y2``, the
    staging folder of ``abcd.v2``.

    Args:
        destination (Path): The folder that the staging folders belong to.

    Returns:
        list[Path]: The staging folders, sorted.
    """
    return sorted(destination.parent.glob(f".{destination.name}{STAGING_MARK}*"))


@contextmanager
def staged(destination):
    """A new, empty folder to write ``destination`` in.

    The folder sits in a staging folder beside the destination, so the move
    stays on one file system. When the block ends without an error, the new
    folder replaces the previous one. When the block raises, staged deletes
    the new folder and keeps the previous one. If a move fails and the
    previous folder cannot return to its place, staged keeps the staging
    folder, and prints where the previous folder is.

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
