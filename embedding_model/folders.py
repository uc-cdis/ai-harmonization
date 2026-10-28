"""Folders that a step writes whole, such as a corpus or a tokenizer.

Each is one folder under the inputs folder, named by the user. A step writes it
inside ``staged``, which moves it into place only once the step has written it
all. So a failed step leaves the previous folder of that name as it was.
"""

import re
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
    takes its place. If that fails, the previous folder is renamed back, so it
    is never lost.

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
    is deleted and the previous one is kept.

    Args:
        destination (Path): The folder to write. Its name is a FOLDER_NAME.

    Yields:
        Path: The new folder, to write in.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        ignore_cleanup_errors=True,
    ) as staging:
        built = Path(staging) / destination.name
        built.mkdir()
        yield built
        replace(built, destination, Path(staging))
