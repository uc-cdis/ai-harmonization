"""What a build was made from, and whether a new build is the same.

Benchmarks and sources each keep a committed manifest per build, under their
own ``manifests/`` folder. It pins what the build is made from, such as a
commit, a release or a record, and records the SHA-256 of every file built.
A new build writes its own manifest, and the two are compared. A build that
matches is kept. A build that differs is kept only when accepted as a new
version. A version committed in git never changes.
``embedding_model/docs/manifests.md`` describes the fields and the rules.

The module also computes the checksums that a manifest records:

- a file's ``sha256``, by ``file_sha256`` for a file on disk, or by ``sha256``
  for bytes in memory. Both give the same value for the same bytes;
- a table's ``content_sha256``, by ``table_sha256``, from the table's content.
"""

import csv
import hashlib
import json
import subprocess
import sys

import pandas as pd
import pyarrow as pa

from embedding_model.folders import staged


def sha256(data):
    """The SHA-256 of bytes that are in memory already, as hex.

    It gives the same value as ``file_sha256`` of a file with these bytes. Use
    it when the code holds the bytes anyway, such as a download just received.
    For a file on disk, use ``file_sha256``, which never holds the file whole.

    Args:
        data (bytes): The bytes.

    Returns:
        str: The SHA-256, as hex.
    """
    return hashlib.sha256(data).hexdigest()


def file_digest(path, algorithm):
    """A file's checksum by one of hashlib's algorithms, as hex, read in chunks.

    The file is read piece by piece, so even a large file is never in memory
    whole. ``file_sha256`` uses it for SHA-256. LOINC's fetch uses it for MD5,
    the checksum that LOINC publishes for each release.

    Args:
        path (Path): The file.
        algorithm (str): A hashlib algorithm, such as ``sha256`` or ``md5``.

    Returns:
        str: The checksum, as hex.
    """
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def file_sha256(path):
    """The SHA-256 of a file's bytes, as hex, read in chunks.

    It gives the same value as ``sha256`` of the file's bytes, but it never
    holds the whole file in memory. So it suits a large file, such as a
    release zip or a corpus's pairs. A manifest records it as a file's
    ``sha256``.

    Args:
        path (Path): The file.

    Returns:
        str: The SHA-256, as hex.
    """
    return file_digest(path, "sha256")


# A missing value in a table's hashed CSV text, as MySQL's and PostgreSQL's
# data files write one. It tells a missing value from an empty text.
MISSING = "\\N"
# The Arrow types that a schema line reads as ``string``.
AS_STRING = {"large_string": "string", "null": "string"}


class Digest:
    """A file-like object that adds the text written to it to a SHA-256.

    pandas streams a CSV only into a file-like object, and a hashlib hash is not
    one: it takes bytes through ``update``, not text through ``write``. With
    this object, ``table_sha256`` hashes a table as pandas writes it, piece by
    piece. So a table of millions of rows, such as a corpus's pairs, is hashed
    without its whole CSV text in memory.
    """

    def __init__(self):
        """Start an empty SHA-256."""
        self.hash = hashlib.sha256()

    def write(self, text):
        """Add text to the hash, as UTF-8, and give back its length."""
        self.hash.update(text.encode("utf-8"))
        return len(text)

    def __iter__(self):
        """Nothing to read: pandas takes only an iterable as a file."""
        return iter(())


def schema_line(frame):
    """A table's column names and their pyarrow types, as one line of text.

    The types are those that pyarrow gives the frame's columns. ``large_string``
    and ``null`` are read as ``string``, so the line does not depend on how
    pandas holds a column.

    Args:
        frame (pd.DataFrame): The table.

    Returns:
        str: The line, such as ``name: string, open: bool``.
    """
    schema = pa.Schema.from_pandas(frame, preserve_index=False)
    return ", ".join(
        f"{field.name}: {AS_STRING.get(str(field.type), str(field.type))}"
        for field in schema
    )


def table_sha256(frame):
    """The SHA-256 of a table's content: its schema line, then its rows as CSV.

    A manifest records it as a table's ``content_sha256``. The content is hashed
    rather than the parquet file, whose bytes depend on the library that wrote
    it. The schema line keeps the types that CSV loses, so ``True`` and the
    text ``"True"`` give different checksums. A missing value is written as
    MISSING, so it differs from an empty text and from the text ``\\N``. The
    separator, the quoting and the line ending are set, and lines end with
    ``\\n`` on every system, so the checksum is the same on Windows. pandas
    writes the CSV to the hash piece by piece, through ``Digest``, so the whole
    text is never held in memory. pandas uses the same pieces for a whole
    text, so the checksum is that of the whole text.

    Args:
        frame (pd.DataFrame): The table.

    Returns:
        str: The SHA-256, as hex.
    """
    # A backslash in a text is written twice, as PostgreSQL writes one, so the
    # text "\N" cannot read like MISSING. Only the text columns that hold a
    # backslash are copied, to keep a large table's memory low.
    escaped = {
        name: frame[name].str.replace("\\", "\\\\", regex=False)
        for name, kind in frame.dtypes.items()
        if pd.api.types.is_string_dtype(kind)
        and frame[name].str.contains("\\", regex=False).any()
    }
    frame = frame.assign(**escaped) if escaped else frame
    digest = Digest()
    digest.write(schema_line(frame) + "\n")
    frame.to_csv(
        digest,
        index=False,
        sep=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
        na_rep=MISSING,
    )
    return digest.hash.hexdigest()


def file_identity(file_entry):
    """What a file is compared by, from its entry in a manifest's ``files``.

    A parquet table has ``content_sha256``, the SHA-256 of its content, and is
    compared by it and its rows. Its file's bytes depend on the library that
    wrote it, so the same table can have another ``sha256`` on another machine.
    Any other file is compared by its rows and its ``sha256``.

    Args:
        file_entry (dict): The file's entry in ``files``.

    Returns:
        dict: Its rows, and the one checksum that it is compared by.
    """
    key = "content_sha256" if "content_sha256" in file_entry else "sha256"
    return {"rows": file_entry.get("rows"), key: file_entry.get(key)}


def identities(manifest):
    """What a build is compared by: its files, then its sources.

    A file is identified as ``file_identity`` says, and a source by its
    SHA-256, or by its version where it has none.

    Args:
        manifest (dict): A manifest.

    Returns:
        dict: One entry per file, by name, and per source, by role and name.
    """
    return {
        **{
            file: file_identity(file_entry)
            for file, file_entry in manifest["files"].items()
        },
        **{
            f"{source['role']} {source['name']!r}": source.get("sha256")
            or source.get("version")
            for source in manifest["sources"]
        },
    }


def differences(committed, built_manifest):
    """What differs between a committed manifest and a new build's manifest.

    A source is identified by its SHA-256 where it has one, and otherwise by
    its version, so a file served under a new release label but with the
    same content is the same source. Dates, counts, citations and licenses
    are not compared.

    Args:
        committed (dict): The manifest under ``manifests/``.
        built_manifest (dict): The manifest of the build just made.

    Returns:
        list[str]: One line per difference; empty when the two match.
    """
    before, after = identities(committed), identities(built_manifest)
    keys = [*before, *(key for key in after if key not in before)]
    return [
        f"{key}: {before.get(key)} -> {after.get(key)}"
        for key in keys
        if before.get(key) != after.get(key)
    ]


def git(folder, *args):
    """Run git in a folder, and give back what it prints.

    Args:
        folder (Path): Where git runs.
        *args (str): git's own arguments, such as ``ls-tree``.

    Returns:
        str: git's output.

    Raises:
        RuntimeError: If git cannot run, such as when it is missing, or if it
            fails. The error holds git's own message, such as ``fatal: not a
            git repository``.
    """
    try:
        result = subprocess.run(
            ["git", *args], cwd=folder, capture_output=True, text=True
        )
    except Exception as error:
        raise RuntimeError(f"git could not run: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout


def committed_version(folder, name):
    """The version of a manifest as committed in git, at ``HEAD``.

    git runs in the folder, and the manifest is looked up in git's own listing
    of it. So no text from the command line reaches a git command.

    Args:
        folder (Path): The ``manifests/`` folder.
        name (str): The manifest's name, without ``.json``.

    Returns:
        str | None: The version, or None when the manifest is not committed
            yet.

    Raises:
        RuntimeError: From ``git``, if git cannot answer: outside a checkout,
            with git missing, when git refuses the folder, or in a checkout
            without a commit. The error holds git's own message.
    """
    listed = git(folder, "ls-tree", "--name-only", "HEAD").splitlines()
    wanted = f"{name}.json"
    entry = next((e for e in listed if e == wanted), None)
    if entry is None:
        return None
    return json.loads(git(folder, "show", f"HEAD:./{entry}"))["version"]


def pinned(manifest, name):
    """One of a build's inputs, as its committed manifest pins it.

    Args:
        manifest (dict): The committed manifest of a benchmark or a source.
        name (str): The input's name in the manifest's ``sources``.

    Returns:
        dict: The manifest's entry for the input: its version, SHA-256 or
            both.

    Raises:
        KeyError: If the manifest has no input of that name.
    """
    for source in manifest.get("sources", []):
        if source.get("name") == name:
            return source
    raise KeyError(f"the manifest has no source named {name!r}")


def write(path, manifest):
    """Write a manifest as JSON, in the one format that every manifest has.

    Args:
        path (Path): Where the manifest goes.
        manifest (dict): The manifest.
    """
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read(path, command, field, name):
    """A manifest that a step wrote, checked to be the manifest of ``name``.

    Each step reads its inputs' manifests with this, so every error names
    the file and the command that writes it again.

    Args:
        path (Path): The manifest.
        command (str): The command that writes the manifest again, for the
            messages.
        field (str): The field that names what was built, such as ``source``.
        name (str): The name that the field must hold, such as ``ncit``.

    Returns:
        dict: The manifest.

    Raises:
        FileNotFoundError: If there is no manifest at ``path``.
        ValueError: If it cannot be read, it is not a JSON object, or it is
            the manifest of another build.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run: {command}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(
            f"{path} cannot be read ({type(error).__name__}: {error});"
            f" run: {command}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} is not a manifest; run: {command}")
    if manifest.get(field) != name:
        raise ValueError(
            f"{path} is the manifest of {field} {manifest.get(field)!r};"
            f" run: {command}"
        )
    return manifest


def warn_if_not_committed(label, version, committed_path, command):
    """Print a warning when a build's version is not the committed version.

    Args:
        label (str): What was built and where, for the message.
        version (str): The build's version.
        committed_path (Path): The committed manifest. Nothing is printed
            when there is none.
        command (str): The command that builds the committed version.
    """
    if not committed_path.exists():
        return
    committed = json.loads(committed_path.read_text(encoding="utf-8"))["version"]
    if committed != version:
        print(
            f"warning: {label} is version {version}, and the committed manifest "
            f"is version {committed}; for version {committed}, run: {command}",
            file=sys.stderr,
        )


class Refused(Exception):
    """A build that differs from its committed manifest, and is not accepted."""


def decide(committed_path, built_manifest, update_manifest, command):
    """Whether a build may replace the working one; raise Refused if not.

    A build that matches its committed manifest may replace the working one. A
    build that differs is accepted only with ``update_manifest``, under a
    version not committed in git. Otherwise the differences are printed, with
    the steps that accept the build, and Refused is raised. ``write_checked``
    calls this inside ``folders.staged``, so a refused build is deleted and
    the working one is kept.

    Args:
        committed_path (Path): The committed manifest.
        built_manifest (dict): The new build's manifest.
        update_manifest (bool): Accept a build that differs.
        command (str): The command that made the build, for the messages.

    Returns:
        list[str]: The differences; empty when the build matches.

    Raises:
        Refused: If the build differs and is not accepted.
    """
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    differing = differences(committed, built_manifest)
    if not differing:
        return differing
    version = committed["version"]
    try:
        head_version = committed_version(committed_path.parent, committed_path.stem)
    except RuntimeError as error:
        refusal = (
            f"git cannot say whether version {version} is committed ({error})."
            " To accept this build, run it in a git checkout of this repository"
            " where `git status` works"
        )
    else:
        if head_version == version:
            refusal = (
                f'version {version} is committed: raise "version" in'
                f" {committed_path}, then run: {command} --update-manifest"
            )
        elif not update_manifest:
            refusal = (
                f"to accept this build as version {version}, run: {command}"
                " --update-manifest"
            )
        else:
            return differing
    print(f"  differs from {committed_path.name}:", file=sys.stderr)
    for line in differing:
        print(f"    {line}", file=sys.stderr)
    print(f"  {refusal}", file=sys.stderr)
    raise Refused(refusal)


def record(committed_path, built_manifest, differing, directory):
    """Report a build that is in place, and record it if it was accepted.

    Args:
        committed_path (Path): The committed manifest.
        built_manifest (dict): The build's manifest.
        differing (list[str]): The differences that ``decide`` returned.
        directory (Path): Where the build now is.
    """
    for file, entry in built_manifest["files"].items():
        print(f"  {directory / file}: {entry['rows']:,} rows")
    if differing:
        write(committed_path, built_manifest)
        print(f"  accepted, and recorded in {committed_path}")
    else:
        print(f"  matches {committed_path.name}")


def write_checked(destination, write, committed_path, update_manifest, command):
    """Write a build whole, and keep it only if its committed manifest allows.

    The build is written through ``folders.staged``. ``decide`` then says
    whether it may replace the working folder, while the build is still in
    the staging folder. A refused build is deleted, and the working folder is
    kept. A kept build is recorded only once it is in place.

    Args:
        destination (Path): The build's folder, such as ``sources/<source>/``.
        write (Callable[[Path], dict]): Writes the build into the folder that
            it is given, and returns the build's manifest.
        committed_path (Path): The committed manifest.
        update_manifest (bool): Accept a build that differs.
        command (str): The command that made the build, for the messages.

    Returns:
        int: 0 if the build matches or was accepted, 1 if it was refused.
    """
    try:
        with staged(destination) as built:
            built_manifest = write(built)
            differing = decide(committed_path, built_manifest, update_manifest, command)
    except Refused:
        return 1
    record(committed_path, built_manifest, differing, destination)
    return 0
