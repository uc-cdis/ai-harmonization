"""What a build was made from, and whether a new build is the same.

Each benchmark and each source has a committed manifest, in the
``manifests/`` folder of its package. The committed manifest pins what the
build is made from, for example a commit, a release or a record. It also holds
the SHA-256 of every file built.

A new build writes its own manifest, and ``decide`` compares the two.
``write_checked`` keeps a build that matches. It keeps a build that differs
only when the build is accepted as a new version. A version committed in git
never changes. ``embedding_model/docs/manifests.md`` describes the fields and
the rules.

The module also computes the checksums that a manifest holds:

- a file's ``sha256``, by ``file_sha256`` for a file on disk, or by ``sha256``
  for bytes in memory. Both give the same value for the same bytes.
- a table's ``content_sha256``, by ``table_sha256``, from the table's content
"""

import csv
import hashlib
import json
import subprocess
import sys

import pandas as pd
import pyarrow as pa

from embedding_model.folders import staged


def sha256(file_bytes):
    """The SHA-256 of bytes that are in memory already, as hex.

    It gives the same value as ``file_sha256`` of a file with these bytes. Use
    it when the code holds the bytes anyway, for example a download just
    received.
    For a file on disk, use ``file_sha256``, which never holds the file whole.

    Args:
        file_bytes (bytes): The bytes.

    Returns:
        str: The SHA-256, as hex.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def file_digest(path, algorithm):
    """A file's checksum by one of hashlib's algorithms, as hex, read in chunks.

    file_digest reads the file piece by piece, so even a large file is never
    in memory whole. ``file_sha256`` uses it for SHA-256. LOINC's fetch uses it
    for MD5, the checksum that LOINC publishes for each release. A checksum
    here checks a file, not a secret, so file_digest makes the hash with
    ``usedforsecurity=False``. An OpenSSL in FIPS mode refuses MD5 without it.

    Args:
        path (Path): The file.
        algorithm (str): A hashlib algorithm, for example ``sha256`` or ``md5``.

    Returns:
        str: The checksum, as hex.
    """
    with open(path, "rb") as stream:
        return hashlib.file_digest(
            stream, lambda: hashlib.new(algorithm, usedforsecurity=False)
        ).hexdigest()


def file_sha256(path):
    """The SHA-256 of a file's bytes, as hex, read in chunks.

    It gives the same value as ``sha256`` of the file's bytes, but it never
    holds the whole file in memory. So it suits a large file, for example a
    release zip or a corpus's pairs. A manifest holds it as a file's
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
    piece. So a table of millions of rows, for example a corpus's pairs, is
    hashed without its whole CSV text in memory.
    """

    def __init__(self):
        """Start an empty SHA-256."""
        self.hash = hashlib.sha256()

    def write(self, text):
        """Add text to the hash, as UTF-8, and return its length."""
        self.hash.update(text.encode("utf-8"))
        return len(text)

    def __iter__(self):
        """Nothing to read: pandas takes only an iterable as a file."""
        return iter(())


def schema_line(frame):
    """A table's column names and their pyarrow types, as one line of text.

    The types are those that pyarrow gives the frame's columns. schema_line
    reads ``large_string`` and ``null`` as ``string``, so the line does not
    depend on how pandas holds a column.

    Args:
        frame (pd.DataFrame): The table.

    Returns:
        str: The line, for example ``name: string, open: bool``.
    """
    schema = pa.Schema.from_pandas(frame, preserve_index=False)
    return ", ".join(
        f"{field.name}: {AS_STRING.get(str(field.type), str(field.type))}"
        for field in schema
    )


def table_sha256(frame):
    """The SHA-256 of a table's content: its schema line, then its rows as CSV.

    A manifest holds it as a table's ``content_sha256``. table_sha256 hashes the
    content, not the parquet file, because the file's bytes depend on the
    library that wrote it. The schema line keeps the types that CSV loses, so
    ``True`` and the text ``"True"`` give different checksums. table_sha256
    writes a missing value as MISSING, so it differs from an empty text and
    from the text ``\\N``.

    table_sha256 sets the separator, the quoting and the line ending. Lines end
    with ``\\n`` on every system, so the checksum is the same on Windows. pandas
    writes the CSV to the hash piece by piece, through ``Digest``, so the whole
    text is never in memory. Hashing the pieces in order gives the SHA-256 of
    the whole text.

    Args:
        frame (pd.DataFrame): The table.

    Returns:
        str: The SHA-256, as hex.
    """
    # Each backslash in a text is written twice, as PostgreSQL writes it, so the
    # text "\N" cannot read like MISSING. Only the text columns that hold a
    # backslash are copied, to keep a large table's memory low. infer_dtype
    # finds a column of texts by its values, missing ones aside.
    escaped = {
        name: frame[name].str.replace("\\", "\\\\", regex=False)
        for name in frame.columns
        if pd.api.types.infer_dtype(frame[name], skipna=True) == "string"
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

    ``differences`` compares a parquet table by its rows and its
    ``content_sha256``, the SHA-256 of its content. The table's file bytes
    depend on the library that wrote them, so the same table can have another
    ``sha256`` on another machine. ``differences`` compares any other file by
    its rows and its ``sha256``.

    Args:
        file_entry (dict): The file's entry in ``files``.

    Returns:
        dict: The file's rows, and the one checksum that it is compared by.
    """
    key = "content_sha256" if "content_sha256" in file_entry else "sha256"
    return {"rows": file_entry.get("rows"), key: file_entry.get(key)}


def identities(manifest):
    """What a build is compared by: its files, then its sources.

    identities identifies a file as ``file_identity`` says. It identifies a
    source by its SHA-256, or by its version where the source has no SHA-256.

    Args:
        manifest (dict): A manifest, or only its ``files``, as in
            ``{"files": ...}``.

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
            for source in manifest.get("sources", [])
        },
    }


def differences(committed, built_manifest):
    """What differs between a committed manifest and a new build's manifest.

    differences identifies a source by its SHA-256 where the source has one,
    and otherwise by its version. So a file with the same content, under a new
    release label, is the same source. differences does not compare dates,
    counts, citations or licenses.

    Args:
        committed (dict): The manifest under ``manifests/``.
        built_manifest (dict): The manifest of the build just made.

    Returns:
        list[str]: One line per difference, or an empty list when the two
            match.
    """
    before, after = identities(committed), identities(built_manifest)
    keys = [*before, *(key for key in after if key not in before)]
    return [
        f"{key}: {before.get(key)} -> {after.get(key)}"
        for key in keys
        if before.get(key) != after.get(key)
    ]


def git(folder, *args):
    """Run git in a folder, and return what it prints.

    Args:
        folder (Path): Where git runs.
        *args (str): git's own arguments, for example ``ls-tree``.

    Returns:
        str: git's output.

    Raises:
        RuntimeError: If git cannot run, for example when it is missing, or if
            git fails. When git fails, the error holds git's own message, for
            example ``fatal: not a git repository``.
    """
    try:
        completed = subprocess.run(
            ["git", *args], cwd=folder, capture_output=True, text=True
        )
    except Exception as error:
        raise RuntimeError(f"git could not run: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {args[0]} failed")
    return completed.stdout


def committed_version(folder, name):
    """The version of a manifest as committed in git, at ``HEAD``.

    committed_version runs git in the folder, and finds the manifest in git's
    own listing of the folder. So no text from the command line reaches a git
    command.

    Args:
        folder (Path): The ``manifests/`` folder.
        name (str): The manifest's name, without ``.json``.

    Returns:
        str | None: The version, or None when the manifest is not committed
            yet.

    Raises:
        RuntimeError: From ``git``, if git cannot answer, in each of these
            cases:

            - outside a checkout
            - with git missing
            - when git refuses the folder
            - in a checkout without a commit

            The error holds git's own message, or says that git could not run.
    """
    listed = git(folder, "ls-tree", "--name-only", "HEAD").splitlines()
    manifest_file = f"{name}.json"
    listed_file = next((file for file in listed if file == manifest_file), None)
    if listed_file is None:
        return None
    return json.loads(git(folder, "show", f"HEAD:./{listed_file}"))["version"]


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

    Each step reads its inputs' manifests with read, so every error names
    the file and the command that writes the file again.

    Args:
        path (Path): The manifest.
        command (str): The command that writes the manifest again, for the
            messages.
        field (str): The field that names what was built, for example
            ``source``.
        name (str): The name that the field must hold, for example ``ncit``.

    Returns:
        dict: The manifest.

    Raises:
        FileNotFoundError: If there is no manifest at ``path``.
        ValueError: In each of these cases:

            - the manifest cannot be read
            - the manifest is not a JSON object
            - the manifest is the manifest of another build
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: {command}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(
            f"{path} cannot be read ({type(error).__name__}: {error})."
            f" Run: {command}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} is not a manifest. Run: {command}")
    if manifest.get(field) != name:
        raise ValueError(
            f"{path} is the manifest of {field} {manifest.get(field)!r}."
            f" Run: {command}"
        )
    return manifest


def check_file(path, manifest, manifest_path, command):
    """Check that a file exists, and has the SHA-256 that its manifest holds.

    The manifest lists each file under its name, in ``files``. So a file
    that differs from its manifest changed after its step wrote it.

    Args:
        path (Path): The file.
        manifest (dict): The manifest, from ``read``.
        manifest_path (Path): Where the manifest is, for the messages.
        command (str): The command that writes the file again, for the
            messages.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the manifest holds no SHA-256 for the file, or the
            file differs from it.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: {command}")
    try:
        checksum = manifest["files"][path.name]["sha256"]
    except Exception as error:
        raise ValueError(
            f"{manifest_path} holds no SHA-256 for {path.name}"
            f" ({type(error).__name__}: {error}). Run: {command}"
        ) from error
    if file_sha256(path) != checksum:
        raise ValueError(f"{path} differs from its manifest. Run: {command}")


def warn_if_not_committed(label, version, committed_path, command):
    """Print a warning when a build's version is not the committed version.

    Args:
        label (str): What was built and where, for the message.
        version (str): The build's version.
        committed_path (Path): The committed manifest. warn_if_not_committed
            prints nothing when there is none.
        command (str): The command that builds the committed version.
    """
    if not committed_path.exists():
        return
    committed = json.loads(committed_path.read_text(encoding="utf-8"))["version"]
    if committed != version:
        print(
            f"warning: {label} is version {version}, and the committed manifest "
            f"is version {committed}. For version {committed}, run: {command}",
            file=sys.stderr,
        )


class Refused(Exception):
    """A build that differs from its committed manifest, and is not accepted."""


def decide(committed_path, built_manifest, update_manifest, command):
    """Whether a build may replace the working one: raise Refused if it may not.

    A build that matches its committed manifest may replace the working one.
    decide accepts a build that differs only with ``update_manifest``, under a
    version not committed in git. Otherwise decide prints the differences, with
    the steps that accept the build, and raises Refused. ``write_checked``
    calls decide inside ``folders.staged``. So ``staged`` deletes a refused
    build, and keeps the working one.

    Args:
        committed_path (Path): The committed manifest.
        built_manifest (dict): The new build's manifest.
        update_manifest (bool): Whether to accept a build that differs.
        command (str): The command that made the build, for the messages.

    Returns:
        list[str]: The differences, or an empty list when the build matches.

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
    for file, file_record in built_manifest["files"].items():
        print(f"  {directory / file}: {file_record['rows']:,} rows")
    if differing:
        write(committed_path, built_manifest)
        print(f"  accepted, and recorded in {committed_path}")
    else:
        print(f"  matches {committed_path.name}")


def write_checked(destination, write, committed_path, update_manifest, command):
    """Write a build whole, and keep it only if its committed manifest allows.

    write_checked writes the build through ``folders.staged``. While the build
    is still in the staging folder, ``decide`` says whether the build may
    replace the working folder. ``staged`` deletes a refused build, and keeps
    the working folder. write_checked calls ``record`` only when the build is
    in place.

    Args:
        destination (Path): The build's folder, for example
            ``sources/<source>/``.
        write (Callable[[Path], dict]): Writes the build into the folder that
            it is given, and returns the build's manifest.
        committed_path (Path): The committed manifest.
        update_manifest (bool): Whether to accept a build that differs.
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
