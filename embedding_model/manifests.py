"""What a build was made from, and whether a new build is the same.

Benchmarks and sources each keep a committed manifest per build, under their
own ``manifests/`` folder. It pins what the build is made from, such as a
commit, a release or a record, and records the SHA-256 of every file built.
A new build writes its own manifest, and the two are compared. A build that
matches is kept. A build that differs is kept only when accepted as a new
version. A version committed in git never changes.
``embedding_model/docs/manifests.md`` describes the fields and the rules.
"""

import json
import subprocess
import sys

from embedding_model.folders import staged


def file_identity(file_entry):
    """What a file is compared by, from its entry in a manifest's ``files``.

    A parquet table has ``csv_sha256``, the SHA-256 of its content, and is
    compared by it and its rows. Its file's bytes depend on the library that
    wrote it, so the same table can have another ``sha256`` on another machine.
    Any other file is compared by its rows and its ``sha256``.

    Args:
        file_entry (dict): The file's entry in ``files``.

    Returns:
        dict: Its rows, and the one checksum that it is compared by.
    """
    key = "csv_sha256" if "csv_sha256" in file_entry else "sha256"
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
    """Run git in a folder, and give back its result, with the output as text.

    Args:
        folder (Path): Where git runs.
        *args (str): git's own arguments, such as ``rev-parse``.

    Returns:
        subprocess.CompletedProcess: The result, with ``stdout`` and ``stderr``
            as text.

    Raises:
        RuntimeError: If git cannot run at all, such as when it is missing.
    """
    try:
        return subprocess.run(
            ["git", *args], cwd=folder, capture_output=True, text=True
        )
    except Exception as error:
        raise RuntimeError(f"git could not run: {error}") from error


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
        RuntimeError: If git cannot be asked: outside a checkout, with git
            missing, or when git refuses the folder.
    """
    inside = git(folder, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0:
        raise RuntimeError(inside.stderr.strip() or "not a git checkout")
    listed = git(folder, "ls-tree", "--name-only", "HEAD")
    wanted = f"{name}.json"
    entry = next((e for e in listed.stdout.splitlines() if e == wanted), None)
    if listed.returncode != 0 or entry is None:
        return None
    shown = git(folder, "show", f"HEAD:./{entry}")
    if shown.returncode != 0:
        return None
    return json.loads(shown.stdout)["version"]


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
            f"git cannot say whether version {version} is committed ({error});"
            " fix that to accept this build"
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
