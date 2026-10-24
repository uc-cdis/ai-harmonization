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
import shutil
import subprocess
import sys
from pathlib import Path


def identities(manifest):
    """What a build is compared by: its files, then its sources.

    A file is identified by its rows and SHA-256, and a source by its SHA-256,
    or by its version where it has none.

    Args:
        manifest (dict): A manifest.

    Returns:
        dict: One entry per file, by name, and per source, by role and name.
    """
    return {
        **manifest["files"],
        **{
            f"{source['role']} {source['name']!r}": source.get("sha256")
            or source.get("version")
            for source in manifest["sources"]
        },
    }


def differences(committed, built):
    """What differs between a committed manifest and a new build's manifest.

    A source is identified by its SHA-256 where it has one, and otherwise by
    its version, so a file served under a new release label but with the
    same content is the same source. Dates, citations and licenses are not
    compared.

    Args:
        committed (dict): The manifest under ``manifests/``.
        built (dict): The manifest of the build just made.

    Returns:
        list[str]: One line per difference; empty when the two match.
    """
    before, after = identities(committed), identities(built)
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
            f"is version {committed}. Run {command} for version {committed}.",
            file=sys.stderr,
        )


def settle(committed_path, built, staging, directory, files, update_manifest):
    """Keep a build that matches its committed manifest, or an accepted build.

    A build that matches replaces the working files. A build that differs
    replaces them only when ``update_manifest`` accepts it, under a version not
    committed in git. It is then recorded as the committed manifest. Otherwise
    the differences are printed, and the working files are left as they were.

    Args:
        committed_path (Path): The committed manifest.
        built (dict): The new build's manifest; its ``files`` are listed with
            their row counts.
        staging (Path): Where the build was written.
        directory (Path): Where the working files are kept.
        files (list[str]): The files to move from ``staging`` to
            ``directory``.
        update_manifest (bool): Accept a build that differs.

    Returns:
        int: 0 if the build was kept, 1 if it was refused.
    """
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    found = differences(committed, built)
    refusal = None
    if found and not update_manifest:
        refusal = "pass --update-manifest to accept this build"
    elif found:
        try:
            version = committed_version(committed_path.parent, committed_path.stem)
        except RuntimeError as error:
            refusal = (
                f"git cannot say whether version {committed['version']} "
                f"is committed ({error}); fix that to accept this build"
            )
        else:
            if version == committed["version"]:
                refusal = (
                    f"version {committed['version']} is committed; raise "
                    f'"version" in {committed_path} to accept this '
                    "build as a new version"
                )
    if refusal:
        print(f"  differs from {committed_path.name}; working files kept:")
        for line in found:
            print(f"    {line}")
        print(f"  {refusal}")
        return 1
    directory.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.move(Path(staging) / file, directory / file)
    for file, entry in built["files"].items():
        print(f"  {directory / file}: {entry['rows']:,} rows")
    if found:
        committed_path.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")
        print(f"  accepted, and recorded in {committed_path}")
    else:
        print(f"  matches {committed_path.name}")
    return 0
