"""Tests for embedding_model.manifests: comparing builds, and what git says."""

import json
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from embedding_model import folders, manifests

PIN = {"role": "mappings", "name": "Fictional source", "version": "1"}


class TestIdentities:
    """What identifies a build: its files, and each source's SHA-256 or version."""

    def test_files_then_sources_by_sha256_or_version(self):
        """Files come first; a source is its SHA-256, else its version."""
        manifest = {
            "files": {"abcd_mappings.csv": {"rows": 1, "sha256": "aa"}},
            "sources": [
                {**PIN, "sha256": "cc", "version": "2"},
                {"role": "source text", "name": "Abcd release", "version": "v1"},
            ],
        }
        assert manifests.identities(manifest) == {
            "abcd_mappings.csv": {"rows": 1, "sha256": "aa"},
            "mappings 'Fictional source'": "cc",
            "source text 'Abcd release'": "v1",
        }


class TestDifferences:
    """Which changes between two manifests count as differences."""

    MANIFEST = {
        "version": "1",
        "built": "2026-01-01",
        "files": {"abcd_mappings.csv": {"rows": 1, "sha256": "aa"}},
        "sources": [{**PIN, "sha256": "cc", "fetched": "2026-01-01"}],
    }

    def test_dates_and_release_labels_do_not_count(self):
        """A rebuild on another day, from a source with the same SHA-256, matches."""
        later = {
            **self.MANIFEST,
            "built": "2026-02-01",
            "sources": [
                {**PIN, "sha256": "cc", "version": "2", "fetched": "2026-02-01"}
            ],
        }
        assert manifests.differences(self.MANIFEST, later) == []

    def test_changed_file_and_source_are_named(self):
        """A file and a source with another SHA-256 are each named."""
        changed = {
            **self.MANIFEST,
            "files": {"abcd_mappings.csv": {"rows": 1, "sha256": "bb"}},
            "sources": [{**PIN, "sha256": "dd"}],
        }
        found = manifests.differences(self.MANIFEST, changed)
        assert len(found) == 2
        assert found[0].startswith("abcd_mappings.csv")
        assert "Fictional source" in found[1]

    def test_a_parquet_table_is_compared_by_its_content(self):
        """A table with the same rows and content matches, though its file's bytes differ."""
        table = {"rows": 2, "sha256": "aa", "csv_sha256": "cc"}
        committed = {**self.MANIFEST, "files": {"abcd.parquet": table}}
        rewritten = {
            **self.MANIFEST,
            "files": {"abcd.parquet": {**table, "sha256": "bb"}},
        }
        assert manifests.differences(committed, rewritten) == []
        changed = {
            **self.MANIFEST,
            "files": {"abcd.parquet": {**table, "csv_sha256": "dd"}},
        }
        assert manifests.differences(committed, changed) != []

    def test_a_table_without_its_file_hash_matches_one_with_it(self):
        """A committed table that records only csv_sha256 matches a build with both."""
        committed = {
            **self.MANIFEST,
            "files": {"abcd.parquet": {"rows": 2, "csv_sha256": "cc"}},
        }
        built = {
            **self.MANIFEST,
            "files": {"abcd.parquet": {"rows": 2, "sha256": "aa", "csv_sha256": "cc"}},
        }
        assert manifests.differences(committed, built) == []


class TestPinned:
    """An input as a committed manifest pins it."""

    def test_the_named_input(self):
        """The input of that name is returned, with its version."""
        manifest = {"sources": [PIN, {"name": "Abcd release", "version": "2"}]}
        assert manifests.pinned(manifest, "Abcd release")["version"] == "2"

    def test_a_missing_input_is_named(self):
        """A manifest without the named input raises, naming it."""
        with pytest.raises(KeyError, match="Abcd release"):
            manifests.pinned({"sources": []}, "Abcd release")


class TestRead:
    """Reading a manifest that a step wrote, with the command that writes it again."""

    COMMAND = "python -m embedding_model abcd build"

    def read(self, path):
        """The manifest at ``path``, read as source abcd's."""
        return manifests.read(path, self.COMMAND, "source", "abcd")

    def test_a_manifest_reads_back(self, tmp_path):
        """A JSON object that names its build reads back as it was written."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"source": "abcd", "files": {}}))
        assert self.read(path) == {"source": "abcd", "files": {}}

    def test_a_written_manifest_reads_back(self, tmp_path):
        """A manifest is written as indented JSON with a final newline, and reads back."""
        path = tmp_path / "manifest.json"
        manifest = {"source": "abcd", "files": {"abcd.parquet": {"rows": 1}}}
        manifests.write(path, manifest)
        assert path.read_text().endswith("}\n")
        assert '\n  "files": {' in path.read_text()
        assert self.read(path) == manifest

    def test_a_missing_manifest_names_the_command(self, tmp_path):
        """A manifest never written is named, with the command that writes it."""
        with pytest.raises(FileNotFoundError, match=f"missing; run: {self.COMMAND}"):
            self.read(tmp_path / "manifest.json")

    @pytest.mark.parametrize("text", ["", "{", "[1, 2]"])
    def test_a_manifest_that_is_not_a_json_object_is_refused(self, text, tmp_path):
        """Text that is not JSON, or JSON that is not an object, is refused."""
        path = tmp_path / "manifest.json"
        path.write_text(text)
        with pytest.raises(ValueError, match=f"manifest.json .*run: {self.COMMAND}"):
            self.read(path)

    @pytest.mark.parametrize("manifest", [{"source": "efgh"}, {"corpus": "abcd"}])
    def test_another_builds_manifest_is_refused(self, manifest, tmp_path):
        """A manifest of another source, or of another kind of build, is refused."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="is the manifest of source"):
            self.read(path)


class TestDecide:
    """Whether a build may replace the working one, against its committed manifest."""

    COMMAND = "python -m embedding_model abcd build"
    BUILT = {
        "version": "1",
        "files": {"abcd.csv": {"rows": 1, "sha256": "aa"}},
        "sources": [PIN],
    }

    @pytest.fixture
    def committed(self, tmp_path):
        """A pins-only committed manifest, version 1, that every build differs from."""
        path = tmp_path / "abcd.json"
        path.write_text(json.dumps({"version": "1", "files": {}, "sources": [PIN]}))
        return path

    def test_a_matching_build_may_be_kept(self, tmp_path, monkeypatch):
        """A build that matches its committed manifest is kept, and git is not asked."""
        path = tmp_path / "abcd.json"
        path.write_text(json.dumps(self.BUILT))
        no_git = MagicMock(side_effect=RuntimeError("git could not run"))
        monkeypatch.setattr(manifests, "committed_version", no_git)
        assert manifests.decide(path, self.BUILT, False, self.COMMAND) == []
        no_git.assert_not_called()

    def test_a_differing_build_needs_update_manifest(
        self, committed, monkeypatch, capsys
    ):
        """Without --update-manifest, a differing build is refused, with the command."""
        monkeypatch.setattr(manifests, "committed_version", lambda *_: None)
        with pytest.raises(manifests.Refused):
            manifests.decide(committed, self.BUILT, False, self.COMMAND)
        error = capsys.readouterr().err
        assert "abcd.csv" in error
        assert f"run: {self.COMMAND} --update-manifest" in error

    def test_a_build_under_an_uncommitted_version_is_accepted(
        self, committed, monkeypatch
    ):
        """With --update-manifest and a version not committed, the build is accepted."""
        monkeypatch.setattr(manifests, "committed_version", lambda *_: None)
        found = manifests.decide(committed, self.BUILT, True, self.COMMAND)
        assert found and found[0].startswith("abcd.csv")

    @pytest.mark.parametrize("update_manifest", [False, True])
    def test_a_committed_version_gives_both_steps(
        self, update_manifest, committed, monkeypatch, capsys
    ):
        """Under a version that git has committed, the refusal says to raise it first."""
        monkeypatch.setattr(manifests, "committed_version", lambda *_: "1")
        with pytest.raises(manifests.Refused):
            manifests.decide(committed, self.BUILT, update_manifest, self.COMMAND)
        error = capsys.readouterr().err
        assert f'raise "version" in {committed}' in error
        assert f"then run: {self.COMMAND} --update-manifest" in error

    def test_no_answer_from_git_refuses(self, committed, monkeypatch):
        """When git cannot say whether the version is committed, the build is refused."""
        no_answer = MagicMock(side_effect=RuntimeError("not a git repository"))
        monkeypatch.setattr(manifests, "committed_version", no_answer)
        with pytest.raises(
            manifests.Refused, match="git cannot say.*`git status` works"
        ):
            manifests.decide(committed, self.BUILT, True, self.COMMAND)


class TestRecord:
    """Reporting a build that is in place, and recording an accepted one."""

    def test_an_accepted_build_is_recorded(self, tmp_path, capsys):
        """A build with differences is written as the committed manifest."""
        committed = tmp_path / "abcd.json"
        built = {**TestDecide.BUILT, "counts": {"rows": 1}}
        manifests.record(committed, built, ["abcd.csv: ..."], tmp_path / "abcd")
        assert json.loads(committed.read_text()) == built
        assert "accepted, and recorded in" in capsys.readouterr().out

    def test_a_matching_build_leaves_the_committed_manifest(self, tmp_path, capsys):
        """A build that matches is reported, and the committed manifest is not written."""
        committed = tmp_path / "abcd.json"
        committed.write_text("{}")
        manifests.record(committed, TestDecide.BUILT, [], tmp_path / "abcd")
        assert committed.read_text() == "{}"
        assert "matches abcd.json" in capsys.readouterr().out


class TestWriteChecked:
    """Writing a build whole, kept only if its committed manifest allows."""

    @pytest.fixture
    def committed(self, tmp_path, monkeypatch):
        """A pins-only committed manifest, under a version that git has not committed."""
        path = tmp_path / "abcd.json"
        path.write_text(json.dumps({"version": "1", "files": {}, "sources": [PIN]}))
        monkeypatch.setattr(manifests, "committed_version", lambda *_: None)
        return path

    @staticmethod
    def write(folder):
        """Write one file of the build, and return the build's manifest."""
        (folder / "abcd.csv").write_text("a\n1\n")
        return TestDecide.BUILT

    def test_a_refused_build_leaves_nothing(self, committed, tmp_path):
        """Without --update-manifest, a differing build is deleted, and exits 1."""
        status = manifests.write_checked(
            tmp_path / "abcd", self.write, committed, False, "abcd build"
        )
        assert status == 1
        assert not (tmp_path / "abcd").exists()

    def test_an_accepted_build_is_in_place_and_recorded(self, committed, tmp_path):
        """An accepted build replaces the folder, and is written as the committed manifest."""
        status = manifests.write_checked(
            tmp_path / "abcd", self.write, committed, True, "abcd build"
        )
        assert status == 0
        assert (tmp_path / "abcd" / "abcd.csv").exists()
        assert json.loads(committed.read_text())["files"] == TestDecide.BUILT["files"]

    def test_a_build_that_cannot_move_into_place_is_not_recorded(
        self, committed, tmp_path, monkeypatch
    ):
        """A failed swap leaves the committed manifest as it was."""

        def failing(*arguments):
            """A swap that fails, as on a full disk."""
            raise OSError("no space left on device")

        monkeypatch.setattr(folders, "replace", failing)
        with pytest.raises(OSError):
            manifests.write_checked(
                tmp_path / "abcd", self.write, committed, True, "abcd build"
            )
        assert json.loads(committed.read_text())["files"] == {}


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
class TestCommittedVersion:
    """Reading a manifest's version as committed in git."""

    def git(self, folder, *args):
        """Run git in a folder, as a fictional committer."""
        subprocess.run(
            ["git", "-c", "user.name=Abcd", "-c", "user.email=abcd@example.org",
             "-C", str(folder), *args],
            check=True, capture_output=True,
        )  # fmt: skip

    def test_the_version_at_head(self, tmp_path):
        """The committed version is read from HEAD, not from the file on disk."""
        folder = tmp_path / "manifests"
        folder.mkdir()
        (folder / "abcd.json").write_text(json.dumps({"version": "1"}))
        self.git(tmp_path, "init", "-q")
        self.git(tmp_path, "add", ".")
        self.git(tmp_path, "commit", "-q", "-m", "abcd")
        (folder / "abcd.json").write_text(json.dumps({"version": "2"}))
        assert manifests.committed_version(folder, "abcd") == "1"
        assert manifests.committed_version(folder, "efgh") is None

    def test_outside_a_checkout_is_an_error(self, tmp_path, monkeypatch):
        """Without a git checkout to ask, the error holds git's own message."""
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        with pytest.raises(RuntimeError, match="not a git repository"):
            manifests.committed_version(tmp_path, "abcd")

    def test_a_checkout_without_a_commit_is_an_error(self, tmp_path):
        """Without a HEAD to read, the answer is an error, not 'not committed'."""
        self.git(tmp_path, "init", "-q")
        with pytest.raises(RuntimeError):
            manifests.committed_version(tmp_path, "abcd")

    def test_a_manifest_that_git_cannot_show_is_an_error(self, monkeypatch, tmp_path):
        """A listed manifest that git then cannot show is an error, not 'not committed'."""
        (tmp_path / "abcd.json").write_text(json.dumps({"version": "1"}))
        self.git(tmp_path, "init", "-q")
        self.git(tmp_path, "add", ".")
        self.git(tmp_path, "commit", "-q", "-m", "abcd")
        run = manifests.subprocess.run

        def failing_show(command, **options):
            """git, except that ``git show`` fails with a message."""
            if command[1] == "show":
                return subprocess.CompletedProcess(command, 128, "", "fatal: abcd")
            return run(command, **options)

        monkeypatch.setattr(manifests.subprocess, "run", failing_show)
        with pytest.raises(RuntimeError, match="fatal: abcd"):
            manifests.committed_version(tmp_path, "abcd")

    def test_git_missing_is_an_error(self, monkeypatch, tmp_path):
        """When git cannot run at all, the answer is an error."""
        missing = MagicMock(side_effect=FileNotFoundError("git"))
        monkeypatch.setattr(manifests.subprocess, "run", missing)
        with pytest.raises(RuntimeError, match="git could not run"):
            manifests.committed_version(tmp_path, "abcd")
