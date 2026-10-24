"""Tests for embedding_model.manifests: comparing builds, and what git says."""

import json
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from embedding_model import manifests

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


class TestSettle:
    """Keeping or refusing a build against its committed manifest."""

    @pytest.fixture
    def setup(self, tmp_path):
        """A pins-only committed manifest, and a build of one file in staging."""
        committed = tmp_path / "abcd.json"
        committed.write_text(
            json.dumps({"version": "1", "files": {}, "sources": [PIN]})
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "abcd.csv").write_text("a\n1\n")
        built = {
            "version": "1",
            "files": {"abcd.csv": {"rows": 1, "sha256": "aa"}},
            "sources": [PIN],
        }
        return committed, built, staging, tmp_path / "working"

    def test_a_differing_build_is_refused_without_update(self, setup, monkeypatch):
        """Without --update-manifest, a differing build is not kept."""
        committed, built, staging, working = setup
        monkeypatch.setattr(manifests, "committed_version", lambda *_: None)
        status = manifests.settle(
            committed, built, staging, working, ["abcd.csv"], False
        )
        assert status == 1
        assert not (working / "abcd.csv").exists()

    def test_an_accepted_build_is_kept_and_recorded(self, setup, monkeypatch):
        """Accepted under an uncommitted version, the build moves and is recorded."""
        committed, built, staging, working = setup
        monkeypatch.setattr(manifests, "committed_version", lambda *_: None)
        status = manifests.settle(
            committed, built, staging, working, ["abcd.csv"], True
        )
        assert status == 0
        assert (working / "abcd.csv").exists()
        assert json.loads(committed.read_text())["files"] == built["files"]

    def test_a_committed_version_is_not_redefined(self, setup, monkeypatch):
        """A differing build under the version git has committed is refused."""
        committed, built, staging, working = setup
        monkeypatch.setattr(manifests, "committed_version", lambda *_: "1")
        status = manifests.settle(
            committed, built, staging, working, ["abcd.csv"], True
        )
        assert status == 1
        assert json.loads(committed.read_text())["files"] == {}

    def test_no_answer_from_git_refuses(self, setup, monkeypatch):
        """When git cannot say whether the version is committed, the build is refused."""
        committed, built, staging, working = setup
        no_answer = MagicMock(side_effect=RuntimeError("not a git repository"))
        monkeypatch.setattr(manifests, "committed_version", no_answer)
        status = manifests.settle(
            committed, built, staging, working, ["abcd.csv"], True
        )
        assert status == 1


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

    def test_outside_a_checkout_is_an_error(self, tmp_path):
        """Without a git checkout to ask, the answer is an error, not 'not committed'."""
        with pytest.raises(RuntimeError):
            manifests.committed_version(tmp_path, "abcd")

    def test_git_missing_is_an_error(self, monkeypatch, tmp_path):
        """When git cannot run at all, the answer is an error."""
        missing = MagicMock(side_effect=FileNotFoundError("git"))
        monkeypatch.setattr(manifests.subprocess, "run", missing)
        with pytest.raises(RuntimeError, match="git could not run"):
            manifests.committed_version(tmp_path, "abcd")
