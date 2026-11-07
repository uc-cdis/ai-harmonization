"""Tests for embedding_model.provenance: the code's commit, a model's revision and the library versions."""

import shutil
import subprocess

import pytest
import yaml

from embedding_model import provenance


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
class TestCodeCommit:
    """The commit that scores and trained models record, and whether the code had uncommitted changes."""

    def git(self, folder, *args):
        """Run git in a folder, as a fictional committer."""
        subprocess.run(
            ["git", "-c", "user.name=Abcd", "-c", "user.email=abcd@example.org",
             "-C", str(folder), *args],
            check=True, capture_output=True,
        )  # fmt: skip

    @pytest.fixture
    def checkout(self, tmp_path):
        """A checkout with a module, a doc and the dependency files, all committed."""
        (tmp_path / "embedding_model" / "docs").mkdir(parents=True)
        (tmp_path / "embedding_model" / "abcd.py").write_text("ABCD = 1\n")
        (tmp_path / "embedding_model" / "docs" / "abcd.md").write_text("# Abcd\n")
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "uv.lock").write_text("version = 1\n")
        self.git(tmp_path, "init", "-q")
        self.git(tmp_path, "add", ".")
        self.git(tmp_path, "commit", "-q", "-m", "abcd")
        return tmp_path

    def test_a_changed_doc_is_not_an_uncommitted_change(self, checkout):
        """Only code and dependency files count; a doc does not."""
        (checkout / "embedding_model" / "docs" / "abcd.md").write_text("# Efgh\n")
        found = provenance.code_commit(checkout / "embedding_model")
        assert len(found["commit"]) == 40
        assert found["uncommitted_changes"] is False

    @pytest.mark.parametrize(
        "changed",
        [
            "embedding_model/abcd.py",
            "embedding_model/new/efgh.py",
            "pyproject.toml",
            "uv.lock",
        ],
    )
    def test_code_and_dependency_files_count(self, checkout, changed):
        """A changed or new Python file, or a changed dependency file, is an uncommitted change."""
        path = checkout / changed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed\n")
        found = provenance.code_commit(checkout / "embedding_model")
        assert found["uncommitted_changes"] is True

    def test_outside_a_checkout_nothing_is_recorded(self, tmp_path, monkeypatch):
        """Without git to ask, the commit and the flag are None."""
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        assert provenance.code_commit(tmp_path) == {
            "commit": None,
            "uncommitted_changes": None,
        }


class TestModelRevision:
    """The Hugging Face commit of the model that was scored."""

    def cached(self, folder, repo_id):
        """Put a fictional model in a Hugging Face cache, at commit abcd1234."""
        model = folder / ("models--" + repo_id.replace("/", "--"))
        (model / "snapshots" / "abcd1234").mkdir(parents=True)
        (model / "refs").mkdir()
        (model / "refs" / "main").write_text("abcd1234")

    def test_a_model_on_disk_has_none(self, tmp_path):
        """A model folder on disk has no Hugging Face commit."""
        assert provenance.model_revision(str(tmp_path)) is None

    def test_a_hugging_face_model_gives_its_cached_commit(self, monkeypatch, tmp_path):
        """A model loaded by id gives the commit that its cached copy was downloaded at."""
        self.cached(tmp_path, "abcd-org/abcd-model")
        monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path))
        assert provenance.model_revision("abcd-org/abcd-model") == "abcd1234"

    def test_a_model_not_in_the_cache_has_none(self, monkeypatch, tmp_path):
        """A model that the cache does not hold gives None, and scoring goes on."""
        monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path))
        assert provenance.model_revision("abcd-org/abcd-model") is None


def test_library_versions_are_text_that_yaml_writes():
    """Each version is a plain str, so a score's config.yaml can hold them."""
    versions = provenance.library_versions()
    assert {"python", "sentence_transformers", "torch"} <= set(versions)
    assert all(type(version) is str for version in versions.values())
    assert yaml.safe_load(yaml.safe_dump(versions)) == versions
