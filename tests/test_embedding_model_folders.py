"""Tests for embedding_model.folders: writing a folder whole, through a staging folder."""

import pathlib

import pytest

from embedding_model.folders import staged


def write_folder(destination, text):
    """A folder that holds one file, ``file.txt``, with the text given."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "file.txt").write_text(text)


def fail_renames(monkeypatch, failing):
    """Make the renames whose numbers are in ``failing`` raise OSError.

    The first rename moves the previous folder aside, the second moves the new
    folder into place, and the third, after a failure, puts the previous one
    back.
    """
    rename, calls = pathlib.Path.rename, []

    def renamed(self, target):
        """Rename, unless this rename's number is to fail."""
        calls.append(self)
        if len(calls) in failing:
            raise OSError("abcd")
        return rename(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", renamed)


class TestStaged:
    """A folder written inside ``staged`` replaces the previous one only when it is complete."""

    def test_a_finished_block_replaces_the_previous_folder(self, tmp_path):
        """The new folder takes the previous one's place, and no staging folder is left."""
        destination = tmp_path / "abcd"
        write_folder(destination, "previous")
        with staged(destination) as built:
            (built / "file.txt").write_text("new")
        assert (destination / "file.txt").read_text() == "new"
        assert [path.name for path in tmp_path.iterdir()] == ["abcd"]

    def test_a_block_that_raises_keeps_the_previous_folder(self, tmp_path):
        """A step that fails leaves the previous folder as it was, and no staging folder."""
        destination = tmp_path / "abcd"
        write_folder(destination, "previous")
        with pytest.raises(ValueError):
            with staged(destination) as built:
                (built / "file.txt").write_text("new")
                raise ValueError("the step failed")
        assert (destination / "file.txt").read_text() == "previous"
        assert [path.name for path in tmp_path.iterdir()] == ["abcd"]

    def test_a_failed_move_puts_the_previous_folder_back(self, tmp_path, monkeypatch):
        """When the new folder cannot move into place, the previous one is put back."""
        destination = tmp_path / "abcd"
        write_folder(destination, "previous")
        fail_renames(monkeypatch, {2})
        with pytest.raises(OSError):
            with staged(destination) as built:
                (built / "file.txt").write_text("new")
        assert (destination / "file.txt").read_text() == "previous"
        assert [path.name for path in tmp_path.iterdir()] == ["abcd"]

    def test_a_previous_folder_that_cannot_be_put_back_is_kept(
        self, tmp_path, monkeypatch, capsys
    ):
        """When putting the previous folder back fails too, it is kept and named."""
        destination = tmp_path / "abcd"
        write_folder(destination, "previous")
        fail_renames(monkeypatch, {2, 3})
        with pytest.raises(OSError):
            with staged(destination) as built:
                (built / "file.txt").write_text("new")
        (staging,) = tmp_path.glob(".abcd.*")
        assert (staging / ".previous" / "file.txt").read_text() == "previous"
        assert f"it is kept in {staging / '.previous'}" in capsys.readouterr().err

    def test_a_staging_folder_left_by_a_stopped_run_is_warned_about(
        self, tmp_path, capsys
    ):
        """The next run warns about a staging folder that a killed run left."""
        left = tmp_path / ".abcd.k1ll3d"
        write_folder(left / "abcd", "half written")
        with staged(tmp_path / "abcd") as built:
            (built / "file.txt").write_text("new")
        assert f"{left} is left from a run that was stopped" in capsys.readouterr().err
        assert left.exists()
