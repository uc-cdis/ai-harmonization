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
        error = capsys.readouterr().err
        assert f"It is kept in {staging / '.previous'}" in error
        assert f"Move it back to {destination}." in error

    def test_a_staging_folder_left_by_a_stopped_run_is_warned_about(
        self, tmp_path, capsys
    ):
        """The next run says that a stopped run's half-written folder can be deleted."""
        left = tmp_path / ".abcd.k1ll3d"
        write_folder(left / "abcd", "half written")
        with staged(tmp_path / "abcd") as built:
            (built / "file.txt").write_text("new")
        assert capsys.readouterr().err == (
            f"warning: {left} is left from a run that was stopped. Nothing in it"
            " is needed, so delete it.\n"
        )
        assert left.exists()

    def test_a_previous_folder_left_by_a_stopped_swap_is_warned_about(
        self, tmp_path, capsys
    ):
        """The warning says to move back a previous folder that a stopped swap left."""
        left = tmp_path / ".abcd.k1ll3d"
        write_folder(left / "abcd", "new")
        write_folder(left / ".previous", "previous")
        destination = tmp_path / "abcd"
        with staged(destination) as built:
            (built / "file.txt").write_text("newer")
        assert capsys.readouterr().err == (
            f"warning: {left} is left from a run that was stopped. It holds the"
            f" previous abcd, in {left / '.previous'}. If this run fails, move it"
            f" back to {destination}. Otherwise delete {left}.\n"
        )
        assert (left / ".previous" / "file.txt").read_text() == "previous"

    def test_a_previous_folder_already_replaced_can_be_deleted(self, tmp_path, capsys):
        """A run stopped after both renames left only a replaced folder, which can be deleted."""
        left = tmp_path / ".abcd.k1ll3d"
        write_folder(left / ".previous", "previous")
        write_folder(tmp_path / "abcd", "new")
        with staged(tmp_path / "abcd") as built:
            (built / "file.txt").write_text("newer")
        assert capsys.readouterr().err == (
            f"warning: {left} is left from a run that was stopped. Nothing in it"
            " is needed, so delete it.\n"
        )

    def test_the_staging_folder_of_a_longer_name_is_not_warned_about(
        self, tmp_path, capsys
    ):
        """A run of abcd says nothing about the staging folder of abcd.v2."""
        write_folder(tmp_path / ".abcd.v2.k1ll3d" / ".previous", "previous")
        with staged(tmp_path / "abcd") as built:
            (built / "file.txt").write_text("new")
        assert capsys.readouterr().err == ""
