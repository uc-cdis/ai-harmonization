"""Tests for embedding_model.download: downloading once, keeping files whole, and requests."""

from unittest.mock import MagicMock

import pytest

from embedding_model import download


def served_html():
    """A session whose every GET returns an HTML page, as its Content-Type says."""
    client = MagicMock()
    client.get.return_value.content = b"<html><a href='x'></a></html>"
    client.get.return_value.headers = {"Content-Type": "text/html; charset=utf-8"}
    return client


class TestCached:
    """Downloading a file once, and keeping only the real file."""

    def test_downloads_once_and_leaves_no_partial_file(self, monkeypatch, tmp_path):
        """The first call fetches and keeps the file; later calls read it back."""
        fetch = MagicMock(return_value=b"data")
        monkeypatch.setattr(download, "fetch", fetch)
        path = tmp_path / "sub" / "file.bin"
        assert download.cached("https://example.org/f", path) == b"data"
        assert download.cached("https://example.org/f", path) == b"data"
        fetch.assert_called_once_with(
            "https://example.org/f", json=None, auth=None, html_ok=False
        )
        assert list(path.parent.iterdir()) == [path]

    def test_an_html_answer_is_not_kept(self, monkeypatch, tmp_path):
        """An HTML page served for a spreadsheet raises, and nothing is kept."""
        monkeypatch.setattr(download, "session", lambda: served_html())
        path = tmp_path / "table.xlsx"
        with pytest.raises(ValueError, match="HTML page"):
            download.cached("https://example.org/table.xlsx", path)
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_check_keeps_nothing(self, monkeypatch, tmp_path):
        """A download that its check refuses raises, and nothing is kept."""
        monkeypatch.setattr(download, "fetch", lambda url, **request: b"abcd")
        check = MagicMock(side_effect=ValueError("not the file"))
        with pytest.raises(ValueError, match="not the file"):
            download.downloaded(
                "https://example.org/f", tmp_path / "f.bin", check=check
            )
        assert list(tmp_path.iterdir()) == []

    def test_a_listing_page_is_kept(self, monkeypatch, tmp_path):
        """A folder listing is HTML by nature, so it is kept."""
        monkeypatch.setattr(download, "session", lambda: served_html())
        path = tmp_path / "listing.html"
        assert download.downloaded("https://example.org/", path) == path
        assert path.read_bytes() == b"<html><a href='x'></a></html>"


class TestFetch:
    """How ``download.fetch`` asks a server."""

    def test_a_body_is_posted_and_credentials_are_sent(self, monkeypatch):
        """Without a body the request is a GET; with one, a POST. Credentials go with both."""
        client = MagicMock()
        client.get.return_value.content = b"abcd"
        monkeypatch.setattr(download, "session", lambda: client)
        url = "https://example.org/a"
        assert download.fetch(url, auth=("abcd", "efgh")) == b"abcd"
        download.fetch(url, json={"abcd": 1})
        client.get.assert_called_once_with(
            url, auth=("abcd", "efgh"), timeout=download.TIMEOUT_SECONDS
        )
        client.post.assert_called_once_with(
            url, json={"abcd": 1}, auth=None, timeout=download.TIMEOUT_SECONDS
        )

    def test_an_html_page_is_refused_unless_asked_for(self, monkeypatch):
        """A text/html response raises with its URL, unless html_ok asks for a page."""
        monkeypatch.setattr(download, "session", lambda: served_html())
        url = "https://example.org/api"
        with pytest.raises(ValueError, match=f"{url} returned an HTML page"):
            download.fetch(url)
        assert download.fetch(url, html_ok=True).startswith(b"<html>")

    def test_the_session_retries_a_busy_server(self):
        """A busy server's answer, 429 or 503, is tried again, after a wait, for a POST too."""
        retries = download.session().get_adapter("https://example.org").max_retries
        assert retries.total > 0
        assert retries.backoff_factor > 0
        assert retries.allowed_methods is None or "POST" in retries.allowed_methods
        assert {429, 503} <= set(retries.status_forcelist)
