"""Downloads from public data providers, with a local copy kept."""

import hashlib
from pathlib import Path

import requests

# Zenodo requires the User-Agent to name the client making the request.
USER_AGENT = (
    "ai-harmonization/embedding_model "
    "(research; +https://github.com/uc-cdis/ai-harmonization)"
)
TIMEOUT_SECONDS = 180
# A server can answer with a page of its own, such as a bot check, and still
# report success. Apart from folder listings, no file kept here is an HTML
# page. So an HTML answer for any other file is not the file.
HTML_START = (b"<!doctype html", b"<html")


def sha256(data):
    """The SHA-256 of some bytes, as hex."""
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    """Download a URL.

    Args:
        url (str): The address to GET.

    Returns:
        bytes: The response body.

    Raises:
        requests.HTTPError: If the server answers with an error status.
    """
    response = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.content


def downloaded(url, path):
    """Download a URL once, keeping the file for later builds.

    The file is written under a temporary name and renamed when complete, so
    an interrupted download never leaves a truncated copy to be read later.

    Args:
        url (str): The address to GET when the file is not there yet.
        path (str | Path): Where the downloaded file is kept.

    Returns:
        Path: Where the file is kept.

    Raises:
        ValueError: If the server answers with an HTML page for a file that is
            not one. Nothing is kept, so the next build tries again.
    """
    path = Path(path)
    if not path.exists():
        data = fetch(url)
        if path.suffix != ".html" and data.lstrip()[:15].lower().startswith(HTML_START):
            raise ValueError(
                f"{url} answered with an HTML page, not the file. Nothing was "
                "kept; try again later."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(data)
        partial.replace(path)
    return path


def cached(url, path):
    """Download a URL once, as ``downloaded`` does, and return its contents.

    Args:
        url (str): The address to GET when the file is not there yet.
        path (str | Path): Where the downloaded file is kept.

    Returns:
        bytes: The file's contents.
    """
    return downloaded(url, path).read_bytes()
