"""Downloads from public data providers, with a local copy kept."""

import functools
from datetime import date
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

# Zenodo requires the User-Agent to name the client making the request.
USER_AGENT = (
    "ai-harmonization/embedding_model "
    "(research; +https://github.com/uc-cdis/ai-harmonization)"
)
TIMEOUT_SECONDS = 180
# A request is tried again when the server is busy, or the connection fails
# before an answer starts. It is tried up to 5 more times, at once and then
# after 2, 4, 8 and 16 seconds, or as long as the server's Retry-After asks.
# A connection lost while the answer arrives is not tried again: nothing is
# kept, so the next fetch asks again. A POST is retried too, since the services
# that receive these POSTs only read data.
RETRIES = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=None,
)
# A server can answer with a page of its own, such as a bot check, and still
# report success. So an HTML answer is not the file, unless the file is kept
# as an HTML page, under one of HTML_SUFFIXES.
HTML_START = (b"<!doctype html", b"<html")
HTML_SUFFIXES = (".html", ".htm")


def modified(path):
    """The day a kept file was written, as ``YYYY-MM-DD``."""
    return date.fromtimestamp(Path(path).stat().st_mtime).isoformat()


def keep(path, data):
    """Write a file whole, or not at all.

    The data is written under a temporary name and renamed when complete, so
    an interrupted write never leaves a truncated copy to be read later.

    Args:
        path (Path): Where the file is kept.
        data (bytes): Its contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    partial.replace(path)


@functools.cache
def session():
    """The requests session that every download shares, retrying as RETRIES says.

    One session is kept, so many requests to one server reuse a connection.
    """
    adapter = HTTPAdapter(max_retries=RETRIES)
    client = requests.Session()
    client.mount("https://", adapter)
    client.mount("http://", adapter)
    client.headers["User-Agent"] = USER_AGENT
    return client


def fetch(url, json=None, auth=None):
    """Download a URL, retrying as RETRIES says.

    Args:
        url (str): The address.
        json (dict | None): A body to POST as JSON. Without one, the request
            is a GET.
        auth (tuple[str, str] | None): A user name and password, sent as
            HTTP Basic authentication.

    Returns:
        bytes: The response body.

    Raises:
        requests.RequestException: If the server cannot be reached, or
            answers with an error status. A busy server's status, 429, 500,
            502, 503 or 504, raises only once the retries are spent.
    """
    client = session()
    if json is None:
        response = client.get(url, auth=auth, timeout=TIMEOUT_SECONDS)
    else:
        response = client.post(url, json=json, auth=auth, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def downloaded(url, path, json=None, auth=None, check=None):
    """Download a URL once, keeping the file for later builds.

    The file is written with ``keep``, so an interrupted download never
    leaves a truncated copy to be read later.

    Args:
        url (str): The address to fetch when the file is not there yet.
        path (str | Path): Where the downloaded file is kept.
        json (dict | None): A body to POST as JSON, as in ``fetch``.
        auth (tuple[str, str] | None): HTTP Basic credentials, as in ``fetch``.
        check (Callable[[bytes], None] | None): Called with a new download
            before it is kept. It raises if the data is not the file, and
            then nothing is kept.

    Returns:
        Path: Where the file is kept.

    Raises:
        requests.RequestException: As in ``fetch``.
        ValueError: If the server answers with an HTML page for a file not
            kept under one of HTML_SUFFIXES. Nothing is kept, so the next
            build tries again.
        Exception: Whatever ``check`` raises.
    """
    path = Path(path)
    if not path.exists():
        data = fetch(url, json=json, auth=auth)
        start = data.lstrip()[:15].lower()
        if path.suffix not in HTML_SUFFIXES and start.startswith(HTML_START):
            raise ValueError(
                f"{url} answered with an HTML page, not the file. Nothing was "
                "kept; try again later."
            )
        if check is not None:
            check(data)
        keep(path, data)
    return path


def cached(url, path, json=None, auth=None):
    """Download a URL once, as ``downloaded`` does, and return its contents.

    Args:
        url (str): The address to fetch when the file is not there yet.
        path (str | Path): Where the downloaded file is kept.
        json (dict | None): A body to POST as JSON, as in ``fetch``.
        auth (tuple[str, str] | None): HTTP Basic credentials, as in ``fetch``.

    Returns:
        bytes: The file's contents.
    """
    return downloaded(url, path, json=json, auth=auth).read_bytes()
