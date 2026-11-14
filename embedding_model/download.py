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
# report success. So a response whose Content-Type is HTML is not the file,
# unless the file is kept as an HTML page, under one of HTML_SUFFIXES.
HTML_SUFFIXES = (".html", ".htm")


def modified(path):
    """The day a kept file was written, as ``YYYY-MM-DD``."""
    return date.fromtimestamp(Path(path).stat().st_mtime).isoformat()


def keep(path, file_bytes):
    """Write a file whole, or not at all.

    The bytes are written under a temporary name and renamed when complete, so
    an interrupted write never leaves a truncated copy to be read later.

    Args:
        path (Path): Where the file is kept.
        file_bytes (bytes): Its contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(file_bytes)
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


def fetch(url, json=None, auth=None, html_ok=False):
    """Download a URL, retrying as RETRIES says.

    Args:
        url (str): The address.
        json (dict | None): A body to POST as JSON. Without one, the request
            is a GET.
        auth (tuple[str, str] | None): A user name and password, sent as
            HTTP Basic authentication.
        html_ok (bool): Whether an HTML page is what is asked for, as for a
            folder listing.

    Returns:
        bytes: The response body.

    Raises:
        requests.RequestException: If the server cannot be reached, or
            answers with an error status. A busy server's status, 429, 500,
            502, 503 or 504, raises only once the retries are spent.
        ValueError: If the response's Content-Type is HTML and ``html_ok``
            is false.
    """
    client = session()
    if json is None:
        response = client.get(url, auth=auth, timeout=TIMEOUT_SECONDS)
    else:
        response = client.post(url, json=json, auth=auth, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    if not html_ok and "text/html" in response.headers.get("Content-Type", ""):
        raise ValueError(
            f"{url} returned an HTML page, not the file. Nothing was kept; try"
            " again later."
        )
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
            kept under one of HTML_SUFFIXES, as in ``fetch``. Nothing is kept,
            so the next build tries again.
        Exception: Whatever ``check`` raises.
    """
    path = Path(path)
    if not path.exists():
        file_bytes = fetch(
            url, json=json, auth=auth, html_ok=path.suffix in HTML_SUFFIXES
        )
        if check is not None:
            check(file_bytes)
        keep(path, file_bytes)
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


def github_archive(repository, commit, folder):
    """A GitHub repository at one commit, as GitHub's archive, downloaded once.

    The archive is kept as ``<commit>.tar.gz`` in ``folder``. The commit pins
    the files in it. GitHub builds the archive on request, so its bytes are
    not pinned, and no checksum of them is checked.

    Args:
        repository (str): The repository's address, such as
            ``https://github.com/abcd/docs``.
        commit (str): The commit's full hash.
        folder (Path): Where the archive is kept.

    Returns:
        bytes: The archive, a ``.tar.gz`` with one top folder, named for the
            repository and the commit.
    """
    name = repository.removeprefix("https://github.com/")
    return cached(
        f"https://codeload.github.com/{name}/tar.gz/{commit}",
        Path(folder) / f"{commit}.tar.gz",
    )
