"""NHANES: survey variables, their codebook texts and their value labels.

NHANES publishes no machine-readable variable list. Its variable-list pages,
one per component, list every variable with its data file. Each data file has
a codebook page. For each variable it gives the fields in CODEBOOK_FIELDS,
such as its question ("English Text") and target group, and a table of its
codes and their labels.

NHANES serves only the current state of its pages, so both kinds are kept
after the first fetch, and identified by their SHA-256. The data files whose
codebook page is not found are kept in a list, so a later fetch needs no
network either.
"""

import hashlib
import re

import pandas as pd
from bs4 import BeautifulSoup

from embedding_model.download import downloaded, keep, modified
from embedding_model.sources.common import open_where
from embedding_model.manifests import sha256

VARIABLE_LIST = (
    "https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component={component}"
)
CODEBOOK = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/DataFiles/{data_file}.htm"
)
# A data file's name, such as "DEMO_J". It goes into a file path and an address,
# so a variable list that names a data file with other characters is refused.
DATA_FILE = re.compile(r"[A-Za-z0-9_]+")
COMPONENTS = ["Demographics", "Dietary", "Examination", "Laboratory", "Questionnaire"]
LIST_COLUMNS = {
    "Variable Name": "name",
    "Variable Description": "description",
    "Data File Name": "data_file",
    "Data File Description": "data_file_description",
    "Begin Year": "begin_year",
    "EndYear": "end_year",
    "Component": "component",
    "Use Constraints": "use_constraints",
}
# Use Constraints restricts access to the data files, not the public-domain
# documentation used here, so both values are open. No value is known to be
# closed, so a new value is reported, and not open until decided.
OPEN_USE_CONSTRAINTS = {"None", "RDC Only"}
CLOSED_USE_CONSTRAINTS = set()
# Text that every page of each kind holds. A page without it is not that page,
# such as a server's own error or bot-check page, though served with success.
LIST_MARKER = b"Use Constraints"
CODEBOOK_MARKER = b'class="vartitle"'
NOT_FOUND = "not_found.txt"
CODEBOOK_FIELDS = {
    "SAS Label:": "sas_label",
    "English Text:": "english_text",
    "English Instructions:": "english_instructions",
    "Target:": "target",
    "Hard Edits:": "hard_edits",
}
# The terms of use, as an SPDX license expression. The terms are in
# embedding_model/docs/licenses.md, which CITATION points to.
LICENSE = "LicenseRef-CDC-Agency-Materials"
CITATION = "embedding_model/docs/licenses.md#nhanes"


def cycle(begin_year):
    """The survey cycle that a data file belongs to, as its codebook's address has it.

    Cycles are two years long and start on an odd year, so a file that begins
    in 2000 is in the 1999 cycle.

    Args:
        begin_year (str): The variable list's ``Begin Year``.

    Returns:
        int: The cycle's first year.
    """
    year = int(begin_year)
    return year if year % 2 else year - 1


def clean(text):
    """A page's text with its runs of whitespace made single spaces."""
    return " ".join(text.split())


def read_variable_list(page):
    """The variables on one variable-list page.

    Args:
        page (bytes): The page.

    Returns:
        pd.DataFrame: One row per variable, with LIST_COLUMNS' names.

    Raises:
        ValueError: If the page's table lacks one of LIST_COLUMNS.
    """
    tables = BeautifulSoup(page, "html.parser").find_all("table")
    table = max(tables, key=lambda t: len(t.find_all("tr")))
    rows = table.find_all("tr")
    header = [clean(cell.get_text()) for cell in rows[0].find_all(["th", "td"])]
    found = []
    for row in rows[1:]:
        cells = [clean(cell.get_text()) for cell in row.find_all("td")]
        if len(cells) == len(header):
            found.append(dict(zip(header, cells)))
    missing = [column for column in LIST_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"the variable list has no column {', '.join(missing)}")
    frame = pd.DataFrame(found, columns=header)[list(LIST_COLUMNS)]
    return frame.rename(columns=LIST_COLUMNS)


def read_codebook(page):
    """The variables of one codebook page, and each one's value table.

    Every row of a value table is kept: a code and its label, a "Range of
    Values" row and a "Missing" row alike. A field given more than once, such
    as a variable's targets, keeps every value, joined by ``|``.

    Args:
        page (bytes): The codebook page.

    Returns:
        tuple[list[dict], list[dict]]: The variables (name and
            CODEBOOK_FIELDS) and their values (name, position, code, label).
    """
    variables, values = [], []
    for block in BeautifulSoup(page, "html.parser").select("div.pagebreak"):
        title = block.find("h3", class_="vartitle")
        if title is None or not title.get("id"):
            continue
        name = title["id"].strip()
        fields = {key: [] for key in CODEBOOK_FIELDS.values()}
        for term in block.find_all("dt"):
            label = clean(term.get_text())
            if label in CODEBOOK_FIELDS and term.find_next_sibling("dd") is not None:
                fields[CODEBOOK_FIELDS[label]].append(
                    clean(term.find_next_sibling("dd").get_text())
                )
        variables.append(
            {"name": name, **{key: "|".join(found) for key, found in fields.items()}}
        )
        table = block.find("table", class_="values")
        for position, row in enumerate(table.select("tbody tr") if table else []):
            cells = [clean(cell.get_text()) for cell in row.find_all("td")]
            if len(cells) >= 2:
                values.append(
                    {
                        "name": name,
                        "position": position,
                        "code": cells[0],
                        "label": cells[1],
                    }
                )
    return variables, values


def page(url, path, marker):
    """Download a page once, and keep it only if it holds its marker.

    Args:
        url (str): The page's address.
        path (Path): Where the page is kept.
        marker (bytes): Text that every page of its kind holds.

    Returns:
        bytes: The page.

    Raises:
        ValueError: If the page lacks its marker. It is not kept, so the next
            fetch downloads it again.
    """

    def check(data):
        """Raise unless the page holds its marker."""
        if marker not in data:
            raise ValueError(f"{url} answered with another page; nothing was kept")

    return downloaded(url, path, check=check).read_bytes()


def not_found(error):
    """Whether a download failed because the server has no such page (404)."""
    response = getattr(error, "response", None)
    return response is not None and response.status_code == 404


def add_open(codebook, variables):
    """Add an ``open`` column to a codebook: true when the row may be used.

    A row is open when its data file's Use Constraints allow use.

    Args:
        codebook (pd.DataFrame): The codebook rows, with ``cycle`` and
            ``data_file``.
        variables (pd.DataFrame): The variables, with their ``open``.

    Returns:
        pd.DataFrame: The codebook, with an ``open`` column.
    """
    files = variables.groupby(["cycle", "data_file"]).open.all().rename("open")
    found = codebook.join(files, on=["cycle", "data_file"])
    return found.assign(open=found.open.fillna(False).astype(bool))


def keep_not_found(path, data_files):
    """Keep the list of data files without a public codebook page.

    Args:
        path (Path): The list, ``codebooks/not_found.txt``.
        data_files (Iterable[str]): The data files, as ``<cycle>/<data file>``.
    """
    keep(path, "".join(f"{data_file}\n" for data_file in sorted(data_files)).encode())


def fetch_source(downloads, manifest):
    """Fetch the variable lists and every data file's codebook.

    Some data files have no public codebook page, such as those restricted
    to the Research Data Center. They are listed in
    ``codebooks/not_found.txt``, and not asked for again. Any other failure
    to download a page stops the fetch.

    Args:
        downloads (Path): Where downloaded pages are kept between fetches.
        manifest (dict): The committed manifest. Nothing is pinned ahead:
            NHANES serves only its current pages.

    Returns:
        tuple[dict[str, pd.DataFrame], list[dict], dict]: The tables ``variables``,
            ``codebook`` and ``values``, which share the key ``cycle``,
            ``data_file`` and ``name``; the manifest's sources; and the counts
            that the fetch prints.

    Raises:
        ValueError: If a variable list is not the page it should be, or names a
            data file with characters other than letters, digits and
            underscores.
        RuntimeError: If a codebook page could not be downloaded for another
            reason than not being found. The pages kept are not downloaded
            again, so a new fetch tries only the rest.
    """
    sources, lists = [], []
    for component in COMPONENTS:
        url = VARIABLE_LIST.format(component=component)
        path = downloads / "lists" / f"{component}.html"
        listing = page(url, path, LIST_MARKER)
        lists.append(read_variable_list(listing))
        sources.append(
            {
                "role": "service",
                "name": f"NHANES variable list, {component}",
                "location": url,
                "fetched": modified(path),
                "sha256": sha256(listing),
                "license": LICENSE,
                "citation": CITATION,
            }
        )
    variables = pd.concat(lists, ignore_index=True)
    variables.insert(
        0,
        "cycle",
        variables.begin_year.map(
            lambda year: str(cycle(year)) if year.isdigit() else ""
        ),
    )
    variables["open"] = open_where(
        variables.use_constraints,
        OPEN_USE_CONSTRAINTS,
        CLOSED_USE_CONSTRAINTS,
        "Use Constraints",
    )

    files = sorted(
        {
            (int(row.cycle), row.data_file)
            for row in variables.itertuples()
            if row.data_file and row.cycle
        }
    )
    odd = sorted({name for _, name in files if not DATA_FILE.fullmatch(name)})
    if odd:
        raise ValueError(f"the variable lists name odd data files: {odd[:5]}")
    listed = downloads / "codebooks" / NOT_FOUND
    known = set(listed.read_text().split()) if listed.exists() else set()
    codebook, values, missing = [], [], []
    digest, read = hashlib.sha256(), []
    for index, (first_year, data_file) in enumerate(files, start=1):
        entry = f"{first_year}/{data_file}"
        if entry in known:
            missing.append(entry)
            continue
        url = CODEBOOK.format(cycle=first_year, data_file=data_file)
        path = downloads / "codebooks" / str(first_year) / f"{data_file}.htm"
        try:
            text = page(url, path, CODEBOOK_MARKER)
        except Exception as error:
            if not_found(error):
                missing.append(entry)
                continue
            # The pages not found so far are listed, with those known before,
            # so the next fetch does not ask for them again.
            keep_not_found(listed, known | set(missing))
            raise RuntimeError(
                f"the codebook page {url} could not be downloaded ({error})."
                " Fetch again: the pages kept are not asked for again."
            ) from error
        read.append(path)
        digest.update(f"{entry}:".encode() + hashlib.sha256(text).digest())
        found, labels = read_codebook(text)
        key = {"cycle": str(first_year), "data_file": data_file}
        codebook += [{**key, **row} for row in found]
        values += [{**key, **row} for row in labels]
        if index % 200 == 0:
            print(f"  {index:,} of {len(files):,} codebooks")
    keep_not_found(listed, missing)
    counts = {
        "variables": len(variables),
        "data_files": len(files),
        "data_files_without_codebook": len(missing),
    }
    print(
        f"  {counts['variables']:,} variables in {counts['data_files']:,} data "
        f"files; {counts['data_files_without_codebook']:,} data files without a "
        "public codebook"
    )
    sources.append(
        {
            "role": "service",
            "name": "NHANES codebooks",
            "location": CODEBOOK,
            "fetched": modified(
                max(read, key=lambda kept: kept.stat().st_mtime, default=listed)
            ),
            "sha256": digest.hexdigest(),
            "license": LICENSE,
            "citation": CITATION,
        }
    )
    tables = {
        "variables": variables.sort_values(
            ["name", "data_file", "begin_year"], ignore_index=True
        ),
        "codebook": add_open(pd.DataFrame(codebook), variables).sort_values(
            ["cycle", "data_file", "name"], ignore_index=True
        ),
        "values": pd.DataFrame(values).sort_values(
            ["cycle", "data_file", "name", "position"], ignore_index=True
        ),
    }
    return tables, sources, counts
