"""PhenX: protocols' variables and their mappings to dbGaP variables.

PhenX's database export holds its protocols, their variables and their
availability, and maps each variable to dbGaP variables of real studies. Its
address serves only the latest export, so the zip is kept after the first
fetch. Its release, such as ``Ver 49.0``, must be the one the manifest pins.
"""

import io
import re
import zipfile

import pandas as pd

from embedding_model.download import downloaded, file_sha256, modified
from embedding_model.sources.common import open_where, pinned

EXPORT = "PhenX database export"
EXPORT_URL = "https://www.phenxtoolkit.org/toolkit_content/documents/data_dictionary/Phenx_DB.zip"
DUMP = "phenx.sql"
LICENSE = (
    "most protocols are freely available, and some have limited availability, "
    "as each protocol's availability says (https://www.phenxtoolkit.org/help/faq)"
)
CITATION = "embedding_model/docs/sources.md#phenx"
TABLES = {
    "variables",
    "protocols",
    "protocols_protocol_availability_types",
    "protocol_availability_types",
    "variable_gap_mappings",
    "protocols_old_protocols_new",
    "releases",
}
OPEN_AVAILABILITY = {"Available"}
CLOSED_AVAILABILITY = {"Limited Availability", ""}
CREATE = re.compile(r"CREATE TABLE `([^`]+)`")
COLUMN = re.compile(r"\s+`([^`]+)`\s")
INSERT = re.compile(r"INSERT INTO `([^`]+)` VALUES ")
ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "Z": "\x1a"}
# A token of an INSERT's values: a quoted string, with MySQL's backslash
# escapes and doubled quotes; a bracket or comma; or a bare value.
TOKEN = re.compile(r"'(?:[^'\\]|\\.|'')*'|[(),]|[^'(),]+", re.S)
ESCAPE = re.compile(r"\\(.)|''", re.S)


def unquoted(text):
    """A quoted MySQL string's value, with its escapes read."""
    return ESCAPE.sub(
        lambda found: ESCAPES.get(found[1], found[1]) if found[1] is not None else "'",
        text[1:-1],
    )


def insert_rows(statement):
    """The rows of one ``INSERT ... VALUES (...),(...);`` statement.

    A value is a quoted string, with MySQL's backslash escapes, ``NULL``, or
    a bare number.

    Args:
        statement (str): The statement, from ``VALUES`` on.

    Returns:
        list[list[str | None]]: The rows, ``NULL`` as None.
    """
    rows, row = [], None
    for token in TOKEN.findall(statement):
        if token == "(":
            row = []
        elif token == ")":
            rows.append(row)
            row = None
        elif row is None or token == "," or not token.strip():
            continue
        elif token.startswith("'"):
            row.append(unquoted(token))
        else:
            row.append(None if token.strip() == "NULL" else token.strip())
    return rows


def read_dump(stream, tables):
    """Chosen tables of a MySQL dump, by name.

    Args:
        stream (file): The dump, opened in binary mode.
        tables (set[str]): The tables to read.

    Returns:
        dict[str, pd.DataFrame]: Each table, with its columns' names from its
            ``CREATE TABLE``. Every cell is a string, ``NULL`` as ``""``.

    Raises:
        ValueError: If a row has another number of values than its table has
            columns: the dump was not read as written.
    """
    columns, current, rows = {}, None, {name: [] for name in tables}
    for line in io.TextIOWrapper(stream, encoding="utf-8"):
        created = CREATE.match(line)
        if created:
            current = created.group(1)
            columns[current] = []
            continue
        if current is not None:
            column = COLUMN.match(line)
            if column:
                columns[current].append(column.group(1))
            elif line.startswith(")"):
                current = None
            continue
        inserted = INSERT.match(line)
        if inserted and inserted.group(1) in tables:
            rows[inserted.group(1)] += insert_rows(line[inserted.end() :])
    for name, found in rows.items():
        expected = len(columns.get(name, []))
        wrong = sum(len(row) != expected for row in found)
        if wrong:
            raise ValueError(
                f"{wrong:,} rows of {name} do not have its {expected} columns"
            )
    return {
        name: pd.DataFrame(found, columns=columns.get(name, [])).fillna("").astype(str)
        for name, found in rows.items()
    }


def by_id(frame):
    """A table's rows in the order of their numeric ``id``."""
    order = frame.id.astype(int).sort_values(kind="stable").index
    return frame.loc[order].reset_index(drop=True)


def read_export(dump):
    """PhenX's variables, protocols, dbGaP mappings and release.

    A protocol is open when its availability is "Available", and a variable
    takes its protocol's ``open``, ``visible`` flag and availability. A
    variable whose protocol is not in ``protocols`` has none of them, and is
    not open.

    Args:
        dump (dict[str, pd.DataFrame]): From ``read_dump``, with TABLES.

    Returns:
        dict[str, pd.DataFrame]: ``variables``, ``protocols``, ``mappings``,
            ``replacements`` (each replaced protocol and the one that
            replaced it) and ``release``: the latest release, by date and
            then by id, one row.
    """
    names = dump["protocol_availability_types"].set_index("id").name
    links = dump["protocols_protocol_availability_types"]
    availability = (
        links.assign(
            availability=links.protocol_availability_type_id.map(names).fillna("")
        )
        .groupby("protocol_id")
        .availability.agg(lambda kinds: "|".join(sorted(set(kinds))))
    )
    protocols = dump["protocols"][
        [
            "id",
            "code",
            "name",
            "description",
            "status",
            "visible",
            "version",
            "release_date",
        ]
    ].copy()
    protocols["availability"] = protocols.id.map(availability).fillna("")
    protocols["open"] = open_where(
        protocols.availability, OPEN_AVAILABILITY, CLOSED_AVAILABILITY, "availability"
    )
    variables = dump["variables"][
        [
            "id",
            "name",
            "code",
            "description",
            "protocol_id",
            "type",
            "units",
            "min",
            "max",
            "variable_values",
            "comment1",
            "comment2",
            "version_num",
        ]
    ].merge(
        protocols[["id", "visible", "availability", "open"]].rename(
            columns={"id": "protocol_id", "visible": "protocol_visible"}
        ),
        on="protocol_id",
        how="left",
    )
    variables["open"] = variables.open.fillna(False).astype(bool)
    variables = variables.fillna("")
    mappings = dump["variable_gap_mappings"][
        [
            "id",
            "variable_id",
            "variable_name",
            "gap_study_accession",
            "gap_study_name",
            "gap_phv",
            "gap_variable_name",
            "gap_variable_desc",
            "map_level",
        ]
    ]
    releases = dump["releases"].assign(order=dump["releases"].id.astype(int))
    release = releases.sort_values(["release_date", "order"]).tail(1)[
        ["name", "release_date"]
    ]
    replacements = dump["protocols_old_protocols_new"][
        ["id", "protocol_old_id", "protocol_new_id"]
    ]
    return {
        "variables": by_id(variables),
        "protocols": by_id(protocols),
        "mappings": by_id(mappings),
        "replacements": by_id(replacements),
        "release": release.reset_index(drop=True),
    }


def check_export(data):
    """Raise unless the data is a zip holding PhenX's dump."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            names = bundle.namelist()
    except zipfile.BadZipFile as error:
        raise ValueError("the PhenX export is not a zip; nothing was kept") from error
    if DUMP not in names:
        raise ValueError(f"the PhenX export holds no {DUMP}; nothing was kept")


def fetch_source(downloads, manifest):
    """Fetch PhenX's latest export, or read the one kept.

    Args:
        downloads (Path): Where downloaded files are kept between fetches.
        manifest (dict): The committed manifest; its export entry gives the
            release.

    Returns:
        tuple[dict[str, pd.DataFrame], list[dict]]: The tables and the
            manifest's sources.

    Raises:
        ValueError: If the export's release is not the one pinned. The
            address serves only the latest export, so a new release needs a
            new version of the manifest.
    """
    pin = pinned(manifest, EXPORT)
    path = downloaded(EXPORT_URL, downloads / "Phenx_DB.zip", check=check_export)
    with zipfile.ZipFile(path) as bundle, bundle.open(DUMP) as stream:
        tables = read_export(read_dump(stream, TABLES))
    release = tables["release"].name.iloc[0]
    if pin.get("version") and release != pin["version"]:
        raise ValueError(
            f"{path.name} is PhenX {release}, and the manifest pins {pin['version']}"
        )
    found = tables["variables"]
    print(f"  {release}: {len(found):,} variables, {found.open.sum():,} open")
    return tables, [
        {
            "role": "service",
            "name": EXPORT,
            "location": EXPORT_URL,
            "version": release,
            "fetched": modified(path),
            "sha256": file_sha256(path),
            "license": LICENSE,
            "citation": CITATION,
        }
    ]
