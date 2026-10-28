"""LOINC: codes and their names, question texts, answer lists and replacements.

A release is pinned by its version, such as ``2.83``, and by the MD5 that the
manifest records for it. The release zip needs a free loinc.org account. With
``LOINC_USERNAME`` and ``LOINC_PASSWORD`` set, it is downloaded through LOINC's
API. Without them, it can be placed by hand in the downloads folder.

Only named columns are read. The SNOMED CT codes and names that the answer
file carries in columns of their own are never read: LOINC's license does not
cover them.
"""

import functools
import hashlib
import json
import os
import zipfile

import pandas as pd

from embedding_model.download import downloaded, fetch
from embedding_model.manifests import file_digest, file_sha256, pinned

RELEASE = "LOINC release"
METADATA = "https://loinc.regenstrief.org/api/v1/Loinc?version={version}"
DOWNLOAD = "https://loinc.regenstrief.org/api/v1/Loinc/Download?version={version}"
# The terms of use, as an SPDX license expression. The terms are in
# embedding_model/docs/licenses.md, which CITATION points to.
LICENSE = "LicenseRef-LOINC"
CITATION = "embedding_model/docs/licenses.md#loinc"
TABLES = {
    "codes": (
        "LoincTable/Loinc.csv",
        [
            "LOINC_NUM",
            "COMPONENT",
            "PROPERTY",
            "TIME_ASPCT",
            "SYSTEM",
            "SCALE_TYP",
            "METHOD_TYP",
            "CLASS",
            "CLASSTYPE",
            "STATUS",
            "LONG_COMMON_NAME",
            "SHORTNAME",
            "DisplayName",
            "CONSUMER_NAME",
            "RELATEDNAMES2",
            "DefinitionDescription",
            "SURVEY_QUEST_TEXT",
            "SURVEY_QUEST_SRC",
            "EXMPL_ANSWERS",
            "EXAMPLE_UNITS",
            "ORDER_OBS",
            "EXTERNAL_COPYRIGHT_NOTICE",
            "EXTERNAL_COPYRIGHT_LINK",
        ],
    ),
    "map_to": ("LoincTable/MapTo.csv", ["LOINC", "MAP_TO", "COMMENT"]),
    "forms": (
        "AccessoryFiles/PanelsAndForms/PanelsAndForms.csv",
        [
            "ID",
            "ParentId",
            "ParentLoinc",
            "SEQUENCE",
            "Loinc",
            "DisplayNameForForm",
            "EXTERNAL_COPYRIGHT_NOTICE",
            "AdditionalCopyright",
        ],
    ),
    "answers": (
        "AccessoryFiles/AnswerFile/AnswerList.csv",
        [
            "AnswerListId",
            "AnswerListName",
            "AnswerStringId",
            "LocalAnswerCode",
            "SequenceNumber",
            "DisplayText",
            "Description",
        ],
    ),
    "answer_links": (
        "AccessoryFiles/AnswerFile/LoincAnswerListLink.csv",
        ["LoincNumber", "AnswerListId", "AnswerListLinkType", "ApplicableContext"],
    ),
}


# Each table's restriction fields: its notices of third-party content.
RESTRICTION_FIELDS = {
    "codes": ["EXTERNAL_COPYRIGHT_NOTICE"],
    "forms": ["EXTERNAL_COPYRIGHT_NOTICE", "AdditionalCopyright"],
}


def read_release(archive):
    """The named columns of LOINC's tables, from a release zip.

    A code, and a form's row, are open when their RESTRICTION_FIELDS are
    empty. A form's row is also closed when its code carries a notice, or when
    a row that it is nested in is closed: forms nest panels several levels
    deep. With closed rows left out, a code is open when it sits in an open
    row, or in no form and has no notice of its own.

    Args:
        archive (str | Path | file): The release zip, or its path.

    Returns:
        dict[str, pd.DataFrame]: ``codes``, ``map_to``, ``forms``, ``answers``
            and ``answer_links``. Every cell is a string. ``codes`` and
            ``forms`` have an ``open`` column.

    Raises:
        KeyError: If the zip lacks one of the tables.
        ValueError: If a table lacks one of the columns.
    """
    tables = {}
    with zipfile.ZipFile(archive) as bundle:
        for table, (end, columns) in TABLES.items():
            member = next((n for n in bundle.namelist() if n.endswith(end)), None)
            if member is None:
                raise KeyError(f"the LOINC zip has no {end}")
            with bundle.open(member) as handle:
                frame = pd.read_csv(
                    handle,
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8-sig",
                    usecols=columns,
                )
            tables[table] = frame[columns]
    for table, fields in RESTRICTION_FIELDS.items():
        notices = tables[table][fields].apply(lambda column: column.str.strip())
        tables[table]["open"] = (notices == "").all(axis=1)
    codes, forms = tables["codes"], tables["forms"]
    parent = dict(zip(forms.ID, forms.ParentId))
    own = dict(
        zip(forms.ID, ~forms.open | forms.Loinc.isin(codes.LOINC_NUM[~codes.open]))
    )

    @functools.cache
    def closed(row):
        """Whether a row, or a row that it is nested in, is closed."""
        return own[row] or (parent[row] != row and closed(parent[row]))

    forms["open"] = ~forms.ID.map(closed)
    only_closed = set(forms.Loinc) - set(forms.Loinc[forms.open])
    codes["open"] &= ~codes.LOINC_NUM.isin(only_closed)
    return tables


def credentials():
    """LOINC's user name and password from the environment, or None."""
    user, password = os.environ.get("LOINC_USERNAME"), os.environ.get("LOINC_PASSWORD")
    return (user, password) if user and password else None


def md5_check(expected):
    """A check, for ``downloaded``, that a zip has every MD5 it should.

    Args:
        expected (dict[str, str | None]): Each MD5 that the zip must have,
            keyed by what it belongs to, such as ``LOINC 2.83's``. A missing
            MD5 is not checked.

    Returns:
        Callable[[bytes], None]: The check.
    """

    def check(data):
        """Raise unless the data has each expected MD5.

        LOINC publishes MD5 checksums, so the check uses MD5. It finds a
        corrupt or wrong download, and is not used for security.
        """
        found = hashlib.md5(data, usedforsecurity=False).hexdigest()
        for whose, md5 in expected.items():
            if md5 and found != md5:
                raise ValueError(
                    f"the download has MD5 {found}, and {whose} is {md5}; "
                    "nothing was kept"
                )

    return check


def fetch_source(downloads, manifest):
    """Fetch the LOINC release that the manifest pins.

    The zip is downloaded through LOINC's API when credentials are set. It
    is kept only if it has the MD5 that the API publishes for the release, and
    the MD5 that the manifest pins, once it pins one. A zip placed by hand must
    have the pinned MD5 too.

    Args:
        downloads (Path): Where downloaded files are kept between fetches.
        manifest (dict): The committed manifest; its release entry gives the
            version, and the MD5 once recorded.

    Returns:
        tuple[dict[str, pd.DataFrame], list[dict], dict]: The tables, the
            manifest's sources, and the counts that the fetch prints.

    Raises:
        FileNotFoundError: If the zip is neither in the downloads folder nor
            downloadable without credentials.
        ValueError: If the zip's MD5 differs from the release's MD5.
    """
    pin = pinned(manifest, RELEASE)
    version = pin["version"]
    path = downloads / f"Loinc_{version}.zip"
    if not path.exists():
        auth = credentials()
        if auth is None:
            raise FileNotFoundError(
                f"{path} is missing. Set LOINC_USERNAME and LOINC_PASSWORD to "
                f"download LOINC {version} with a loinc.org account, or download "
                f"Loinc_{version}.zip from https://loinc.org/downloads/ into "
                f"{downloads}."
            )
        published = json.loads(fetch(METADATA.format(version=version), auth=auth))
        published = published[0] if isinstance(published, list) else published
        check = md5_check(
            {
                f"LOINC {version}'s": published["downloadMD5Hash"],
                "the manifest's": pin.get("md5"),
            }
        )
        downloaded(DOWNLOAD.format(version=version), path, auth=auth, check=check)
    md5 = file_digest(path, "md5")
    if pin.get("md5") and md5 != pin["md5"]:
        raise ValueError(
            f"{path} has MD5 {md5}, but the manifest pins {pin['md5']}. Delete"
            " it to download the pinned release again. To use this release,"
            " follow 'Newer releases' in embedding_model/docs/sources.md#loinc"
        )
    tables = read_release(path)
    codes = tables["codes"]
    counts = {"codes": len(codes), "not_open": int((~codes.open).sum())}
    print(f"  {counts['codes']:,} codes, {counts['not_open']:,} not open")
    sources = [
        {
            "role": "release",
            "name": RELEASE,
            "location": DOWNLOAD.format(version=version),
            "version": version,
            "md5": md5,
            "sha256": file_sha256(path),
            "license": LICENSE,
            "citation": CITATION,
        }
    ]
    return tables, sources, counts
