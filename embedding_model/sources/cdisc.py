"""CDISC controlled terminology: codelists and their terms, from NCI EVS.

EVS publishes each CDISC terminology package as a tab-separated file, with a
new release each quarter. Its archive keeps every release under its date.
Each package is read from the archive, at the release date that the manifest
pins. So a new CDISC release does not change what the fetch reads, until the
manifest pins the new release.
"""

import csv
import io
from urllib.parse import quote

import pandas as pd

from embedding_model.download import cached
from embedding_model.manifests import pinned, sha256

ARCHIVE = "https://evs.nci.nih.gov/ftp1/CDISC/{folder}/Archive/{file}"
# Each package's folder and its file's name, before the release date.
PACKAGES = {
    "sdtm": ("SDTM", "SDTM Terminology"),
    "adam": ("ADaM", "ADaM Terminology"),
    "protocol": ("Protocol", "Protocol Terminology"),
    "glossary": ("Glossary", "CDISC Glossary"),
}
COLUMNS = [
    "Code",
    "Codelist Code",
    "Codelist Extensible (Yes/No)",
    "Codelist Name",
    "CDISC Submission Value",
    "CDISC Synonym(s)",
    "CDISC Definition",
    "NCI Preferred Term",
]
# The terms of use, as an SPDX license expression. The terms are in
# embedding_model/docs/licenses.md, which CITATION points to.
LICENSE = "LicenseRef-NCI-CDISC"
CITATION = "embedding_model/docs/licenses.md#cdisc"


def release_name(package):
    """The manifest's name for a package's release, such as ``CDISC SDTM release``."""
    return f"CDISC {PACKAGES[package][0]} release"


def read_package(data, package):
    """One package's rows, with every cell a string.

    ``keep_default_na=False`` keeps submission values such as ``NA`` and
    ``None`` as the text they are, not as missing. The files quote nothing,
    so a double quote is read as text.

    Args:
        data (bytes): The package's tab-separated file.
        package (str): The package's key in PACKAGES.

    Returns:
        pd.DataFrame: The COLUMNS, a ``package`` column, and ``open``.

    Raises:
        ValueError: If the file lacks one of COLUMNS.
    """
    frame = pd.read_csv(
        io.BytesIO(data),
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        usecols=COLUMNS,
    )
    return frame[COLUMNS].assign(package=package, open=True)


def fetch_source(downloads, manifest):
    """Fetch every package at the release date that the manifest pins.

    Args:
        downloads (Path): Where downloaded files are kept between fetches.
        manifest (dict): The committed manifest; each package's release entry
            gives its date.

    Returns:
        tuple[dict[str, pd.DataFrame], list[dict], dict]: The table ``terms``,
            every package's rows together, the manifest's sources, and the
            counts that the fetch prints.
    """
    frames, sources, rows = [], [], {}
    for package, (folder, stem) in PACKAGES.items():
        released = pinned(manifest, release_name(package))["version"]
        name = f"{stem} {released}.txt"
        url = ARCHIVE.format(folder=folder, file=quote(name))
        data = cached(url, downloads / name)
        frame = read_package(data, package)
        rows[package] = len(frame)
        print(f"  {package:<9}{rows[package]:>8,} rows")
        frames.append(frame)
        sources.append(
            {
                "role": "release",
                "name": release_name(package),
                "location": url,
                "version": released,
                "sha256": sha256(data),
                "license": LICENSE,
                "citation": CITATION,
            }
        )
    terms = pd.concat(frames, ignore_index=True).sort_values(
        ["package", "Codelist Code", "Code"], ignore_index=True
    )
    return {"terms": terms}, sources, {"rows_by_package": rows}
