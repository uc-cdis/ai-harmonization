"""
Gen3 / BDC utilities for fetching preharmonized study data.

Covers two workflows:
  1. PFB workflow  — discover studies, download PFB (.avro), convert to TSVs
  2. Metadata workflow — download dbGaP data_dict.xml / var_report.xml files
"""

import csv
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

import requests


# ── Constants ─────────────────────────────────────────────────────────────────

METADATA_XML_KEYWORDS = ("data_dict", "var_report")
ARCHIVE_KEYWORDS = ("data_dictionary", "variable_report", "data_dict", "var_report")
ARCHIVE_SUFFIXES = (".tar.gz", ".zip", ".tar")


# ── Delimited file reading ─────────────────────────────────────────────────────


def read_sv_as_list(filename, delimiter=",", encoding="utf-8"):
    """Read a separated-values file and return all rows as a list of dictionaries.

    "SV" is delimiter-agnostic: pass ``delimiter=','`` for CSV or ``'\\t'`` for TSV.

    Args:
        filename (str): Path to the file to read.
        delimiter (str): Field separator. Defaults to ``','``.
        encoding (str): File encoding. Defaults to ``'utf-8'``.

    Returns:
        list[dict]: One dict per data row, keyed by the header row.
    """
    with open(filename, encoding=encoding) as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        return list(reader)


# ── Filename helpers ───────────────────────────────────────────────────────────


def is_direct_metadata(filename):
    """Return True if filename is a dbGaP data_dict or var_report XML."""
    lower = filename.lower()
    return lower.endswith(".xml") and any(kw in lower for kw in METADATA_XML_KEYWORDS)


def is_metadata_archive(filename):
    """Return True if filename is a data_dictionary or variable_report archive."""
    lower = filename.lower()
    return any(kw in lower for kw in ARCHIVE_KEYWORDS) and any(
        lower.endswith(s) for s in ARCHIVE_SUFFIXES
    )


# ── Archive extraction ─────────────────────────────────────────────────────────


def extract_metadata_from_archive(archive_path, dest_dir):
    """Extract data_dict / var_report XMLs from a zip or tar archive.

    Args:
        archive_path (str): Path to the archive file (.zip, .tar, .tar.gz, etc.).
        dest_dir (str): Directory where extracted XML files are written.

    Returns:
        list[str]: Filenames of successfully extracted XML files.
    """
    extracted = []
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.namelist():
                    if is_direct_metadata(os.path.basename(member)):
                        filename = os.path.basename(member)
                        with (
                            zf.open(member) as src,
                            open(os.path.join(dest_dir, filename), "wb") as dst,
                        ):
                            dst.write(src.read())
                        extracted.append(filename)
        else:
            with tarfile.open(archive_path, "r:*") as tf:
                for member in tf.getmembers():
                    if is_direct_metadata(os.path.basename(member.name)):
                        filename = os.path.basename(member.name)
                        f = tf.extractfile(member)
                        if f:
                            with open(os.path.join(dest_dir, filename), "wb") as dst:
                                dst.write(f.read())
                            extracted.append(filename)
    except Exception as e:
        print(
            f"  [! Error] Failed to extract archive {os.path.basename(archive_path)}: {e}"
        )
    return extracted


# ── Study discovery ────────────────────────────────────────────────────────────


def get_active_preharmonized_studies(mds, limit=2000):
    """Query Gen3 Metadata Service and return active studies with preharmonized data.

    Filters out:
      - Studies with doi_tombstone=True (archived/retired)
      - Studies without a "Preharmonized" object in gen3_discovery.objects

    Args:
        mds: Gen3Metadata client instance
        limit (int): Maximum number of studies to retrieve from MDS

    Returns:
        dict: {guid: metadata} for qualifying studies
    """
    all_studies = mds.query(
        query="",
        return_full_metadata=True,
        _guid_type="discovery_metadata",
        limit=limit,
    )
    return {
        guid: data
        for guid, data in all_studies.items()
        if data.get("gen3_discovery", {}).get("doi_tombstone") in ["", None]
        and any(
            isinstance(obj, dict)
            and (
                "Preharmonized" in obj.get("description", "")
                or "Preharmonized" in obj.get("display_name", "")
            )
            for obj in data.get("gen3_discovery", {}).get("objects", [])
        )
    }


def select_studies(studies, mode, selected_ids=None, max_count=None):
    """Return a list of (study_id, metadata) pairs based on selection mode.

    Args:
        studies (dict): {guid: metadata} from get_active_preharmonized_studies()
        mode (str): One of 'all', 'max', 'selected'
        selected_ids (list[str]): Required when mode='selected'
        max_count (int): Required when mode='max'

    Returns:
        list[tuple[str, dict]]
    """
    if mode == "all":
        return list(studies.items())
    elif mode == "max":
        return list(studies.items())[:max_count]
    elif mode == "selected":
        result = [(sid, studies[sid]) for sid in (selected_ids or []) if sid in studies]
        not_found = [sid for sid in (selected_ids or []) if sid not in studies]
        if not_found:
            print(
                f"Warning: {len(not_found)} selected IDs not found in the catalog: {not_found}"
            )
        return result
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use all, max, or selected.")


# ── PFB workflow ───────────────────────────────────────────────────────────────


def download_pfb_for_study(study_metadata, study_dir, file_client):
    """Download the preharmonized PFB (.avro) for a study if not already present.

    Args:
        study_metadata (dict): Entry from get_active_preharmonized_studies()
        study_dir (str): Directory to save the .avro file
        file_client: Gen3File client instance

    Returns:
        str | None: Filename of the downloaded .avro, or None on failure
    """
    os.makedirs(study_dir, exist_ok=True)
    avro_filename = next(
        (f for f in os.listdir(study_dir) if f.endswith(".avro")), None
    )
    if avro_filename:
        print("  Archive already exists. Skipping download.")
        return avro_filename

    for obj in study_metadata.get("gen3_discovery", {}).get("objects", []):
        if not isinstance(obj, dict):
            continue
        if "Preharmonized" in obj.get("description", "") or "Preharmonized" in obj.get(
            "display_name", ""
        ):
            guid = obj.get("guid")
            if guid:
                print(f"  Downloading PFB (GUID: {guid})...")
                try:
                    file_client.download_single(guid, path=study_dir)
                    return next(
                        (f for f in os.listdir(study_dir) if f.endswith(".avro")), None
                    )
                except Exception as e:
                    print(f"  [! Error] Failed to download {guid}: {e}")
            break
    return None


def convert_pfb_to_tsv(directory, output_dir=None):
    """Convert every PFB (.avro) file in a directory to TSVs using the gen3 CLI.

    ``gen3 pfb to ... tsv`` writes one TSV per node in the PFB.

    Args:
        directory (str): Directory containing the .avro file(s) to convert.
            Non-avro files are skipped.
        output_dir (str | None): Name of the output directory, relative to
            ``directory``. ``None`` writes each PFB to its own
            ``{pfb_stem}__TSVS`` directory. Pass ``'tsvs'`` to use the gen3
            CLI's own default, which the metadata workflow below expects.

    Returns:
        bool: True if every PFB converted successfully or was already converted.
            False if any conversion failed. True with no PFBs present.
    """
    all_succeeded = True

    for pfb_path in sorted(Path(directory).iterdir()):
        if not pfb_path.is_file():
            continue
        if pfb_path.suffix != ".avro":
            print(f"  Skipping non-avro ({pfb_path.suffix}) file: {pfb_path.name}")
            continue

        target = output_dir or f"{pfb_path.stem}__TSVS"
        target_path = os.path.join(directory, target)
        if os.path.exists(target_path) and any(
            f.endswith(".tsv") for f in os.listdir(target_path)
        ):
            print(f"  TSVs already exist in {target}/. Skipping conversion.")
            continue

        print(f"  Converting {pfb_path.name} to TSVs in {target}/...")
        try:
            subprocess.run(
                ["gen3", "pfb", "to", "-i", pfb_path.name, "tsv", target],
                cwd=directory,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  [! Error] PFB conversion failed for {pfb_path.name}: {e}")
            all_succeeded = False

    return all_succeeded


# ── Metadata (data_dict / var_report) workflow ────────────────────────────────


def download_study_metadata(study_dir, file_client, session=None):
    """Download data_dict.xml and var_report.xml files for a study.

    Reads ga4gh_drs_uri entries from TSV files produced by the PFB workflow,
    downloads matching metadata files via presigned URLs, and extracts archives.

    Args:
        study_dir (str): Root directory for this study (contains tsvs/ subdirectory)
        file_client: Gen3File client instance
        session: requests.Session (optional; one is created if not provided)

    Returns:
        list[str]: Filenames that failed to download
    """
    session = session or requests.Session()
    tsvs_dir = os.path.join(study_dir, "tsvs")
    if not os.path.exists(tsvs_dir):
        print("  No tsvs/ directory found. Run the PFB workflow first.")
        return []

    tsv_files = [f for f in os.listdir(tsvs_dir) if f.endswith(".tsv")]
    if not tsv_files:
        print("  tsvs/ directory is empty.")
        return []

    metadata_dir = os.path.join(study_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    downloaded = set(os.listdir(metadata_dir))
    failed = []

    for tsv in tsv_files:
        rows = read_sv_as_list(os.path.join(tsvs_dir, tsv), delimiter="\t")

        for row in rows:
            drs_uri = row.get("ga4gh_drs_uri") or ""
            file_name = row.get("submitter_id") or ""

            if not file_name or not drs_uri.strip().startswith("drs"):
                continue
            if not is_direct_metadata(file_name) and not is_metadata_archive(file_name):
                continue
            if file_name in downloaded:
                continue

            parts = drs_uri.split(":")
            guid = ":".join(parts[2:]) if len(parts) >= 3 else parts[-1].lstrip("/")
            destination = os.path.join(metadata_dir, file_name)

            print(f"  Downloading: {file_name}")
            try:
                presigned_data = file_client.get_presigned_url(guid)
                if not presigned_data or "url" not in presigned_data:
                    print(f"    [! Error] No presigned URL for {guid}")
                    failed.append(file_name)
                    continue

                with session.get(presigned_data["url"], stream=True) as response:
                    response.raise_for_status()
                    with open(destination, "wb") as out:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                out.write(chunk)

                downloaded.add(file_name)

                if is_metadata_archive(file_name):
                    print("    Extracting archive...")
                    try:
                        extracted = extract_metadata_from_archive(
                            destination, metadata_dir
                        )
                        if extracted:
                            print(f"    Extracted: {extracted}")
                            downloaded.update(extracted)
                    finally:
                        if os.path.exists(destination):
                            os.remove(destination)
                else:
                    print("    Saved.")

            except Exception as e:
                print(f"    [! Error] {file_name}: {e}")
                failed.append(file_name)
                if os.path.exists(destination) and not os.path.isdir(destination):
                    os.remove(destination)

    if failed:
        failed_path = os.path.join(study_dir, "failed_downloads.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=4)
        print(f"  Saved {len(failed)} failed download(s) to {failed_path}")

    return failed


def check_missing_metadata(studies_to_run, studies_dir):
    """Return studies missing data_dict.xml or var_report.xml files.

    Args:
        studies_to_run (list[tuple[str, dict]]): Output of select_studies()
        studies_dir (str): Root directory containing per-study subdirectories

    Returns:
        list[dict]: One entry per study with missing files, with keys:
                    study_id, missing_data_dict, missing_var_report
    """
    missing = []
    for study_id, _ in studies_to_run:
        metadata_dir = os.path.join(studies_dir, study_id, "metadata")
        if os.path.exists(metadata_dir):
            files = os.listdir(metadata_dir)
            has_dict = any(
                is_direct_metadata(f) and "data_dict" in f.lower() for f in files
            )
            has_report = any(
                is_direct_metadata(f) and "var_report" in f.lower() for f in files
            )
        else:
            has_dict = has_report = False

        if not has_dict or not has_report:
            missing.append(
                {
                    "study_id": study_id,
                    "missing_data_dict": not has_dict,
                    "missing_var_report": not has_report,
                }
            )
    return missing
