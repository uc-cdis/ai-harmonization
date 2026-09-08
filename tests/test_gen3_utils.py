"""Tests for ai_harmonization.gen3_utils — filename helpers, study selection,
archive extraction, delimited-file reading and PFB conversion."""

import io
import os
import tarfile
import zipfile
from unittest.mock import patch

import pytest

from ai_harmonization.gen3_utils import (
    convert_pfb_to_tsv,
    extract_metadata_from_archive,
    is_direct_metadata,
    is_metadata_archive,
    read_sv_as_list,
    select_studies,
)


class TestIsDirectMetadata:
    @pytest.mark.parametrize(
        "fname",
        [
            "pht123_data_dict.xml",
            "pht123_var_report.xml",
            "STUDY_DATA_DICT.XML",
        ],
    )
    def test_returns_true(self, fname):
        assert is_direct_metadata(fname)

    @pytest.mark.parametrize(
        "fname",
        [
            "pht123_data_dict.tar.gz",
            "readme.txt",
            "pht123.avro",
            "data_dict.json",
        ],
    )
    def test_returns_false(self, fname):
        assert not is_direct_metadata(fname)


class TestIsMetadataArchive:
    @pytest.mark.parametrize(
        "fname",
        [
            # Long-form names.
            "data_dictionary.tar.gz",
            "variable_report.zip",
            "data_dictionary.tar",
            # Short-form names: dbGaP publishes archives under both styles.
            "data_dict.zip",
            "var_report.tar.gz",
            "phs999999.v1.pht999998.v1_data_dict.tar.gz",
            "phs999999.v1.pht999998.v1_var_report.zip",
            # Case is ignored.
            "STUDY_DATA_DICT.TAR.GZ",
        ],
    )
    def test_returns_true(self, fname):
        assert is_metadata_archive(fname)

    @pytest.mark.parametrize(
        "fname",
        [
            "pht123_data_dict.xml",  # a direct XML, not an archive
            "data_dictionary.pdf",  # right keyword, wrong suffix
            "other_archive.zip",  # right suffix, no metadata keyword
            "phenotype_manifest.tar.gz",
        ],
    )
    def test_returns_false(self, fname):
        assert not is_metadata_archive(fname)

    def test_xml_and_archive_checks_do_not_overlap(self):
        """download_study_metadata branches on these, so a file must match at
        most one: XMLs are saved directly, archives get extracted."""
        for fname in ["pht1_data_dict.xml", "pht1_data_dict.tar.gz", "readme.txt"]:
            assert not (is_direct_metadata(fname) and is_metadata_archive(fname))


class TestSelectStudies:
    @pytest.fixture
    def studies(self):
        return {
            "phs001": {"name": "Study 1"},
            "phs002": {"name": "Study 2"},
            "phs003": {"name": "Study 3"},
        }

    def test_mode_all_returns_all(self, studies):
        result = select_studies(studies, mode="all")
        assert len(result) == 3

    def test_mode_max_limits_count(self, studies):
        result = select_studies(studies, mode="max", max_count=2)
        assert len(result) == 2

    def test_mode_selected_returns_only_selected(self, studies):
        result = select_studies(
            studies, mode="selected", selected_ids=["phs001", "phs003"]
        )
        assert [r[0] for r in result] == ["phs001", "phs003"]

    def test_mode_selected_warns_on_missing(self, studies, capsys):
        select_studies(studies, mode="selected", selected_ids=["phs001", "phs999"])
        assert "not found" in capsys.readouterr().out

    def test_mode_selected_skips_missing(self, studies):
        result = select_studies(
            studies, mode="selected", selected_ids=["phs001", "phs999"]
        )
        assert len(result) == 1

    def test_unknown_mode_raises(self, studies):
        with pytest.raises(ValueError):
            select_studies(studies, mode="random")


class TestExtractMetadataFromArchive:
    def _make_zip(self, path, members):
        with zipfile.ZipFile(path, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)

    def _make_tar(self, path, members):
        with tarfile.open(path, "w:gz") as tf:
            for name, content in members.items():
                data = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    def test_extracts_xml_from_zip(self, tmp_path):
        archive = tmp_path / "data_dictionary.zip"
        dest = tmp_path / "out"
        dest.mkdir()
        self._make_zip(
            str(archive),
            {
                "subdir/pht001_data_dict.xml": "<x/>",
                "subdir/readme.txt": "ignore me",
            },
        )
        extracted = extract_metadata_from_archive(str(archive), str(dest))
        assert extracted == ["pht001_data_dict.xml"]
        assert (dest / "pht001_data_dict.xml").exists()

    def test_extracts_xml_from_tar(self, tmp_path):
        archive = tmp_path / "variable_report.tar.gz"
        dest = tmp_path / "out"
        dest.mkdir()
        self._make_tar(
            str(archive),
            {
                "pht001_var_report.xml": "<x/>",
                "other.csv": "skip",
            },
        )
        extracted = extract_metadata_from_archive(str(archive), str(dest))
        assert extracted == ["pht001_var_report.xml"]

    def test_non_metadata_files_skipped(self, tmp_path):
        archive = tmp_path / "data_dictionary.zip"
        dest = tmp_path / "out"
        dest.mkdir()
        self._make_zip(str(archive), {"notes.txt": "ignore", "data.csv": "also ignore"})
        extracted = extract_metadata_from_archive(str(archive), str(dest))
        assert extracted == []

    def test_bad_archive_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        dest = tmp_path / "out"
        dest.mkdir()
        result = extract_metadata_from_archive(str(bad), str(dest))
        assert result == []


class TestReadSvAsList:
    def test_reads_csv(self, tmp_path):
        path = tmp_path / "rows.csv"
        path.write_text("name,age\nada,36\ngrace,45\n")
        rows = read_sv_as_list(str(path))
        assert rows == [{"name": "ada", "age": "36"}, {"name": "grace", "age": "45"}]

    def test_reads_tsv_with_delimiter(self, tmp_path):
        path = tmp_path / "rows.tsv"
        path.write_text(
            "submitter_id\tga4gh_drs_uri\npht1_data_dict.xml\tdrs://dg.4503:abc\n"
        )
        rows = read_sv_as_list(str(path), delimiter="\t")
        assert rows[0]["ga4gh_drs_uri"] == "drs://dg.4503:abc"

    def test_header_only_returns_empty(self, tmp_path):
        path = tmp_path / "rows.tsv"
        path.write_text("a\tb\n")
        assert read_sv_as_list(str(path), delimiter="\t") == []

    def test_missing_column_is_absent_not_an_error(self, tmp_path):
        """The metadata workflow relies on this to skip TSVs without DRS URIs."""
        path = tmp_path / "rows.tsv"
        path.write_text("other_column\tvalue\nfoo\tbar\n")
        rows = read_sv_as_list(str(path), delimiter="\t")
        assert rows[0].get("ga4gh_drs_uri") is None


class TestConvertPfbToTsv:
    """The gen3 CLI is mocked out; these cover the dispatch and skip logic."""

    @staticmethod
    def _write_tsv(directory, subdir):
        os.makedirs(os.path.join(directory, subdir), exist_ok=True)
        with open(os.path.join(directory, subdir, "subject.tsv"), "w") as f:
            f.write("a\tb\n")

    def test_converts_avro_with_explicit_output_dir(self, tmp_path):
        (tmp_path / "study.avro").write_bytes(b"avro")
        with patch("ai_harmonization.gen3_utils.subprocess.run") as run:
            assert convert_pfb_to_tsv(str(tmp_path), output_dir="tsvs") is True
        command = run.call_args.args[0]
        assert command == ["gen3", "pfb", "to", "-i", "study.avro", "tsv", "tsvs"]
        assert run.call_args.kwargs["cwd"] == str(tmp_path)

    def test_default_output_dir_is_per_pfb(self, tmp_path):
        (tmp_path / "study.avro").write_bytes(b"avro")
        with patch("ai_harmonization.gen3_utils.subprocess.run") as run:
            convert_pfb_to_tsv(str(tmp_path))
        assert run.call_args.args[0][-1] == "study__TSVS"

    def test_skips_non_avro_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore me")
        with patch("ai_harmonization.gen3_utils.subprocess.run") as run:
            assert convert_pfb_to_tsv(str(tmp_path)) is True
        run.assert_not_called()

    def test_skips_when_tsvs_already_exist(self, tmp_path):
        (tmp_path / "study.avro").write_bytes(b"avro")
        self._write_tsv(str(tmp_path), "tsvs")
        with patch("ai_harmonization.gen3_utils.subprocess.run") as run:
            assert convert_pfb_to_tsv(str(tmp_path), output_dir="tsvs") is True
        run.assert_not_called()

    def test_empty_output_dir_does_not_count_as_converted(self, tmp_path):
        (tmp_path / "study.avro").write_bytes(b"avro")
        os.makedirs(tmp_path / "tsvs")
        with patch("ai_harmonization.gen3_utils.subprocess.run") as run:
            convert_pfb_to_tsv(str(tmp_path), output_dir="tsvs")
        run.assert_called_once()

    def test_converts_every_pfb_in_directory(self, tmp_path):
        (tmp_path / "one.avro").write_bytes(b"avro")
        (tmp_path / "two.avro").write_bytes(b"avro")
        with patch("ai_harmonization.gen3_utils.subprocess.run") as run:
            convert_pfb_to_tsv(str(tmp_path))
        assert run.call_count == 2

    def test_returns_false_when_cli_fails(self, tmp_path):
        (tmp_path / "study.avro").write_bytes(b"avro")
        with patch(
            "ai_harmonization.gen3_utils.subprocess.run", side_effect=OSError("boom")
        ):
            assert convert_pfb_to_tsv(str(tmp_path)) is False

    def test_one_failure_does_not_stop_the_others(self, tmp_path):
        (tmp_path / "one.avro").write_bytes(b"avro")
        (tmp_path / "two.avro").write_bytes(b"avro")
        with patch(
            "ai_harmonization.gen3_utils.subprocess.run",
            side_effect=[OSError("boom"), None],
        ) as run:
            assert convert_pfb_to_tsv(str(tmp_path)) is False
        assert run.call_count == 2

    def test_no_pfbs_present_succeeds(self, tmp_path):
        assert convert_pfb_to_tsv(str(tmp_path)) is True
