"""Tests for embedding_model.corpus and embedding_model.leakage: planted leaks, the split, the build."""

import json
import pathlib

import pandas as pd
import pytest

from embedding_model import benchmarks, corpus, leakage
from embedding_model.__main__ import main
from embedding_model.download import file_sha256
from embedding_model.pairs import common, pairs_sha256

# A fictional benchmark: two source variables, from two fictional studies.
MAPPINGS = pd.DataFrame(
    {
        "source_study_id": ["phs999901", "phs999902"],
        "source_table_id": ["pht999901.v1", "pht999902.v1"],
        "source_variable_id": ["phv99999901.v1", "phv99999902.v1"],
        "source_variable_name": ["abcd_score", "efgh_level"],
        "source_type": ["float", "string"],
        "source_description": ["Abcd efgh score", ""],
        "source_values": ["", ""],
        "expected_target": ["node1.abcd_index", "node2.efgh_value;node3.efgh_value"],
    }
)
LONG_DESCRIPTION = (
    "Abcd index of the ijkl, measured at the first visit of each mnop participant."
)
TARGETS = pd.DataFrame(
    {
        "target_property": [
            "node1.abcd_index",
            "node2.efgh_value",
            "node3.efgh_value",
            "node4.ijkl_other",
        ],
        "target_type": ["float", "string", "string", "string"],
        "target_description": [
            LONG_DESCRIPTION,
            "The efgh value",
            "Efgh value in node three",
            "Other ijkl",
        ],
        "target_values": ["", "", "", ""],
    }
)


def pair(anchor_name="", positive_name="", **fields):
    """One pair row, with every column, the given fields set."""
    row = {
        "anchor": anchor_name,
        "positive": positive_name,
        "source": "ncit",
        "kind": "name",
        "item": "C999901",
        "term_type": "SY",
        "term_group": "paraphrase",
        "status": "current",
        "study": "",
        "value_list": "",
        "anchor_name": anchor_name,
        "anchor_type": "string",
        "anchor_description": "",
        "anchor_values": [],
        "positive_name": positive_name,
        "positive_type": "string",
        "positive_description": "",
        "positive_values": [],
    }
    row.update(fields)
    return row


# One planted leak of each kind, in the shapes that pairs prepare writes: a
# second definition is a description_description pair's anchor name, and a
# name_to_description pair's description is its positive name.
PLANTED = [
    pair("Abcd Score", "Abcd Index"),
    pair("efgh_level", "Efgh", positive_description="The efgh value"),
    pair("Abcd efgh score", "abcd_index", kind="description_description"),
    pair("efgh value", "EFGH LEVEL"),
    pair("abcd score", "Abcd", positive_description=f"{LONG_DESCRIPTION} More."),
    pair("abcd_score", f"{LONG_DESCRIPTION} More.", kind="name_to_description"),
    pair("Mnop", "Qrst", study="phs999902|phs999777"),
]
INNOCENT = [
    pair("abcd score", "Something else"),
    pair("abcd score", "efgh value"),
    pair("Uvwx", "Yzab", anchor_values=["abcd index"], positive_values=["abcd score"]),
    pair(
        "efgh_level", "Cdef", positive_description="Efgh value in node three and more"
    ),
    pair("Ghij", "Klmn", study="phs999555"),
]


class TestLeakage:
    """Leakage prevention on a fictional benchmark, one planted leak of each kind."""

    @pytest.fixture
    def audited(self):
        """The audit of the planted and the innocent pairs, in that order."""
        pairs = pd.DataFrame(PLANTED + INNOCENT)
        return leakage.audit(pairs, {"abcd": (TARGETS, MAPPINGS)})

    def test_each_planted_leak_is_dropped_and_nothing_else(self, audited):
        """Every planted leak is dropped, with its finding, and no innocent pair is."""
        findings, _ = audited
        assert findings.pair.tolist() == list(range(len(PLANTED)))
        assert findings.finding.tolist() == [leakage.ANSWER_KEY] * 6 + [
            leakage.BENCHMARK_STUDY
        ]

    def test_the_fields_that_matched_are_named(self, audited):
        """Each answer key names the two fields that matched, and how."""
        findings, _ = audited
        assert findings.fields.tolist()[:6] == [
            "anchor name = source name, positive name = target name",
            "anchor name = source name, positive description = target description",
            "anchor name = source description, positive name = target name",
            "positive name = source name, anchor name = target name",
            "anchor name = source name, positive description ~ target description",
            "anchor name = source name, positive name ~ target description",
        ]
        assert findings.target_property.tolist()[0] == "node1.abcd_index"
        assert findings.target_description.tolist()[0] == LONG_DESCRIPTION
        assert findings.source_variable_description.tolist()[0] == "Abcd efgh score"
        assert findings.target_description.tolist()[6] == ""
        assert findings.study_id.tolist()[6] == "phs999902"

    def test_a_pair_that_leaks_two_benchmarks_has_a_finding_for_each(self):
        """One pair can leak two benchmarks, and each one gets its finding."""
        pairs = pd.DataFrame([pair("Abcd Score", "Abcd Index")])
        findings, _ = leakage.audit(
            pairs, {"abcd": (TARGETS, MAPPINGS), "efgh": (TARGETS, MAPPINGS)}
        )
        assert findings.benchmark.tolist() == ["abcd", "efgh"]
        assert findings.pair.tolist() == [0, 0]

    def test_accents_and_case_do_not_hide_an_answer_key(self):
        """A pair whose texts differ from a benchmark's only by accents and case is an answer key."""
        pairs = pd.DataFrame([pair("Ábcd Scöre", "ABCD INDEX")])
        findings, _ = leakage.audit(pairs, {"abcd": (TARGETS, MAPPINGS)})
        assert findings.finding.tolist() == [leakage.ANSWER_KEY]

    def test_shared_vocabulary_is_reported(self, audited):
        """Benchmark texts in the kept pairs are counted, not dropped."""
        shared = audited[1]["abcd"]
        assert shared["rows_with_shared_source"] == 2
        assert shared["rows_with_shared_target"] == 1
        assert {item["text"] for item in shared["most_common"]} == {
            "abcd score",
            "efgh level",
            "efgh value",
        }

    def test_near_duplicates_need_whole_words_and_ten_of_them(self):
        """One text contains the other as whole words, and the shorter has 10 words."""
        reference = "one two three four five six seven eight nine ten"
        texts = [
            f"before {reference} after",
            f"{reference} eleven",
            "one two three four five six seven eight nine",
            f"x{reference}",
            reference,
        ]
        matched = leakage.near_duplicates(
            texts, [reference, f"zero {reference}"], min_words=10
        )
        assert matched == {
            (f"before {reference} after", reference),
            (f"{reference} eleven", reference),
            (reference, f"zero {reference}"),
        }

    def test_the_threshold_is_read_when_the_audit_runs(self, monkeypatch):
        """The module's threshold applies when the audit runs, and must cover a shingle."""
        pairs = pd.DataFrame(
            [
                pair(
                    "efgh_level",
                    "Cdef",
                    positive_description="Efgh value in node three and more",
                )
            ]
        )
        benchmark = {"abcd": (TARGETS, MAPPINGS)}
        assert leakage.audit(pairs, benchmark)[0].empty
        monkeypatch.setattr(leakage, "NEAR_DUPLICATE_WORDS", 5)
        findings, _ = leakage.audit(pairs, benchmark)
        assert findings.fields.tolist() == [
            "anchor name = source name, positive description ~ target description"
        ]
        with pytest.raises(ValueError):
            leakage.near_duplicates(["a b"], ["a b"], min_words=2)

    def test_benchmark_texts_are_cleaned_and_named_after_their_last_dot(self):
        """A benchmark's texts are cleaned before keying, as the pairs' texts were."""
        mappings = MAPPINGS.assign(
            source_description=["Abcd &amp; efgh <b>score</b>", ""],
            expected_target=["node1.sub.abcd_index", "node4.ijkl_other"],
        )
        targets = TARGETS.assign(
            target_property=["node1.sub.abcd_index", *TARGETS.target_property[1:]]
        )
        sources, expected = leakage.benchmark_texts("abcd", targets, mappings)
        assert "abcd efgh score" in set(sources.key)
        assert "abcd index" in set(expected.key)

    def test_an_expected_target_missing_from_the_targets_is_an_error(self):
        """A mapping to a target that the targets file lacks cannot be checked."""
        mappings = MAPPINGS.assign(expected_target=["node9.wxyz", "node4.ijkl_other"])
        with pytest.raises(ValueError, match="node9.wxyz"):
            leakage.benchmark_texts("abcd", TARGETS, mappings)

    def test_a_target_listed_twice_is_an_error(self):
        """A targets file that lists one property twice cannot be checked."""
        targets = pd.concat([TARGETS, TARGETS.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="node1.abcd_index"):
            leakage.benchmark_texts("abcd", targets, MAPPINGS)

    def test_keys_are_those_of_pairs(self):
        """A key is what pairs.common.key writes: accents go, Greek letters stay."""
        texts = pd.Series(["Abcd_Efgh (ijkl)!", "α-tocopherol", "Größe", None])
        assert leakage.keys(texts).tolist() == [
            common.key("Abcd_Efgh (ijkl)!"),
            common.key("α-tocopherol"),
            common.key("Größe"),
            "",
        ]
        assert leakage.keys(texts).tolist()[1] == "α tocopherol"


def write_pairs(inputs, source, rows):
    """Write a fictional source's pairs and their manifest, as ``pairs prepare`` would."""
    folder = inputs / "pairs"
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).assign(source=source)
    frame.to_parquet(folder / f"{source}.parquet", index=False)
    written = pd.read_parquet(folder / f"{source}.parquet")
    record = {
        "source": source,
        "built": "2026-01-01",
        "files": {
            f"{source}.parquet": {
                "rows": len(written),
                "sha256": file_sha256(folder / f"{source}.parquet"),
                "csv_sha256": pairs_sha256(written),
            }
        },
        "sources": [
            {
                "role": "fetch",
                "name": source,
                "version": "1",
                "fetched": "2026-01-01",
                "files": {},
            }
        ],
        "counts": {},
    }
    (folder / f"{source}_manifest.json").write_text(json.dumps(record))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """An inputs folder with the fictional benchmark and two sources' pairs."""
    monkeypatch.setattr(corpus.benchmarks, "BUILDERS", {"abcd": None})
    targets = TARGETS.to_dict("records")
    mappings = [
        {**row, "expected_target": row["expected_target"].split(";")}
        for row in MAPPINGS.to_dict("records")
    ]
    (tmp_path / "benchmarks" / "abcd").mkdir(parents=True)
    benchmarks.write_benchmark(
        tmp_path / "benchmarks" / "abcd", "abcd", "1", targets, mappings, [], {}
    )
    duplicate = pair("Wxyz", "Wxyz Abcd")
    write_pairs(tmp_path, "ncit", PLANTED[:3] + [duplicate, pair("Stuv", "Stuv Abcd")])
    write_pairs(tmp_path, "cdisc", [duplicate, pair("Opqr", "Opqr Abcd")])
    return tmp_path


class TestBuild:
    """``corpus build`` on a fictional inputs folder."""

    def test_the_corpus_and_its_manifest(self, inputs, capsys):
        """Duplicates are kept once, leaks dropped, and the manifest counts both."""
        assert corpus.build("abcd-corpus", ["ncit", "cdisc"], inputs) == 0
        folder = inputs / "corpora" / "abcd-corpus"
        kept = pd.read_parquet(folder / "pairs.parquet")
        dropped = pd.read_parquet(folder / "dropped.parquet")
        written = json.loads((folder / "manifest.json").read_text())
        assert sorted(kept.anchor_name) == ["Opqr", "Stuv", "Wxyz"]
        assert set(kept.split) <= {corpus.TRAIN, corpus.VALIDATION}
        assert kept.set_index("anchor_name").source["Wxyz"] == "ncit"
        assert dropped.finding.tolist() == [leakage.ANSWER_KEY] * 3
        assert dropped.benchmark_source_variable_name.tolist() == [
            "abcd_score",
            "efgh_level",
            "abcd_score",
        ]
        assert written["counts"]["rows"] == {
            "combined": 7,
            "duplicates_removed": 1,
            "leakage_removed": 3,
            "kept": 3,
            corpus.TRAIN: int((kept.split == corpus.TRAIN).sum()),
            corpus.VALIDATION: int((kept.split == corpus.VALIDATION).sum()),
        }
        assert written["counts"]["duplicates_by_source"] == {"cdisc": 1}
        assert written["counts"]["leakage_findings"]["by_benchmark"] == {"abcd": 3}
        assert written["files"]["pairs.parquet"] == {
            "rows": 3,
            "sha256": file_sha256(folder / "pairs.parquet"),
            "csv_sha256": pairs_sha256(kept),
        }
        assert written["files"]["dropped.parquet"] == {
            "rows": 3,
            "sha256": file_sha256(folder / "dropped.parquet"),
            "csv_sha256": pairs_sha256(dropped),
        }
        assert [(entry["role"], entry["name"]) for entry in written["sources"]] == [
            ("pairs", "ncit"),
            ("pairs", "cdisc"),
            ("benchmark", "abcd"),
        ]
        assert written["settings"]["validation_share"] == corpus.VALIDATION_SHARE
        read, record = corpus.read_corpus(inputs, "abcd-corpus")
        assert read.equals(kept) and record == written
        printed = capsys.readouterr().out
        assert "checked against: abcd v1" in printed
        assert "3 pairs dropped for leakage (findings: answer_key 3)" in printed

    def test_a_rebuild_replaces_the_previous_corpus(self, inputs, capsys):
        """A second build of one name replaces the first, and leaves no work folder."""
        assert corpus.build("abcd", ["ncit"], inputs) == 0
        assert corpus.build("abcd", ["cdisc"], inputs) == 0
        kept = pd.read_parquet(inputs / "corpora" / "abcd" / "pairs.parquet")
        assert set(kept.source) == {"cdisc"}
        assert [path.name for path in (inputs / "corpora").iterdir()] == ["abcd"]
        assert "0 pairs dropped for leakage\n" in capsys.readouterr().out

    def test_a_copy_gives_its_study_to_the_pair_kept(self, inputs, capsys):
        """A later source's copy with a benchmark study gets the kept pair dropped."""
        write_pairs(inputs, "ncit", [pair("Wxyz", "Wxyz Abcd"), pair("Stuv", "Stuv")])
        write_pairs(inputs, "phenx", [pair("Wxyz", "Wxyz Abcd", study="phs999902")])
        assert corpus.build("abcd", ["ncit", "phenx"], inputs) == 0
        dropped = pd.read_parquet(inputs / "corpora" / "abcd" / "dropped.parquet")
        assert dropped[["source", "finding", "study"]].values.tolist() == [
            ["ncit", leakage.BENCHMARK_STUDY, "phs999902"]
        ]

    def test_the_pair_kept_holds_the_studies_of_every_copy(self, inputs):
        """Copies in two sources give the pair kept both of their studies."""
        write_pairs(inputs, "ncit", [pair("Wxyz", "Wxyz Abcd", study="phs999555")])
        write_pairs(inputs, "phenx", [pair("Wxyz", "Wxyz Abcd", study="phs999444")])
        assert corpus.build("abcd", ["ncit", "phenx"], inputs) == 0
        kept = pd.read_parquet(inputs / "corpora" / "abcd" / "pairs.parquet")
        assert kept[["source", "study"]].values.tolist() == [
            ["ncit", "phs999444|phs999555"]
        ]

    def test_a_corpus_is_read_back_checked(self, inputs):
        """read_corpus refuses a bad name, a missing corpus, and changed pairs."""
        assert corpus.build("abcd-corpus", ["ncit", "cdisc"], inputs) == 0
        with pytest.raises(ValueError, match="not a corpus name"):
            corpus.read_corpus(inputs, "../abcd")
        with pytest.raises(FileNotFoundError, match="corpus build --name efgh"):
            corpus.read_corpus(inputs, "efgh")
        path = inputs / "corpora" / "abcd-corpus" / "pairs.parquet"
        pd.read_parquet(path).iloc[:1].to_parquet(path, index=False)
        with pytest.raises(ValueError, match="differs from its manifest"):
            corpus.read_corpus(inputs, "abcd-corpus")

    def test_a_corpus_manifest_or_columns_that_do_not_fit_are_refused(self, inputs):
        """An empty manifest, another corpus's, or a missing column names the command."""
        assert corpus.build("abcd-corpus", ["ncit", "cdisc"], inputs) == 0
        folder = inputs / "corpora" / "abcd-corpus"
        record = (folder / "manifest.json").read_text()
        (folder / "manifest.json").write_text("")
        with pytest.raises(ValueError, match="corpus build --name abcd-corpus"):
            corpus.read_corpus(inputs, "abcd-corpus")
        (folder / "manifest.json").write_text(
            record.replace('"corpus": "abcd-corpus"', '"corpus": "efgh"')
        )
        with pytest.raises(ValueError, match="the manifest of corpus 'efgh'"):
            corpus.read_corpus(inputs, "abcd-corpus")
        pairs = pd.read_parquet(folder / "pairs.parquet").drop(columns="split")
        pairs.to_parquet(folder / "pairs.parquet", index=False)
        rewritten = json.loads(record)
        rewritten["files"]["pairs.parquet"]["sha256"] = file_sha256(
            folder / "pairs.parquet"
        )
        (folder / "manifest.json").write_text(json.dumps(rewritten))
        with pytest.raises(ValueError, match="columns that its step writes"):
            corpus.read_corpus(inputs, "abcd-corpus")

    def test_a_corpus_named_previous_can_be_rebuilt(self, inputs):
        """The swap's own folder name cannot clash with a corpus's name."""
        assert corpus.build("previous", ["ncit"], inputs) == 0
        assert corpus.build("previous", ["cdisc"], inputs) == 0

    def test_a_failed_swap_puts_the_previous_corpus_back(self, inputs, monkeypatch):
        """When the new corpus cannot move into place, the previous one stays."""
        assert corpus.build("abcd", ["ncit"], inputs) == 0
        rename, calls = pathlib.Path.rename, []

        def failing(self, target):
            """Fail the second rename, the new corpus's move into place."""
            calls.append(self)
            if len(calls) == 2:
                raise OSError("abcd")
            return rename(self, target)

        monkeypatch.setattr(pathlib.Path, "rename", failing)
        with pytest.raises(OSError):
            corpus.build("abcd", ["cdisc"], inputs)
        monkeypatch.setattr(pathlib.Path, "rename", rename)
        kept = pd.read_parquet(inputs / "corpora" / "abcd" / "pairs.parquet")
        assert set(kept.source) == {"ncit"}

    def test_a_manifest_without_a_field_names_the_command(self, inputs, capsys):
        """A pairs manifest that lacks a field asks for pairs prepare again."""
        path = inputs / "pairs" / "ncit_manifest.json"
        record = json.loads(path.read_text())
        del record["files"]
        path.write_text(json.dumps(record))
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        error = capsys.readouterr().err
        assert "KeyError: 'files'" in error
        assert "pairs prepare --source ncit" in error
        write_pairs(inputs, "ncit", [pair("Wxyz", "Wxyz Abcd")])
        record = json.loads(path.read_text())
        del record["sources"][0]["version"]
        path.write_text(json.dumps(record))
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "KeyError: 'version'" in capsys.readouterr().err

    def test_another_sources_manifest_is_refused(self, inputs, capsys):
        """A pairs manifest that names another source is not read as this one's."""
        manifests = inputs / "pairs"
        (manifests / "ncit_manifest.json").write_text(
            (manifests / "cdisc_manifest.json").read_text()
        )
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "the manifest of source 'cdisc'" in capsys.readouterr().err

    def test_a_benchmark_that_cannot_be_checked_stops_the_build(self, inputs, capsys):
        """A benchmark whose expected target is not among its targets stops the build."""
        targets = TARGETS[TARGETS.target_property == "node1.abcd_index"]
        mappings = [
            {**row, "expected_target": row["expected_target"].split(";")}
            for row in MAPPINGS.to_dict("records")
        ]
        benchmarks.write_benchmark(
            inputs / "benchmarks" / "abcd",
            "abcd",
            "1",
            targets.to_dict("records"),
            mappings,
            [],
            {},
        )
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        error = capsys.readouterr().err
        assert "leakage prevention cannot run" in error
        assert "node2.efgh_value" in error
        assert not (inputs / "corpora" / "abcd").exists()

    def test_sources_without_pairs_build_no_corpus(self, inputs, capsys):
        """Sources that hold no pairs are reported, and no corpus is written."""
        write_pairs(inputs, "ncit", pd.DataFrame([pair()]).iloc[:0])
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "no pairs in ncit" in capsys.readouterr().err
        assert not (inputs / "corpora" / "abcd").exists()

    def test_a_corpus_that_would_be_empty_is_not_written(self, inputs, capsys):
        """When leakage prevention drops every pair, no corpus is written."""
        write_pairs(inputs, "ncit", PLANTED[:3])
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "dropped all 3 pairs" in capsys.readouterr().err
        assert not (inputs / "corpora" / "abcd").exists()

    def test_pairs_that_cannot_be_read_name_the_source(self, inputs, capsys):
        """A broken pairs file names its source and the command that remakes it."""
        (inputs / "pairs" / "ncit_manifest.json").write_text("")
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        error = capsys.readouterr().err
        assert "ncit_manifest.json cannot be read" in error
        assert "pairs prepare --source ncit" in error

    def test_a_missing_pairs_manifest_is_named(self, inputs, capsys):
        """Pairs without their manifest name the manifest, not the pairs."""
        (inputs / "pairs" / "ncit_manifest.json").unlink()
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        error = capsys.readouterr().err
        assert "ncit_manifest.json is missing" in error
        assert "pairs prepare --source ncit" in error

    def test_pairs_without_a_column_are_refused(self, inputs, capsys):
        """Pairs whose columns are not those of pairs prepare are not read."""
        write_pairs(inputs, "ncit", [pair("Wxyz", "Wxyz Abcd")])
        path = inputs / "pairs" / "ncit.parquet"
        pd.read_parquet(path).drop(columns="study").to_parquet(path, index=False)
        record = json.loads((inputs / "pairs" / "ncit_manifest.json").read_text())
        record["files"]["ncit.parquet"]["sha256"] = file_sha256(path)
        (inputs / "pairs" / "ncit_manifest.json").write_text(json.dumps(record))
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "does not have the columns" in capsys.readouterr().err

    def test_pairs_are_checked_against_the_fetch_in_the_inputs(self, inputs, capsys):
        """Pairs from the fetch in the inputs are read; pairs from another are not."""
        fetch = inputs / "sources" / "ncit" / "manifest.json"
        fetch.parent.mkdir(parents=True)
        fetch.write_text(json.dumps({"source": "ncit", "files": {}}))
        assert corpus.build("abcd", ["ncit"], inputs) == 0
        files = {"abcd.parquet": {"rows": 1, "csv_sha256": "abcd"}}
        fetch.write_text(json.dumps({"source": "ncit", "files": files}))
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "come from another fetch" in capsys.readouterr().err

    def test_a_fetch_that_cannot_be_read_names_its_command(self, inputs, capsys):
        """A broken fetch manifest asks for the fetch again, not for the pairs."""
        fetch = inputs / "sources" / "ncit" / "manifest.json"
        fetch.parent.mkdir(parents=True)
        fetch.write_text("")
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "sources fetch --source ncit" in capsys.readouterr().err

    def test_the_split_follows_a_fixed_hash_of_the_group(self):
        """A group's side comes from a fixed hash, and NCIt and CDISC share groups."""
        assert corpus.bucket("ncit:C0") == 117
        assert corpus.bucket("ncit:C51") == 0
        pairs = pd.DataFrame(
            {
                "source": ["ncit", "cdisc", "cadsr", "ncit", "cdisc"],
                "item": ["C0", "C0", "C0", "C51", "C51"],
            }
        )
        assert corpus.group_keys(pairs).tolist() == [
            "ncit:C0",
            "ncit:C0",
            "cadsr:C0",
            "ncit:C51",
            "ncit:C51",
        ]
        assert corpus.splits(pairs).tolist() == [
            corpus.TRAIN,
            corpus.TRAIN,
            corpus.TRAIN,
            corpus.VALIDATION,
            corpus.VALIDATION,
        ]

    def test_no_benchmark_built_stops_the_build(self, tmp_path, capsys):
        """Without a benchmark, the corpus cannot be checked, so it is not built."""
        assert corpus.build("abcd", ["ncit"], tmp_path) == 1
        assert "benchmarks build" in capsys.readouterr().err
        assert not (tmp_path / "corpora").exists()

    def test_a_benchmark_with_a_file_missing_stops_the_build(self, inputs, capsys):
        """A built benchmark that cannot be read is an error, not skipped."""
        (inputs / "benchmarks" / "abcd" / "abcd_targets.csv").unlink()
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "a benchmark cannot be read" in capsys.readouterr().err

    def test_a_source_not_prepared_names_the_command(self, inputs, capsys):
        """A source with no pairs stops the build and names the command that makes them."""
        assert corpus.build("abcd", ["ncit", "phenx"], inputs) == 1
        assert "pairs prepare --source phenx" in capsys.readouterr().err

    def test_an_unknown_source_is_named(self, inputs, capsys):
        """A source that is not one of READERS is named, not skipped."""
        assert corpus.build("abcd", ["ncit", "abcd"], inputs) == 1
        assert "unknown: ['abcd']" in capsys.readouterr().err

    def test_pairs_changed_since_their_manifest_are_refused(self, inputs, capsys):
        """Pairs whose content differs from their manifest's checksum are not read."""
        path = inputs / "pairs" / "ncit.parquet"
        pd.read_parquet(path).iloc[:1].to_parquet(path, index=False)
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "differs from its manifest" in capsys.readouterr().err

    def test_pairs_are_checked_by_the_bytes_of_their_file(self, inputs, capsys):
        """The same table written again with other bytes is not the file that was written."""
        path = inputs / "pairs" / "ncit.parquet"
        pd.read_parquet(path).to_parquet(path, index=False, compression="gzip")
        assert corpus.build("abcd", ["ncit"], inputs) == 1
        assert "differs from its manifest" in capsys.readouterr().err

    def test_a_name_that_is_not_a_folder_name_is_refused(self, inputs, capsys):
        """A corpus name must be a plain folder name."""
        assert corpus.build("../abcd", ["ncit"], inputs) == 1
        assert "not a corpus name" in capsys.readouterr().err

    def test_the_command(self, inputs):
        """``corpus build`` builds a corpus from the sources named."""
        argv = [
            "corpus",
            "build",
            "--inputs",
            str(inputs),
            "--name",
            "abcd",
            "--source",
            "ncit",
        ]
        assert main(argv) == 0
        assert (inputs / "corpora" / "abcd" / "manifest.json").exists()
