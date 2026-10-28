"""Tests for embedding_model.pairs: the shared star, and each source on small fictional tables."""

import json

import pandas as pd

from embedding_model import pairs, sources
from embedding_model.__main__ import main
from embedding_model.download import file_sha256
from embedding_model.pairs import (
    cadsr,
    cde,
    cdisc,
    common,
    loinc,
    ncit,
    nhanes,
    obo,
    phenx,
)
from embedding_model.pairs.common import (
    CURRENT,
    DRAFT,
    RETIRED,
    Item,
    Text,
)


def table(**columns):
    """A table from its columns, each a list."""
    return pd.DataFrame(columns)


def made(items):
    """The pairs that ``star`` makes, as (kind, anchor name, positive name) rows."""
    found, _ = common.star(items, source="abcd")
    return list(zip(found.kind, found.anchor_name, found.positive_name))


class TestClean:
    """The cleaning that every text goes through."""

    def test_markup_entities_and_spaces(self):
        """HTML tags go, entities are unescaped, and whitespace runs become one space."""
        assert common.clean("<u>Over the</u>  past <b>7 days</b>, how") == (
            "Over the past 7 days, how"
        )
        assert common.clean("ABCD1 c.460G&gt;A") == "ABCD1 c.460G>A"
        assert common.clean(" Abcd\n\tefgh\xa0ijkl ") == "Abcd efgh ijkl"

    def test_text_in_angle_brackets_is_kept(self):
        """A placeholder in angle brackets is text, not a tag, beside real tags too."""
        assert common.clean("Before <abcd time point>") == "Before <abcd time point>"
        assert common.clean("<b>Before</b> <abcd time point>") == (
            "Before <abcd time point>"
        )
        assert common.clean("Abcd <5 <i>efgh</i>") == "Abcd <5 efgh"

    def test_inline_tags_join_and_block_tags_separate(self):
        """Inline tags leave no space, so units stay whole; block tags add a space."""
        assert common.clean("Abcd H<sub>2</sub>O") == "Abcd H2O"
        assert common.clean("<i>in vitro</i>-abcd") == "in vitro-abcd"
        assert common.clean("Abcd<br>efgh<p>ijkl</p>") == "Abcd efgh ijkl"

    def test_mojibake_is_repaired(self):
        """UTF-8 read as cp1252 is repaired, beside correct text; other text is kept."""
        assert common.clean("I\u00e2\u20ac\u2122m abcd") == "I\u2019m abcd"
        assert common.clean("Caf\u00e9 \u00e2\u20ac\u2122s \u2265 5") == (
            "Caf\u00e9 \u2019s \u2265 5"
        )
        assert common.clean("\u00c3bcd") == "\u00c3bcd"

    def test_mojibake_read_wrongly_twice_or_with_undefined_bytes(self):
        """Text read wrongly twice, or with cp1252's undefined bytes, is repaired."""
        twice = "Iâ€™m".encode("utf-8").decode("cp1252")
        assert common.clean(twice) == "I’m"
        assert common.clean("say â€œhiâ€\x9d") == ("say “hi”")

    def test_correct_text_that_looks_like_mojibake_is_kept(self):
        """A repair that would give a letter of another script is not made."""
        assert common.clean("Fuß–Länge") == "Fuß–Länge"

    def test_an_accented_capital_before_punctuation_is_kept(self):
        """A correct accented capital before a quote or a sign is not repaired."""
        for text in ("JOS\u00c9\u2019s", "FIANC\u00c9\u00ae", "\u201cCAF\u00c9\u201d"):
            assert common.clean(text) == text

    def test_entities_are_read_once_and_only_when_whole(self):
        """Entities are unescaped once, tag or no tag, and only with their semicolon."""
        assert common.clean("I&acirc;&#128;&#153;m") == "I\u2019m"
        assert common.clean("x &amp;lt;5") == "x &lt;5"
        assert common.clean("<b>x</b> &amp;lt;5") == "x &lt;5"
        assert common.clean("Yes&not sure") == "Yes&not sure"

    def test_brackets_that_are_not_a_tag_are_kept(self):
        """Text that only starts like a tag, without a tag's attributes, is kept."""
        assert common.clean("a<b and c>d <i>e</i>") == "a<b and c>d e"
        assert common.clean('<i class="abcd">e</i>') == "e"

    def test_a_superscript_follows_a_caret(self):
        """An exponent stays apart from its base."""
        assert common.clean("10<sup>9</sup>/L, kg/m<sup>2</sup>") == "10^9/L, kg/m^2"

    def test_entities_with_digits_in_their_names(self):
        """Entities such as &sup2; and &frac12;, and upper-case hex ones, are unescaped."""
        assert common.clean("kg/m&sup2;, &frac12; cup, &#X41;") == (
            "kg/m\u00b2, \u00bd cup, A"
        )

    def test_a_lone_cp1252_byte_is_its_character(self):
        """A cp1252 byte read as Latin-1 becomes its character, not nothing."""
        assert common.clean("Age 18\x9665, don\x92t") == "Age 18\u201365, don\u2019t"
        assert common.clean("Age 18\ufffd65") == "Age 18 65"

    def test_escaped_characters_are_read(self):
        """R's <U+00A0> and JSON-style escapes are read as their characters."""
        assert common.clean("Abcd?<U+00A0>") == "Abcd?"
        assert common.clean("Abcd" + "\\" + "u2122 480") == "Abcd\u2122 480"

    def test_a_cp1252_sign_on_the_way_is_repaired_too(self):
        """A text read wrongly twice through cp1252's own "˜" is repaired."""
        twice = "\u2018abcd".encode("utf-8").decode("cp1252")
        twice = twice.encode("utf-8").decode("cp1252")
        assert common.clean(twice) == "\u2018abcd"

    def test_a_lost_character_read_as_mac_roman_is_a_space(self):
        """The replacement character read as Mac Roman is a space."""
        assert common.clean("abcd\u00d4\u00f8\u03a9efgh") == "abcd efgh"

    def test_an_escape_beyond_unicode_is_kept(self):
        """An escape that names no character stays as it is."""
        assert common.clean("Abcd <U+110000>") == "Abcd <U+110000>"

    def test_characters_that_hold_no_text_are_removed(self):
        """Byte order marks, zero-width spaces, soft hyphens and direction marks go."""
        assert common.clean("\ufeffAB\u200bCD\u00ad") == "ABCD"
        assert common.clean("\u202aAbcd\u202c efgh\u2066") == "Abcd efgh"

    def test_escaped_markup_and_more_tags(self):
        """Escaped tags are removed as tags; headings and table parts are tags too."""
        assert common.clean("&lt;b&gt;Abcd&lt;/b&gt;") == "Abcd"
        assert common.clean("<p>Abcd</p><hr><h3>Efgh</h3>") == "Abcd Efgh"
        assert common.clean("<table><thead><tr><th>Abcd</th></tr></thead></table>") == (
            "Abcd"
        )

    def test_unicode_is_normalised(self):
        """A letter and its combining accent become one character."""
        assert common.clean("Abcde\u0301") == "Abcd\u00e9"

    def test_missing_is_empty(self):
        """A missing text cleans to an empty one."""
        assert common.clean(None) == ""
        assert common.clean(float("nan")) == ""

    def test_a_key_is_the_text_as_the_tokenizer_normalises_it(self):
        """Case, accents and punctuation do not change a key; other letters stay."""
        assert common.key("Ábcd_Éfgh (IJKL)") == "abcd efgh ijkl"
        assert common.key("Größe") == common.key("GROSSE") == "grosse"
        assert common.key("α-Abcd") == "α abcd"


class TestNamesAndTypes:
    """What counts as a name, its term group, and the type words."""

    def test_a_name_has_two_letters_in_a_row(self):
        """Ids and numbers are not names."""
        assert common.is_name("Abcd")
        assert common.is_name("AB")
        assert not common.is_name("2223612v3.0:2178533v1.0")
        assert not common.is_name("535")

    def test_term_group_comes_from_the_shape(self):
        """Codes and short forms are expansions, and words are paraphrases.

        Words joined by "-" are a paraphrase when one of them is long.
        """
        expansions = [
            "ABCD",
            "ABCD_EFG_TXT",
            "AB12",
            "aBcD",
            "abcd1",
            "AB-12C",
            "Abc-Def",
        ]
        for text in expansions:
            assert common.term_group(text) == common.EXPANSION
        paraphrases = [
            "abcd",
            "Abcd",
            "Abcd efgh",
            "ABCD EFGH",
            "Pre-Abcdef",
            "A-Bcdefg",
        ]
        for text in paraphrases:
            assert common.term_group(text) == common.PARAPHRASE

    def test_a_name_with_a_number_is_words(self):
        """A long word joined to a number is a name, not a short form."""
        assert common.term_group("Abcdefg-10") == common.PARAPHRASE
        assert common.term_group("abcdefg-10") == common.PARAPHRASE
        assert common.term_group("AB-10") == common.EXPANSION

    def test_codes_go_when_every_value_has_one(self):
        """A list whose values all carry a code loses the codes, in every style."""
        for coded, labels in (
            (["0= None", "1= Abcd"], ["None", "Abcd"]),
            (["(1) = Abcd", "(2) = Efgh"], ["Abcd", "Efgh"]),
            (["0 - abcd", "1 - efgh"], ["abcd", "efgh"]),
            (["999: Abcd", "1: Efgh"], ["Abcd", "Efgh"]),
            (["(1) All abcd", "(2) Some abcd"], ["All abcd", "Some abcd"]),
            (["1. Abcd", "2. Efgh"], ["Abcd", "Efgh"]),
        ):
            assert common.value_labels(coded) == labels

    def test_a_scale_with_bare_numbers_keeps_its_codes(self):
        """When some values are bare numbers, the numbers are the values."""
        scale = ["0 - No abcd", "1", "2", "10 - Worst abcd"]
        assert common.value_labels(scale) == scale
        ranges = ["2 - 3 times", "4 - 5 times"]
        assert common.value_labels(ranges) == ranges

    def test_a_dashed_code_needs_three_coded_values(self):
        """A dash code needs three values or more, each with its own code."""
        assert common.value_labels(["0-Abcd", "1-Efgh", "2-Ijkl"]) == [
            "Abcd",
            "Efgh",
            "Ijkl",
        ]
        for names in (
            ["2-Abcdef Mno", "3-Abcdef Pqr"],
            ["3-Abcdef", "2-Abcdef", "1-Abcdef"],
            ["2-Abcdef Mno", "3-Abcdef Pqr", "3-Abcdef Stu"],
        ):
            assert common.value_labels(names) == names

    def test_a_question_is_a_paraphrase(self):
        """A question's wording is a paraphrase, whatever its shape."""
        assert common.term_group("ABCD?", question=True) == common.PARAPHRASE

    def test_type_words(self):
        """The source's own word, else dbGaP's words for an item with and without values."""
        assert common.type_word("Value List", ["a"]) == "Value List"
        assert common.type_word("", ["a"]) == "string/encoded"
        assert common.type_word("", []) == "string"

    def test_texts_of_one_key_keep_every_study(self):
        """Spellings of one text are one anchor, with the studies of all of them."""
        item = Item(
            id="abcd1",
            canonical="Abcd",
            texts=[
                Text("HEIGHT", "dbGaP variable name", study="phs999999"),
                Text("height", "dbGaP variable name", study="phs999997|phs999998"),
            ],
        )
        found, _ = common.star([item], source="abcd")
        assert found.study.tolist() == ["phs999997|phs999998|phs999999"]

    def test_pairs_that_the_tokenizer_cannot_tell_apart_are_kept_once(self):
        """Pairs that differ only in case or accents are copies; the first is kept."""
        found = pd.DataFrame(
            {
                "anchor": ["ABCD", "abcd", "abcd_x"],
                "positive": ["Abcd Éfgh", "abcd efgh", "abcd efgh"],
                "study": ["phs999902", "phs999901", ""],
            }
        )
        kept = common.distinct_pairs(found)
        assert kept.anchor.tolist() == ["ABCD", "abcd_x"]
        assert kept.study.tolist() == ["phs999901|phs999902", ""]

    def test_a_copy_in_an_earlier_source_is_found_once_normalised(self):
        """A later source's pair is not new when only its case or accents differ."""
        seen = set()
        earlier = pd.DataFrame({"anchor": ["Abcd"], "positive": ["Éfgh"]})
        later = pd.DataFrame({"anchor": ["ABCD", "Ijkl"], "positive": ["efgh", "Mnop"]})
        assert pairs.new_pairs(earlier, seen).tolist() == [True]
        assert pairs.new_pairs(later, seen).tolist() == [False, True]


class TestStar:
    """Pairs around each item's canonical, the same for every source."""

    def test_each_text_twice_and_each_other_description_once(self):
        """Each text gives two pairs, and each other description gives one pair."""
        item = Item(
            id="abcd1",
            canonical="Abcd efgh",
            description="An abcd.",
            descriptions=["Another abcd.", "an abcd"],
            texts=[Text("AE", "AB"), Text("Efgh abcd", "SY")],
        )
        assert made([item]) == [
            ("name", "AE", "Abcd efgh"),
            ("name_description", "AE", "Abcd efgh"),
            ("name", "Efgh abcd", "Abcd efgh"),
            ("name_description", "Efgh abcd", "Abcd efgh"),
            ("description_description", "Another abcd.", "Abcd efgh"),
        ]

    def test_texts_once_without_the_canonical_or_ids(self):
        """A text equal to the canonical, a repeated text and a text with no name are skipped."""
        item = Item(
            id="abcd1",
            canonical="Abcd efgh",
            texts=[
                Text("abcd-efgh", "SY"),
                Text("AE", "AB"),
                Text("ae", "SY"),
                Text("1234", "SY"),
            ],
        )
        assert made([item]) == [("name", "AE", "Abcd efgh")]

    def test_a_lone_canonical_is_paired_with_its_description(self):
        """An item with no other name pairs its canonical with its description alone."""
        found, _ = common.star(
            [Item(id="abcd1", canonical="Abcd", description="An abcd.", type="Number")],
            source="abcd",
        )
        row = found.iloc[0]
        assert (row.kind, row.anchor, row.positive) == (
            "name_to_description",
            "Abcd (Number)",
            "An abcd. (Number)",
        )

    def test_pairs_once_per_value_list(self):
        """An item with two lists has its pairs twice, each list named, values in their order."""
        item = Item(
            id="abcd1",
            canonical="Abcd",
            lists=[("Has_Abcd_Value", ["b", "a"]), ("Has_Efgh_Value", ["c"])],
            texts=[Text("AB", "AB")],
        )
        found, _ = common.star([item], source="abcd")
        assert found.value_list.tolist() == ["Has_Abcd_Value", "Has_Efgh_Value"]
        assert found.positive.tolist() == [
            "Abcd (string/encoded) Values: b | a",
            "Abcd (string/encoded) Values: c",
        ]
        assert found.anchor.tolist() == ["AB (string)", "AB (string)"]

    def test_each_side_keeps_its_fields(self):
        """Each side's name, type, description and values are kept beside its text."""
        item = Item(
            id="abcd1",
            canonical="Abcd",
            type="Value List",
            description="An abcd.",
            lists=[("", ["Yes", "No"])],
            texts=[
                Text("Is it abcd?", "Question Text", question=True, study="phs999999")
            ],
        )
        row = common.star([item], source="abcd")[0].iloc[1]
        assert row[common.COLUMNS[2:]].to_dict() == {
            "source": "abcd",
            "kind": "name_description",
            "item": "abcd1",
            "term_type": "Question Text",
            "term_group": "paraphrase",
            "status": "current",
            "study": "phs999999",
            "value_list": "",
            "anchor_name": "Is it abcd?",
            "anchor_type": "Value List",
            "anchor_description": "",
            "anchor_values": [],
            "positive_name": "Abcd",
            "positive_type": "Value List",
            "positive_description": "An abcd.",
            "positive_values": ["Yes", "No"],
        }

    def test_a_retired_item_gives_its_names_to_its_replacement(self):
        """The retired item's canonical and texts point to the replacement, through a chain."""
        retired = Item(
            id="abcd1",
            canonical="Old abcd",
            texts=[Text("OA", "AB")],
            status=RETIRED,
            replaced_by=["abcd2"],
        )
        between = Item(
            id="abcd2", canonical="Older", status=RETIRED, replaced_by=["abcd3"]
        )
        current = Item(id="abcd3", canonical="New abcd", description="An abcd.")
        found, _ = common.star([retired, between, current], source="abcd")
        old = found[found.anchor_name.isin(["Old abcd", "OA"])]
        assert set(old.positive_name) == {"New abcd"}
        assert set(old.status) == {RETIRED}
        assert old.term_type.tolist() == ["canonical"] * 2 + ["AB"] * 2

    def test_a_retired_item_without_one_keeps_only_new_names(self):
        """Without a replacement, only names that no current item has are used."""
        current = Item(id="abcd1", canonical="Abcd", texts=[Text("AB", "AB")])
        retired = Item(
            id="abcd2",
            canonical="Abcd",
            description="An old abcd.",
            texts=[Text("ab", "AB"), Text("ABC", "AB")],
            status=RETIRED,
        )
        assert made([current, retired]) == [
            ("name", "AB", "Abcd"),
            ("name", "ABC", "Abcd"),
            ("name_description", "ABC", "Abcd"),
        ]

    def test_a_retired_canonical_a_current_item_has_is_not_paired_alone(self):
        """A retired canonical that a current item has gives no name_to_description pair."""
        current = Item(id="abcd1", canonical="Abcd")
        retired = Item(
            id="abcd2", canonical="Abcd", description="An old abcd.", status=RETIRED
        )
        assert made([current, retired]) == []

    def test_an_empty_description_gives_way_to_the_next(self):
        """When the first description is empty, the next one is the description."""
        found = common.cleaned(
            Item(
                id="abcd1", canonical="Abcd", description=" ", descriptions=["An abcd."]
            )
        )
        assert (found.description, found.descriptions) == ("An abcd.", [])

    def test_lists_equal_once_cleaned_are_kept_once(self):
        """Lists that clean to the same values count once; a list of nothing goes."""
        found = common.cleaned(
            Item(
                id="abcd1",
                canonical="Abcd",
                lists=[
                    ("Has_A_Value", ["A&amp;B"]),
                    ("Has_B_Value", ["A&B"]),
                    ("C", [" "]),
                ],
            )
        )
        assert found.lists == [("Has_A_Value", ["A&B"])]

    def test_two_lists_of_one_name_are_both_kept(self):
        """Lists without a name are told apart by their values."""
        found = common.cleaned(
            Item(id="abcd1", canonical="Abcd", lists=[("", ["Yes"]), ("", ["Low"])])
        )
        assert found.lists == [("", ["Yes"]), ("", ["Low"])]

    def test_a_chain_passes_through_a_closed_retired_item(self):
        """A closed retired item in the middle of a chain does not stop it."""
        items = [
            Item(id="abcd1", canonical="Old", status=RETIRED, replaced_by=["abcd2"]),
            Item(
                id="abcd2",
                canonical="Older",
                status=RETIRED,
                replaced_by=["abcd3"],
                open=False,
            ),
            Item(id="abcd3", canonical="New"),
        ]
        assert made(items) == [("name", "Old", "New")]

    def test_a_draft_can_replace_and_a_closed_item_keeps_its_names(self):
        """A draft replacement is used; a closed current item's names are not reused."""
        items = [
            Item(id="abcd1", canonical="Old", status=RETIRED, replaced_by=["abcd2"]),
            Item(id="abcd2", canonical="New", status=DRAFT),
            Item(id="abcd3", canonical="Efgh", texts=[Text("EF", "AB")], open=False),
            Item(
                id="abcd4", canonical="Ijkl", texts=[Text("EF", "AB")], status=RETIRED
            ),
        ]
        assert made(items) == [("name", "Old", "New")]

    def test_a_duplicate_keeps_the_current_copy(self):
        """A pair from both a retired and a current item keeps the current status."""
        items = [
            Item(
                id="abcd1",
                canonical="Old abcd",
                texts=[Text("AB", "SY")],
                status=RETIRED,
                replaced_by=["abcd2"],
            ),
            Item(id="abcd2", canonical="Abcd", texts=[Text("AB", "SY")]),
        ]
        found, _ = common.star(items, source="abcd")
        assert found[found.anchor_name == "AB"].status.tolist() == [CURRENT]

    def test_a_duplicate_keeps_the_studies_of_every_copy(self):
        """A pair from two items keeps one copy, with the studies of both."""
        items = [
            Item(
                id=f"abcd{number}",
                canonical="Abcd",
                texts=[Text("AB", "dbGaP variable name", study=study)],
            )
            for number, study in ((1, "phs999998"), (2, "phs999997"))
        ]
        found, counts = common.star(items, source="abcd")
        assert found.study.tolist() == ["phs999997|phs999998"]
        assert counts["duplicates_removed"] == 1

    def test_a_text_equal_to_the_canonical_is_no_description(self):
        """A description that repeats the canonical gives no pair of its own."""
        item = Item(
            id="abcd1",
            canonical="Abcd efgh",
            description="Abcd efgh.",
            descriptions=["abcd-efgh"],
        )
        assert made([item]) == []

    def test_a_description_without_a_name_in_it_is_not_paired(self):
        """Another description such as "-" or "N/A" gives no pair."""
        item = Item(
            id="abcd1",
            canonical="Abcd",
            description="An abcd.",
            descriptions=["-", "N/A"],
        )
        assert made([item]) == [("name_to_description", "Abcd", "An abcd.")]

    def test_a_question_loses_its_item_number_and_a_name_keeps_it(self):
        """A form's item number goes from a question text, not from other texts."""
        item = Item(
            id="abcd1",
            canonical="Abcd",
            texts=[
                Text("24. Is it abcd?", "Question Text", question=True),
                Text("(5a) How abcd?", "Question Text", question=True),
                Text("13f-2. Abcd weak?", "Question Text", question=True),
                Text("1. e During the past abcd", "Question Text", question=True),
                Text("W2 - Unhappy with abcd", "Question Text", question=True),
                Text("C. What abcd?", "Question Text", question=True),
                Text("H. Abcdef result", "Question Text", question=True),
                Text("2.5 mg of abcd?", "Question Text", question=True),
                Text("F361. Abcd medicine", "dbGaP variable description"),
            ],
        )
        assert [anchor for _, anchor, _ in made([item])] == [
            "Is it abcd?",
            "How abcd?",
            "Abcd weak?",
            "During the past abcd",
            "Unhappy with abcd",
            "What abcd?",
            "H. Abcdef result",
            "2.5 mg of abcd?",
            "F361. Abcd medicine",
        ]

    def test_a_text_that_differs_only_by_an_item_number_is_the_canonical(self):
        """A text equal to the canonical but for a form's item number is skipped."""
        item = Item(
            id="abcd1", canonical="Is it abcd?", texts=[Text("Q2. Is it abcd?", "SY")]
        )
        assert made([item]) == []

    def test_empty_template_headings_go_from_a_description(self):
        """A template's headings with nothing after them leave a description."""
        item = common.cleaned(
            Item(
                id="abcd1",
                canonical="Abcd",
                description="An abcd. EXAMPLE(S): efgh OTHER NAME(S): NOTE(S):",
                descriptions=["An efgh. EXAMPLE(S): OTHER NAME(S): NOTE(S): Ijkl."],
            )
        )
        assert (item.description, item.descriptions) == (
            "An abcd. EXAMPLE(S): efgh",
            ["An efgh. NOTE(S): Ijkl."],
        )

    def test_a_draft_is_used_and_marked(self):
        """A draft item is used like a current item, and it keeps its status."""
        found, _ = common.star(
            [
                Item(
                    id="abcd1", canonical="Abcd", texts=[Text("AB", "AB")], status=DRAFT
                )
            ],
            source="abcd",
        )
        assert found.status.tolist() == [DRAFT]

    def test_closed_items_and_duplicates_are_left_out_and_counted(self):
        """An item that is not open gives no pairs; a repeated pair is kept once."""
        items = [
            Item(id="abcd1", canonical="Abcd", texts=[Text("AB", "AB")]),
            Item(id="abcd2", canonical="Abcd", texts=[Text("AB", "SY")]),
            Item(id="abcd3", canonical="Efgh", texts=[Text("EF", "AB")], open=False),
        ]
        found, counts = common.star(items, source="abcd")
        assert found.item.tolist() == ["abcd1"]
        assert counts == {"items": 3, "items_not_open": 1, "duplicates_removed": 1}


class TestNcit:
    """NCIt's concepts."""

    TABLES = {
        "concepts": table(
            code=["C1", "C2", "C3"],
            preferred_name=["Abcd", "Efgh", "Old abcd"],
            status=["", "Provisional_Concept", "Retired_Concept"],
            open=[True, True, True],
        ),
        "synonyms": table(
            code=["C1", "C1", "C1", "C2", "C2"],
            name=["AB", "Abcd grammar", "Abcd syn", "EF", "EFG"],
            term_type=["AB", "AD", "SY", "PT", "CA3"],
        ),
        "definitions": table(
            code=["C1", "C1", "C1"],
            definition=["An abcd (CDISC)", "An abcd. (NCI)", "The abcd. (NCI04)"],
            kind=["ALT_DEFINITION", "DEFINITION", "DEFINITION"],
            source=["CDISC", "NCI", "NCI-GLOSS"],
        ),
        "values": table(
            code=["C1", "C1", "C1", "C1"],
            association=["Has_B_Value", "Has_B_Value", "Has_A_Value", "Has_A_Value"],
            value_code=["C9", "C8", "C8", "C9"],
            value=["Zeta", "Alpha", "Alpha", "Zeta"],
        ),
        "replacements": table(
            code=["C3"],
            action=["merge"],
            date=["2026-01-01"],
            replacement_code=["C1"],
            replacement_name=["Abcd"],
        ),
    }

    def test_concepts(self):
        """Same-meaning synonyms; the NCI definition first, sources off; lists once, sorted."""
        first, second, third = ncit.items(self.TABLES)
        assert [text.text for text in first.texts] == ["AB", "Abcd syn"]
        assert first.description == "An abcd."
        assert first.descriptions == ["The abcd.", "An abcd"]
        assert first.lists == [("Has_A_Value", ["Alpha", "Zeta"])]
        assert [text.text for text in second.texts] == ["EF", "EFG"]
        assert (second.status, third.status) == (DRAFT, RETIRED)
        assert third.replaced_by == ["C1"]

    def test_a_source_inside_an_open_parenthesis_is_kept(self):
        """A source that closes a parenthesis opened before it is part of the text."""
        text = "Abcd (for the National Abcd Institute (NCI)"
        assert ncit.without_source(text, {"NCI"}) == text
        assert ncit.without_source("1) Abcd 2) Efgh. (NCI)", {"NCI"}) == (
            "1) Abcd 2) Efgh."
        )

    def test_a_new_term_type_is_reported(self, capsys):
        """A term type in neither list is not used, and is reported."""
        tables = dict(
            self.TABLES, synonyms=self.TABLES["synonyms"].assign(term_type="ZZ")
        )
        assert ncit.items(tables)[0].texts == []
        assert "'ZZ'" in capsys.readouterr().out


class TestObo:
    """The ontologies' terms."""

    def test_terms(self):
        """EXACT synonyms with their types; obsolete terms with their replacements."""
        tables = {
            "terms": table(
                id=["ABCD:1", "ABCD:2"],
                ontology=["abcd", "abcd"],
                label=["abcd", "old abcd (obsolete)"],
                obsolete=[False, True],
                replaced_by=["", "ABCD:1|ABCD:3"],
                consider=["", ""],
                open=[True, True],
            ),
            "synonyms": table(
                id=["ABCD:1", "ABCD:1", "ABCD:1"],
                ontology=["abcd"] * 3,
                name=["AB", "abcd-like", "abcd thing"],
                scope=["EXACT", "RELATED", "EXACT"],
                synonym_types=["abbreviation", "", ""],
            ),
            "definitions": table(
                id=["ABCD:1"], ontology=["abcd"], definition=["An abcd."]
            ),
        }
        first, second = obo.items(tables)
        assert [(t.text, t.term_type) for t in first.texts] == [
            ("AB", "abbreviation"),
            ("abcd thing", "EXACT"),
        ]
        assert first.description == "An abcd."
        assert (second.status, second.replaced_by) == (RETIRED, ["ABCD:1", "ABCD:3"])
        assert second.canonical == "old abcd"

    def test_a_second_definition_is_kept(self):
        """A term with two definitions has the second as another description."""
        tables = {
            "terms": table(
                id=["ABCD:1"],
                ontology=["abcd"],
                label=["abcd"],
                obsolete=[False],
                replaced_by=[""],
                open=[True],
            ),
            "synonyms": table(id=[], name=[], scope=[], synonym_types=[]),
            "definitions": table(
                id=["ABCD:1", "ABCD:1"], definition=["An abcd.", "The abcd."]
            ),
        }
        (term,) = obo.items(tables)
        assert (term.description, term.descriptions) == ("An abcd.", ["The abcd."])

    def test_an_obsolete_terms_texts_lose_their_marker(self):
        """An obsolete term's synonyms and definition lose the word "obsolete"."""
        tables = {
            "terms": table(
                id=["ABCD:1"],
                ontology=["abcd"],
                label=["abcd"],
                obsolete=[True],
                replaced_by=[""],
                open=[True],
            ),
            "synonyms": table(
                id=["ABCD:1"],
                name=["obsolete efgh"],
                scope=["EXACT"],
                synonym_types=[""],
            ),
            "definitions": table(id=["ABCD:1"], definition=["OBSOLETE. An abcd."]),
        }
        (term,) = obo.items(tables)
        assert ([t.text for t in term.texts], term.description) == (
            ["efgh"],
            "An abcd.",
        )


class TestLoinc:
    """LOINC's codes."""

    def tables(self):
        """A code with two names, a question, two form rows and two answer lists."""
        return {
            "codes": table(
                LOINC_NUM=["9999-1", "9999-2"],
                STATUS=["ACTIVE", "DEPRECATED"],
                LONG_COMMON_NAME=["Abcd [Efgh] in Ijkl", "Deprecated deprecated abcd"],
                SHORTNAME=["Abcd Ij-kLmn", "DeprecatedAbcd Ij"],
                DisplayName=["Abcd (Ij)", ""],
                SURVEY_QUEST_TEXT=["How abcd?", ""],
                DefinitionDescription=["An abcd.", ""],
                open=[True, True],
            ),
            "forms": table(
                Loinc=["9999-1", "9999-1"],
                DisplayNameForForm=["Abcd today?", "Abcd closed?"],
                open=[True, False],
            ),
            "answers": table(
                AnswerListId=["LL9999-1", "LL9999-1", "LL9999-2"],
                SequenceNumber=["10", "2", "1"],
                DisplayText=["Ten", "Two", "Panel only"],
            ),
            "answer_links": table(
                LoincNumber=["9999-1", "9999-1"],
                AnswerListId=["LL9999-1", "LL9999-2"],
                AnswerListLinkType=["EXAMPLE", "NORMATIVE"],
                ApplicableContext=["", "9999-9"],
            ),
            "map_to": table(LOINC=["9999-2"], MAP_TO=["9999-1"]),
        }

    def test_codes(self):
        """Names and questions, open form rows only, and the default list in its order."""
        first, second = loinc.items(self.tables())
        assert [(t.text, t.question) for t in first.texts] == [
            ("Abcd Ij-kLmn", False),
            ("Abcd (Ij)", False),
            ("How abcd?", True),
            ("Abcd today?", True),
        ]
        assert first.lists == [("LL9999-1 (EXAMPLE)", ["Two", "Ten"])]
        assert (second.status, second.replaced_by) == (RETIRED, ["9999-1"])
        assert (second.canonical, second.texts[0].text) == ("abcd", "Abcd Ij")


class TestCdisc:
    """CDISC's codelists and terms."""

    def test_codelists_and_terms(self):
        """A term's rows are merged; a codelist lists its terms; a question goes to the source side."""
        tables = {
            "terms": table(
                **{
                    "Code": ["C9", "C1", "C1", "C2"],
                    "Codelist Code": ["", "C9", "C8", "C9"],
                    "Codelist Name": [
                        "Abcd Test Name",
                        "Abcd Test Name",
                        "Abcd Code",
                        "x",
                    ],
                    "CDISC Submission Value": ["ABCDTN", "ABCD1", "Abcd one", "ABCD2"],
                    "CDISC Synonym(s)": ["Abcd Test Name", "Abcd one; A1", "", ""],
                    "CDISC Definition": [
                        "A list of abcd tests.",
                        "Abcd - How abcd are you? (i.e., how much abcd)",
                        "Abcd - How abcd are you? (i.e., how much abcd)",
                        "The second abcd. (Abcd Glossary)",
                    ],
                    "NCI Preferred Term": [
                        "CDISC Abcd Test Name Terminology",
                        "Abcd One",
                        "Abcd One",
                        "Abcd Two",
                    ],
                    "open": [True] * 4,
                }
            )
        }
        codelist, first, second = cdisc.items(tables)
        assert (codelist.canonical, codelist.lists) == (
            "Abcd Test Name",
            [("", ["Abcd One", "Abcd Two"])],
        )
        assert [t.text for t in codelist.texts] == ["ABCDTN", "Abcd Test Name"]
        assert [(t.text, t.question) for t in first.texts] == [
            ("ABCD1", False),
            ("Abcd one", False),
            ("Abcd one", False),
            ("A1", False),
            ("Abcd - How abcd are you? (i.e., how much abcd)", True),
        ]
        assert (first.canonical, first.description) == ("Abcd One", "")
        assert second.description == "The second abcd."

    def test_a_question_may_end_with_a_note_and_a_full_stop(self):
        """A question followed by a note in parentheses and a full stop is a question."""
        assert cdisc.is_question("Abcd - Have you abcd? (Lifetime).")
        assert not cdisc.is_question("Abcd - An abcd (Lifetime).")

    def test_a_long_note_after_the_last_sentence_is_content(self):
        """Only a short source note after the last sentence is removed."""
        note = (
            "A middle abcd. (With an even number of abcds, it is halfway between two.)"
        )
        assert cdisc.without_source(note) == note
        assert (
            cdisc.without_source("A middle abcd. (Abcd Glossary)") == "A middle abcd."
        )

    def test_a_source_after_a_reference_or_with_parentheses_is_removed(self):
        """A source after a bracketed reference, or holding parentheses, is removed."""
        assert cdisc.without_source("An abcd. [After Efgh] (Abcd Glossary)") == (
            "An abcd. [After Efgh]"
        )
        assert cdisc.without_source("An abcd. (After 21 CFR 210.3(b)(8))") == "An abcd."


class TestCde:
    """The NIH CDE Repository's data elements."""

    def test_elements(self, capsys):
        """The first designation is the canonical; a question tag or mark makes a question."""
        tables = {
            "elements": table(
                tiny_id=["abcd1", "abcd2"],
                steward=["NCI", "Abcd"],
                registration_status=["Qualified", "Abcd status"],
                datatype=["Value List", "Text"],
                open=[True, True],
            ),
            "designations": table(
                tiny_id=["abcd1", "abcd1", "abcd1", "abcd1", "abcd2"],
                position=[1, 0, 2, 3, 0],
                designation=[
                    "Is it abcd?",
                    "Abcd indicator",
                    "Abcd ind",
                    "7. Abcd today?",
                    "Deprecated",
                ],
                tags=["Question text", "", "Short Name", "", ""],
            ),
            "definitions": table(
                tiny_id=["abcd1", "abcd1"],
                position=[1, 0],
                definition=["Another abcd.", "An abcd._The efgh."],
                tags=["", ""],
            ),
            "values": table(
                tiny_id=["abcd1", "abcd1"],
                position=[0, 1],
                permissible_value=["1", "2"],
                value_meaning_name=["Yes", ""],
            ),
        }
        first, second = cde.items(tables)
        assert first.canonical == "Abcd indicator"
        assert [(t.text, t.question) for t in first.texts] == [
            ("Is it abcd?", True),
            ("Abcd ind", False),
            ("7. Abcd today?", True),
        ]
        assert (first.description, first.descriptions) == (
            "An abcd. The efgh.",
            ["Another abcd."],
        )
        assert (first.type, first.lists) == ("Value List", [("", ["Yes", "2"])])
        assert (second.status, second.canonical, second.texts) == (CURRENT, "", [])
        assert "'Abcd status'" in capsys.readouterr().out


class TestCadsr:
    """caDSR's data elements."""

    def test_elements(self):
        """The short name, name types and every question type but caDSR's placeholder."""
        tables = {
            "elements": table(
                public_id=["9999901"],
                long_name=["Abcd Category Text"],
                short_name=["ABCD_CAT_TXT"],
                definition=["An abcd._The efgh. [Manually-curated]."],
                workflow_status=["DRAFT NEW"],
                datatype=["CHARACTER"],
                open=[True],
            ),
            "alternate_names": table(
                public_id=["9999901", "9999901"],
                name=["abcd_category", "Abcd Context"],
                type=["GDC Alt Name", "USED_BY"],
            ),
            "documents": table(
                public_id=["9999901"] * 4,
                type=[
                    "Preferred Question Text",
                    "REFERENCE",
                    "Application Standard Question Text",
                    "Preferred Question Text",
                ],
                text=[
                    "Abcd?",
                    "Abcd et al.",
                    "Abcd category:",
                    "Data Element Abcd Category Text does not have Preferred "
                    "Question Text",
                ],
            ),
            "values": table(
                public_id=["9999901", "9999901"],
                position=[1, 0],
                value=["B", "A"],
                meaning=["", "Abcd"],
            ),
        }
        (element,) = cadsr.items(tables)
        assert [(t.text, t.question) for t in element.texts] == [
            ("ABCD_CAT_TXT", False),
            ("abcd_category", False),
            ("Abcd?", True),
            ("Abcd category:", True),
        ]
        assert element.description == "An abcd. The efgh."
        assert (element.status, element.type, element.lists) == (
            DRAFT,
            "CHARACTER",
            [("", ["Abcd", "B"])],
        )

    def test_java_types_go(self):
        """A long name loses its Java type, and a definition the phrase about it."""
        tables = {
            "elements": table(
                public_id=["9999902"],
                long_name=["Abcd Name java.lang.String"],
                short_name=["ABCD_NM"],
                definition=["An abcd._Value Domain for java language String datatype."],
                workflow_status=["RETIRED ARCHIVED"],
                datatype=["CHARACTER"],
                open=[True],
            ),
            "alternate_names": table(
                public_id=["9999902"], name=["Abcd Context"], type=["USED_BY"]
            ),
            "documents": table(
                public_id=["9999902"], type=["REFERENCE"], text=["Abcd et al."]
            ),
            "values": table(
                public_id=["9999902"], position=[0], value=["A"], meaning=[""]
            ),
        }
        (element,) = cadsr.items(tables)
        assert (element.canonical, common.clean(element.description)) == (
            "Abcd Name",
            "An abcd.",
        )

    def test_java_phrases_go_and_the_text_around_them_stays(self):
        """Each kind of Java phrase goes, even when no full stop comes before it."""
        assert cadsr.without_java("Abcd Date java.util.Date") == "Abcd Date"
        for definition in [
            "An abcd. Java Date Type",
            "An abcd. Generic value domain for a primitive java dataype that is an "
            "Integer (32-bit two's complement).",
        ]:
            assert cadsr.without_java(definition) == "An abcd."
        assert cadsr.without_java(
            "An abcd Generic value domain for a java datatype that is a class. An efgh."
        ) == ("An abcd An efgh.")

    def test_joined_definitions_are_spaced(self):
        """caDSR's joined concept definitions get a space; its curation mark goes."""
        assert common.cadsr_definition("An abcd._The efgh.:An ijkl. -A mnop.") == (
            "An abcd. The efgh. An ijkl. A mnop."
        )
        assert (
            common.cadsr_definition("An abcd._the efgh_Ijkl.")
            == "An abcd. the efgh Ijkl."
        )
        for label in ["An abcd. EXAMPLE(S): Efgh", "An abcd. Example(s): efgh"]:
            assert common.cadsr_definition(label) == label
        assert common.cadsr_definition("An abcd. [Manually-curated]") == "An abcd."


class TestNhanes:
    """NHANES' variables."""

    def test_variables(self):
        """One item per data file, the English Text first, and only value labels as values."""
        key = {"cycle": ["2017", "2019"], "data_file": ["ABCD_J", "ABCD_K"]}
        tables = {
            "variables": table(
                **key,
                name=["ABCDXYZ", "ABCDXYZ"],
                description=["Abcd score", "Abcd score"],
                open=[True, True],
            ),
            "codebook": table(
                cycle=["2017"],
                data_file=["ABCD_J"],
                name=["ABCDXYZ"],
                sas_label=["Abcd score"],
                english_text=["How abcd are you?"],
                open=[True],
            ),
            "values": table(
                cycle=["2017"] * 3,
                data_file=["ABCD_J"] * 3,
                name=["ABCDXYZ"] * 3,
                position=[0, 1, 2],
                code=["1", "2 to 9", "."],
                label=["Very", "Range of Values", "Missing"],
            ),
        }
        first, second = nhanes.items(tables)
        assert (first.id, first.canonical, first.lists) == (
            "2017/ABCD_J/ABCDXYZ",
            "How abcd are you?",
            [("", ["Very"])],
        )
        assert [t.text for t in first.texts] == ["ABCDXYZ", "Abcd score", "Abcd score"]
        assert (second.canonical, second.lists) == ("Abcd score", [])


class TestPhenx:
    """PhenX's variables and their dbGaP mappings."""

    def test_variables(self):
        """Visible variables only, closed ones marked; comparable mappings with their studies."""
        tables = {
            "variables": table(
                id=["1", "2", "3"],
                name=["PX999901_Abcd", "PX999902_Old", "PX999903_Closed"],
                description=["How abcd?", "Old abcd?", "Closed abcd?"],
                type=["encoded values", "integer", "integer"],
                variable_values=["1 = Yes|2 = No|0..9", "", ""],
                protocol_visible=["1", "0", "1"],
                open=[True, True, False],
            ),
            "mappings": table(
                variable_id=["1", "1", "1"],
                gap_study_accession=["phs999999", "phs999998", "phs999997"],
                gap_variable_name=["ABCD", "ABCD", "EFGH"],
                gap_variable_desc=["Abcd answer", "Abcd answer", "Efgh answer"],
                map_level=["comparable", "comparable", "related"],
            ),
        }
        variable, closed = phenx.items(tables)
        assert (variable.open, closed.open) == (True, False)
        assert (variable.canonical, variable.type, variable.lists) == (
            "How abcd?",
            "encoded values",
            [("", ["Yes", "No"])],
        )
        assert [(t.text, t.study) for t in variable.texts] == [
            ("PX999901_Abcd", ""),
            ("ABCD", "phs999998|phs999999"),
            ("Abcd answer", "phs999998|phs999999"),
        ]

    def test_quotes_around_a_description_go(self):
        """A description wrapped in quotes loses them; quotes inside it stay."""
        assert phenx.unquoted('"How abcd?"') == "How abcd?"
        assert phenx.unquoted('"Abcd" or "efgh"?') == '"Abcd" or "efgh"?'

    def test_value_labels(self):
        """Codes are taken off; ranges list no values; a tab also separates entries."""
        assert phenx.labels("(-3) = Abcd|? = Efgh\vIjkl|0..110") == [
            "Abcd",
            "Efgh",
            "Ijkl",
        ]
        assert phenx.labels("Abcd = 0|Efgh ijkl = 1") == ["Abcd", "Efgh ijkl"]
        assert phenx.labels("1 = Abcd,2 = Efgh, ijkl") == ["Abcd", "Efgh, ijkl"]
        assert phenx.labels("A = 1 mm|= = Abcd") == ["1 mm", "Abcd"]
        assert phenx.labels("Abcd = n/t|001 - 700 = efgh") == ["Abcd", "efgh"]
        assert phenx.labels("99/99/9999 = abcd|>=18 = efgh") == ["abcd", "efgh"]
        assert phenx.labels("Yes = Classify as abcd|NA = Not abcd") == [
            "Yes",
            "Not abcd",
        ]


def fetched_nhanes(inputs, name="nhanes"):
    """Write a fictional NHANES fetch under a source's name, in the fetch layout."""
    directory = inputs / "sources" / name
    directory.mkdir(parents=True)
    tables = {
        "variables": table(
            cycle=["2017"],
            name=["ABCDXYZ"],
            description=["Abcd score"],
            data_file=["ABCD_J"],
            open=[True],
        ),
        "codebook": table(
            cycle=["2017"],
            data_file=["ABCD_J"],
            name=["ABCDXYZ"],
            sas_label=["Abcd score"],
            english_text=["How abcd are you?"],
            open=[True],
        ),
        "values": table(
            cycle=["2017"],
            data_file=["ABCD_J"],
            name=["ABCDXYZ"],
            position=[0],
            code=["1"],
            label=["Very"],
        ),
    }
    sources.write_source(directory, name, "1", tables, [], {})


class TestPrepare:
    """``pairs prepare``: each source's file, its manifest, and combining sources."""

    def test_prepare_writes_the_pairs_and_their_manifest(self, tmp_path, capsys):
        """The pairs are written with a manifest of their fetch, checksum and counts."""
        fetched_nhanes(tmp_path)
        assert (
            main(["pairs", "prepare", "--inputs", str(tmp_path), "--source", "nhanes"])
            == 0
        )
        assert f"-> {tmp_path / 'pairs' / 'nhanes.parquet'}" in capsys.readouterr().out
        found = pd.read_parquet(tmp_path / "pairs" / "nhanes.parquet")
        assert found.anchor.tolist() == ["ABCDXYZ (string)", "Abcd score (string)"]
        assert set(found.source) == {"nhanes"}
        written = json.loads((tmp_path / "pairs" / "nhanes_manifest.json").read_text())
        assert written["files"] == {
            "nhanes.parquet": {
                "rows": 2,
                "sha256": file_sha256(tmp_path / "pairs" / "nhanes.parquet"),
                "csv_sha256": pairs.pairs_sha256(found),
            }
        }
        assert written["counts"]["kind"] == {"name": 2}
        fetch = json.loads(
            (tmp_path / "sources" / "nhanes" / "manifest.json").read_text()
        )
        assert written["sources"] == [
            {
                "role": "fetch",
                "name": "nhanes",
                "version": "1",
                "fetched": fetch["fetched"],
                "files": fetch["files"],
            }
        ]

    def test_an_unfetched_source_is_reported(self, tmp_path, capsys):
        """A source that was not fetched is reported, with the command to fetch it."""
        assert pairs.prepare(["nhanes"], tmp_path) == 1
        assert "sources fetch --source nhanes" in capsys.readouterr().err

    def test_a_run_counts_pairs_an_earlier_source_has(
        self, tmp_path, monkeypatch, capsys
    ):
        """When two sources in one run share a pair, the later source counts it."""
        fetched_nhanes(tmp_path)
        fetched_nhanes(tmp_path, "ncit")
        monkeypatch.setattr(pairs, "READERS", {"ncit": nhanes, "nhanes": nhanes})
        assert pairs.prepare(["nhanes", "ncit"], tmp_path) == 0
        out = capsys.readouterr().out
        assert out.index("ncit:") < out.index("nhanes:")
        assert "kept there once: nhanes 2" in out
