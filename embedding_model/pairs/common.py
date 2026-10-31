"""What the pair readers share: one item shape, one cleaning, and the star of pairs.

Each source is read into items: a concept, a term, a code, an element or a
variable. An item has one canonical text, and other texts that mean the same
thing. ``star`` turns the items into pairs, the same way for every source:
each other text, on the source side, against the canonical with its type,
description and values, on the target side.
"""

import functools
import html
import re
import sys
from dataclasses import dataclass, field

import ftfy
import pandas as pd
from bs4 import BeautifulSoup

from ai_harmonization.dbgap import DBGAP_TYPE_MAP
from embedding_model.normalization import (
    BERT_CLEANING,
    TOKENIZER_NORMALIZER,
    normalised,
)
from embedding_model.texts import DEFAULT_FORMAT, FORMATS, property_text

# An item's status, kept in the pairs' status column.
CURRENT = "current"
RETIRED = "retired"
DRAFT = "draft"
# An anchor's term group, kept in the term_group column.
EXPANSION = "expansion"
PARAPHRASE = "paraphrase"
# The four kinds of pair, kept in the kind column.
NAME = "name"
NAME_DESCRIPTION = "name_description"
DESCRIPTION_DESCRIPTION = "description_description"
NAME_TO_DESCRIPTION = "name_to_description"
# What joins a pair's dbGaP studies in its study column, as in
# "phs999901|phs999902".
STUDY_SEPARATOR = "|"
# The columns of a source's pairs, in order. docs/pairs.md describes each one.
COLUMNS = [
    "anchor",
    "positive",
    "source",
    "kind",
    "item",
    "term_type",
    "term_group",
    "status",
    "study",
    "value_list",
    "anchor_name",
    "anchor_type",
    "anchor_description",
    "anchor_values",
    "positive_name",
    "positive_type",
    "positive_description",
    "positive_values",
]
# Two letters in a row, in any script: what makes a text a name, not an id.
LETTERS = re.compile(r"[^\W\d_]{2}")
# What key() turns into a space: punctuation, underscores and whitespace.
NOT_ALPHANUMERIC = re.compile(r"[\W_]+")
# A word, or one letter.
WORD = re.compile(r"[A-Za-z][a-z]*")
# What joins words into one token, as in "Pre-Operative" or "Signs/Symptoms".
JOINED = re.compile(r"[-/]")
# The fewest letters of a long word. Joined words are a paraphrase when one of
# them is this long, as in "Pre-Operative", and a short form when none is, as
# in "Val-Trp".
LONG_WORD = 5
# The HTML tag names that texts hold. Only these are read as tags: BeautifulSoup
# alone reads any text in angle brackets as a tag, and removes it. So "Candida
# <Debaryomycetaceae> infection" would lose "<Debaryomycetaceae>".
TAGS = (
    "a|abbr|b|big|blockquote|br|caption|center|code|col|colgroup|dd|del|div|dl|dt"
    "|em|font|h[1-6]|hr|i|img|ins|li|ol|p|pre|s|small|span|strike|strong|sub|sup"
    "|table|tbody|td|tfoot|th|thead|tr|tt|u|ul"
)
# A tag's attributes: each a name, "=" and a value, as in class="note".
ATTRIBUTES = r"""(?:\s+[\w:-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'<>]+))*\s*/?"""
# An opening, closing or empty tag of one of TAGS, as in "<b>", "</p>" or "<br/>".
HTML_TAG = re.compile(rf"</?(?:{TAGS}){ATTRIBUTES}>", re.IGNORECASE)
# A superscript's opening tag. Its text gets a "^" before it, so an exponent
# stays apart from its base: "10<sup>9</sup>" becomes "10^9".
SUPERSCRIPT = re.compile(r"<sup\b", re.IGNORECASE)
# Tags that end a line or a cell: their text is apart from the text around it.
BLOCK_TAG = re.compile(
    r"</?(?:blockquote|br|caption|dd|div|dl|dt|h[1-6]|hr|li|ol|p|pre|table|tbody"
    r"|td|tfoot|th|thead|tr|ul)\b",
    re.IGNORECASE,
)
# A character written as an escape: R's "<U+00A0>" or a JSON-style "\u2122".
# No codec reads R's escapes, and Python's "unicode_escape" codec garbles text
# that is not ASCII, as "Café" to "CafÃ©". ftfy's decode_escapes reads "\u2122",
# but it also turns the "\n" and "\t" of a text such as "C:\new\table" into a
# line break and a tab.
ESCAPE = re.compile(r"<U\+([0-9A-Fa-f]{4,6})>|\\u([0-9A-Fa-f]{4})")
# The settings of ftfy's fix_text, which unescapes HTML entities and repairs
# text read with the wrong encoding. It repeats its fixes until the text stops
# changing. So a text that a source escaped twice, such as "&amp;lt;5", becomes
# "<5". unescape_html is True, since
# ftfy's default, "auto", leaves the entities of a text that has a "<", such as
# a tag. uncurl_quotes is off, so quotes stay as the source writes them.
FIXES = ftfy.TextFixerConfig(unescape_html=True, uncurl_quotes=False, explain=False)
# The replacement character's UTF-8 bytes, read as Mac Roman. ftfy leaves them
# as they are, so they are turned back into the replacement character.
LOST_IN_MAC_ROMAN = "\u00d4\u00f8\u03a9"
# A form's number for an item, before its wording: "24. ", "(5a) ", "13f-2. ",
# "34.How", "14.02 How", "1.c During" or "W2 - ". A letter for a sub-item may
# follow it, as in "1. e During".
ITEM_NUMBER = re.compile(
    r"^\(?(?:"
    r"[A-Za-z]{0,2}\d{1,3}[a-z]?(?:-\d{1,2}[a-z]?)?[.)](?:\s+|(?=[A-Z]))"
    r"|\d{1,3}\.(?:\d{2}|[a-z])\s+(?=[A-Z])"
    r"|[A-Za-z]{1,2}\d{1,3}[a-z]?\s+-\s+"
    r")(?:[a-z]\s+(?=[A-Z]))?"
)
# A value's code written before its label: "0= None", "(2) = Mild", "0 -
# absent", "999: Unknown", "(1) All of the time" or "1. Very". Only "=" may be
# followed by anything. The others need a word or a bracket after them, so
# "2 - 3 times" keeps its numbers.
CODED_VALUE = re.compile(
    r"^\s*(?:"
    r"\(?-?\d{1,3}\)?\s*=\s*(?=\S)"
    r"|\(?-?\d{1,3}\)?(?:\s*:|\s+-|\.)\s*(?=[A-Za-z(])"
    r"|\(-?\d{1,3}\)\s+(?=[A-Za-z])"
    r")"
)
# A code joined to its label by "-" alone, as in "0-Never".
DASHED_CODE = re.compile(r"^\s*\(?-?\d{1,3}\)?-(?=[A-Za-z])")
# caDSR writes an element's definition as its concepts' definitions joined by
# "_", ":" or "-", as in "A body part._The middle of it.", or by "_" alone, as
# in "a part_The middle". Each join becomes a space. A label such as
# "EXAMPLE(S):" keeps its colon.
CONCEPT_JOIN = re.compile(
    r"(?:(?<=\.)|(?<=\))(?<!\([Ss]\)))\s*(?:[:_]+|-(?=\s*[A-Z]))\s*"
    r"|(?<=[a-z0-9\]])_(?=[A-Z][a-z ])"
)
# The Java type that ends the long name of an element of an old software model,
# as in "Gene Alias Name java.lang.String".
JAVA_TYPE = re.compile(r"\s*\bjava\.(?:lang|util|sql)\.[A-Za-z]+")
# The phrase about that Java type in the element's definition: "Generic value
# domain for a java datatype that ...", "Value Domain for java language String
# datatype." or "Java Date Type". It often follows the text before it with no
# full stop, so only the phrase is removed, not its sentence.
JAVA_VALUE_DOMAIN = re.compile(
    r"\s*(?:Generic value domain for a (?:primitive )?java\b.*?(?:\.(?=\s|$)|$)"
    r"|Value Domain for java language \w+ datatype\b\.?"
    r"|Java (?:\w+ Type|Double Wrapper for primitive type double)\b\.?)",
    re.IGNORECASE,
)
# The curation mark that caDSR leaves in a definition, such as
# "[Manually-curated]" or "[Standard-based]", with a full stop after it, if any.
CURATION = re.compile(
    r"\s*\[(?:manually|standard|system)[-_ ]\w*\](?:\.(?=\s|$))?", re.IGNORECASE
)
# An empty heading of the template that some NCIt and caDSR definitions
# follow, as in "A thing. EXAMPLE(S): OTHER NAME(S): NOTE(S):".
EMPTY_HEADING = re.compile(
    r"\s*\b(?:EXAMPLE|OTHER NAME|NOTE)\(S\):\s*(?=(?:EXAMPLE|OTHER NAME|NOTE)\(S\):|$)"
)
# The notices that publishers add to definitions about Executive Order 14168.
# They say nothing about the concept: NCIt's, after "IMPORTANT NOTICE:" or in
# brackets, and the one that it gives as a retired concept's whole
# definition; caDSR's; and NIH CDE's. CDISC copies NCIt's.
NOTICE = re.compile(
    r"\[?(?:IMPORTANT NOTICE: )?The NCI Thesaurus contains biomedical"
    r" terminologies that NCI does not own or control\. This concept contains"
    r" gender-related content that does not comply with Executive Order"
    r" 14168\.\]?"
    r"|This concept has been retired and marked obsolete to ensure compliance"
    r" with the Executive Order 14168 which mandates that gender should not be"
    r" used as a replacement for sex\."
    r"|This content is being reviewed for compliance with Executive Order 14168"
    r" which mandates that gender should not be used as a replacement for sex\."
    r'|NLM disclaimer: Consistent with Executive Order 14168, "gender" should'
    r' not be used as a replacement for the element "sex" in the NIH CDE'
    r" repository\."
)
# A question's letter item, as in "C. What year?". It is taken only from a text
# that ends with "?", since "H. Influenzae" names a germ.
LETTER_ITEM = re.compile(r"^[A-Z]\.\s+(?=[A-Z][a-z])")
# How LOINC and the NIH CDE mark a deprecated name: "Deprecated", even twice
# or with no space after it.
DEPRECATED = re.compile(r"^(?:deprecated\s*)+", re.IGNORECASE)


@dataclass(frozen=True)
class Text:
    """A text of an item other than its canonical one, for the source side.

    Attributes:
        text (str): The text.
        term_type (str): The source's own label for it, such as ``SY``.
        question (bool): Whether it is a question's wording, which is always a
            paraphrase.
        study (str): The dbGaP studies it comes from, joined by ``|``, if any.
    """

    text: str
    term_type: str
    question: bool = False
    study: str = ""


@dataclass
class Item:
    """One concept, term, code, element or variable of a source.

    Attributes:
        id (str): The source's own id for it.
        canonical (str): Its canonical text, for the target side.
        type (str): The source's own type word, or "" when it has none.
        description (str): Its description, or "".
        lists (list[tuple[str, list[str]]]): Its value lists, each with the
            list's name. Its pairs are made once per list.
        texts (list[Text]): Its other texts of the same meaning.
        descriptions (list[str]): Its other descriptions.
        status (str): CURRENT, RETIRED or DRAFT.
        replaced_by (list[str]): The ids of the items that replace it.
        open (bool): Whether its restriction field allows its use.
    """

    id: str
    canonical: str
    type: str = ""
    description: str = ""
    lists: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    descriptions: list = field(default_factory=list)
    status: str = CURRENT
    replaced_by: list = field(default_factory=list)
    open: bool = True


def escaped(match):
    """The character that an escape stands for, else the escape itself."""
    point = int(match.group(1) or match.group(2), 16)
    return chr(point) if point <= sys.maxunicode else match.group(0)


def without_tags(text):
    """A text with its HTML tags removed by BeautifulSoup, and the rest kept.

    Only what HTML_TAG finds is read as a tag. The text around the tags is
    escaped first, so "<5", "a<b and c>d" and "&" stay as they are. A
    superscript is written after a "^", so "10<sup>9</sup>" is "10^9".

    Args:
        text (str): A text with tags.

    Returns:
        str: The text without them.
    """
    # The text is copied piece by piece into parts. last is where the text not
    # yet copied starts.
    parts, last = [], 0
    for tag in HTML_TAG.finditer(text):
        # A block tag ends a line or a cell, so a space keeps the words on its
        # two sides apart.
        space = " " if BLOCK_TAG.match(tag.group(0)) else ""
        # The text before the tag, escaped: "<" becomes "&lt;" and "&" becomes
        # "&amp;", so BeautifulSoup cannot read any of it as markup.
        parts += [html.escape(text[last : tag.start()], quote=False), space]
        # The tag itself, not escaped, so BeautifulSoup reads it as a tag and
        # removes it. An opening <sup> is followed by "^".
        parts += [tag.group(0), space, "^" if SUPERSCRIPT.match(tag.group(0)) else ""]
        last = tag.end()
    # The text after the last tag, escaped too.
    parts.append(html.escape(text[last:], quote=False))
    # get_text() drops the tags, and reads "&lt;" and "&amp;" as "<" and "&".
    return BeautifulSoup("".join(parts), "html.parser").get_text()


# Values such as "Yes" or "Unknown" repeat across many items. The cache keeps
# the last 65,536 (2^16) clean texts, so a repeated text is cleaned once: on
# caDSR's 79,157 elements, two thirds of the calls are answered from it.
@functools.lru_cache(maxsize=65_536)
def clean(text):
    """A text with its markup removed and its characters and spaces normalised.

    Escapes such as R's ``<U+00A0>`` are read first. ftfy then unescapes
    entities such as ``&gt;``, repairs text read with the wrong encoding, such
    as ``Iâ€™m`` to ``I’m``, removes control characters and normalises Unicode
    to NFC. HTML tags are removed, and other text in angle brackets, such as
    ``<time point>``, is kept. A replacement character becomes a space. BERT's
    text cleaning then removes the characters that hold no text, and every run
    of whitespace becomes one space.

    These steps repeat until the text stops changing, as ftfy repeats its own
    fixes. So a clean text cleans to itself, even when one step leaves work for
    an earlier one: ``Caf&lt;U+00E9&gt;`` becomes ``Caf<U+00E9>`` in the first
    pass, and ``Café`` in the second.

    Args:
        text (str | None): The text as the source gives it.

    Returns:
        str: The clean text, or "" for a missing one.
    """
    if not isinstance(text, str):
        return ""
    while (cleaner := cleaning_pass(text)) != text:
        text = cleaner
    return text


def cleaning_pass(text):
    """A text after one pass of ``clean``'s steps, in their order."""
    text = ESCAPE.sub(escaped, text)
    text = ftfy.fix_text(text, FIXES).replace(LOST_IN_MAC_ROMAN, "\ufffd")
    if HTML_TAG.search(text):
        text = without_tags(text)
    text = BERT_CLEANING.normalize_str(text.replace("\ufffd", " "))
    return " ".join(text.split())


def without_java(text):
    """A long name or definition without its Java type and the phrase about it.

    Args:
        text (str): The long name or definition.

    Returns:
        str: The text without them.
    """
    return JAVA_TYPE.sub("", JAVA_VALUE_DOMAIN.sub("", text))


def cadsr_definition(text):
    """A caDSR definition, with each join a space and no curation mark.

    caDSR joins its concepts' definitions with "_", ":" or "-", as in ``A body
    part._The middle of it.``. It marks some definitions as curated, as in
    ``[Manually-curated]``. The mark is removed, and each join becomes a space.
    The NCI-stewarded elements of the NIH CDE copy such definitions.

    Args:
        text (str): The definition.

    Returns:
        str: The fixed definition.
    """
    return CONCEPT_JOIN.sub(" ", CURATION.sub("", text))


def undeprecated(name):
    """A name without the "Deprecated" that marks it, as in ``Deprecated Abcd``."""
    return DEPRECATED.sub("", name)


def key(text):
    """A text's key for comparing texts.

    The text is normalised as the tokenizer normalises it, then case-folded,
    with punctuation as spaces. So ``Größe`` and ``GROSSE`` have one key, and
    ``Abcdé_Efgh`` and ``abcde efgh`` too.
    """
    folded = TOKENIZER_NORMALIZER.normalize_str(text).casefold()
    return NOT_ALPHANUMERIC.sub(" ", folded).strip()


def is_name(text):
    """Whether a text is a name: it has two letters in a row, unlike an id."""
    return bool(LETTERS.search(text))


def is_expansion_token(text):
    """Whether a text is an expansion token, shaped like a short form or a code.

    An expansion token is one token with a digit, an underscore or two
    capitals, such as ``BMI``, ``IL-17D`` or ``RACE_CAT_TXT``. Words joined by
    "-" or "/" are not an expansion token when one of them has five letters or
    more, as in ``Pre-Operative`` or ``T-Lymphocytes``. A number may be one of
    the parts, as in ``Galectin-10``. Short parts make a short form, as in
    ``Val-Trp``.

    Args:
        text (str): The clean text.

    Returns:
        bool: Whether it is an expansion token.
    """
    parts = JOINED.split(text)
    words = all(WORD.fullmatch(part) or part.isdigit() for part in parts)
    long_word = any(WORD.fullmatch(part) and len(part) >= LONG_WORD for part in parts)
    if " " in text or words and long_word:
        return False
    capitals = sum(character.isupper() for character in text)
    return capitals >= 2 or "_" in text or any(c.isdigit() for c in text)


def term_group(text, question=False):
    """Whether a source-side text is an expansion or a paraphrase of its target.

    An expansion token is an expansion. Anything else, and every question's
    wording, is a paraphrase.

    Args:
        text (str): The clean text.
        question (bool): Whether the text is a question's wording.

    Returns:
        str: EXPANSION or PARAPHRASE.
    """
    if not question and is_expansion_token(text):
        return EXPANSION
    return PARAPHRASE


def type_word(own, values):
    """An item's type: the source's own word, else one of DBGAP_TYPE_MAP's words.

    Args:
        own (str): The source's type word, or "".
        values (list[str] | None): The item's values on this side.

    Returns:
        str: ``own``, or ``string/encoded`` with values and ``string`` without.
    """
    if own:
        return own
    return DBGAP_TYPE_MAP["encoded"] if values else DBGAP_TYPE_MAP["string"]


def render(name, type_, description, values):
    """One side's text, in the default format, which scoring also uses.

    Neither side of a training pair has a node, so the node is "" and the text
    starts at the name, as a source variable's text does in scoring.
    """
    return property_text(
        FORMATS[DEFAULT_FORMAT], "", name, type_, description, values or None
    )


def value_labels(values):
    """A value list's labels, without the codes written before them.

    The values are cleaned first. The codes are removed only when every value
    of the list has a code. No two codes may be the same, and no two labels may
    be the same. So ``0 - Never | 1 | ... | 10 - Always`` keeps its numbers,
    which are its values. A code joined by "-" alone, as in ``0-Never``, is
    removed only from a list of three values or more, since ``2-Dimensional``
    is a name.

    Args:
        values (list[str]): The values, as the source gives them.

    Returns:
        list[str]: The labels, in the same order.
    """
    values = [clean(value) for value in values]
    for code, fewest in ((CODED_VALUE, 1), (DASHED_CODE, 3)):
        code_matches = [code.match(value) for value in values]
        if len(values) < fewest or not all(code_matches):
            continue
        codes = [match.group(0).strip() for match in code_matches]
        labels = [value[match.end() :] for value, match in zip(values, code_matches)]
        if len(set(codes)) == len(codes) and len(set(labels)) == len(labels):
            return labels
    return values


def without_item_number(text):
    """A form's wording for an item, without the number that the form gives it."""
    return ITEM_NUMBER.sub("", text)


def worded(text):
    """A source-side text cleaned, and a question's text without its item number.

    A form numbers its items, as in ``24. What is your marital status?``. The
    number is the form's layout, so a question text loses it. Other texts
    keep their numbers, such as a study variable's ``F361. Wheezing medicine``.

    Args:
        text (Text): The text.

    Returns:
        str: The clean text.
    """
    clean_text = clean(text.text)
    if not text.question:
        return clean_text
    clean_text = without_item_number(clean_text)
    return LETTER_ITEM.sub("", clean_text) if clean_text.endswith("?") else clean_text


def cleaned(item):
    """An item with every text cleaned, and texts that clean to nothing dropped.

    When the description is empty, the first other description takes its
    place. A description loses the publishers' notices about Executive Order
    14168, in NOTICE, and then the empty headings of the template that some
    NCIt and caDSR definitions follow, such as ``NOTE(S):``. A description
    that was only a notice is dropped. Value lists that clean to the same
    values are kept once.
    """
    texts = [
        Text(text, original.term_type, original.question, original.study)
        for original in item.texts
        if (text := worded(original))
    ]
    descriptions = [
        text
        for text in (
            " ".join(EMPTY_HEADING.sub(" ", NOTICE.sub(" ", clean(original))).split())
            for original in [item.description, *item.descriptions]
        )
        if text
    ]
    distinct_lists = {}
    for name, values in item.lists:
        values = [value for value in map(clean, values) if value]
        if values:
            distinct_lists.setdefault(tuple(values), name)
    return Item(
        id=item.id,
        canonical=clean(item.canonical),
        type=clean(item.type),
        description=next(iter(descriptions), ""),
        lists=[(name, list(values)) for values, name in distinct_lists.items()],
        texts=texts,
        descriptions=descriptions[1:],
        status=item.status,
        replaced_by=item.replaced_by,
        open=item.open,
    )


def replacements(item, by_id):
    """The open items, current or draft, that replace a retired item.

    Each chain is followed through retired items, open or not, since none of
    their own texts are used.

    Args:
        item (Item): A retired item.
        by_id (dict[str, Item]): Every item of the source, by id.

    Returns:
        list[Item]: The replacements, in the order given, each once.
    """
    replacing_items, seen, waiting = [], {item.id}, list(item.replaced_by)
    while waiting:
        replacement = by_id.get(waiting.pop(0))
        if replacement is None or replacement.id in seen:
            continue
        seen.add(replacement.id)
        if replacement.status == RETIRED:
            waiting += replacement.replaced_by
        elif replacement.open and replacement.canonical:
            replacing_items.append(replacement)
    return replacing_items


def joined_studies(studies):
    """Several study texts joined into one, each study once.

    A study text holds dbGaP studies joined by STUDY_SEPARATOR. The result holds every
    study of the texts given, once each and sorted. Empty parts are dropped.
    For example, ``["phs999902|phs999901", "", "phs999901"]`` gives
    ``"phs999901|phs999902"``.

    Args:
        studies (Iterable[str]): Study texts.

    Returns:
        str: The studies, joined by STUDY_SEPARATOR.
    """
    parts = {part for study in studies for part in study.split(STUDY_SEPARATOR)}
    return STUDY_SEPARATOR.join(sorted(parts - {""}))


def normalised_text(text):
    """A text as the tokenizer normalises it, to compare two texts as the model reads them."""
    return TOKENIZER_NORMALIZER.normalize_str(text)


def distinct(texts, canonical):
    """The texts that are names, each once, without texts equal to the canonical.

    Texts are compared as the tokenizer normalises them, as copies are, and
    without a form's item number. So ``Q2. Abcd?`` equals ``Abcd?``, and
    ``ABCD`` equals ``abcd``, but ``abcd_efgh`` and ``abcd efgh`` differ, as
    their tokens do. Equal texts become one text. It keeps the first spelling
    and the studies of all of them.
    """
    by_key, canonical_key = {}, normalised_text(without_item_number(canonical))
    for text in texts:
        text_key = normalised_text(without_item_number(text.text))
        if text_key and text_key != canonical_key and is_name(text.text):
            first = by_key.setdefault(text_key, text)
            if text.study and text.study != first.study:
                study = joined_studies([first.study, text.study])
                by_key[text_key] = Text(
                    first.text, first.term_type, first.question, study
                )
    return list(by_key.values())


def item_pairs(
    target, names, status, anchor_type, descriptions, to_description, source
):
    """The pairs of one target: a star around its canonical, once per value list.

    Each name gives a ``name`` pair, and a ``name_description`` pair when the
    target has a description. Each other description gives a
    ``description_description`` pair, unless it repeats the description or
    the canonical. When no name is left, the canonical is paired with the
    description alone, as ``name_to_description``, unless the two are the
    same text.

    Args:
        target (Item): The item on the target side.
        names (list[Text]): The texts for the source side.
        status (str): The status of the item the texts come from.
        anchor_type (str): The source side's type.
        descriptions (list[str]): Other descriptions, for the source side.
        to_description (bool): Whether the canonical may be paired with the
            description when no name is left.
        source (str): The source's name, written into each pair.

    Returns:
        list[tuple]: The pairs, as rows in COLUMNS' order.
    """
    names = distinct(names, target.canonical)
    description_key, canonical_key = key(target.description), key(target.canonical)
    skipped = {"", description_key, canonical_key}
    others = {}
    for text in descriptions:
        text_key = key(text)
        if text_key not in skipped and is_name(text):
            others.setdefault(text_key, text)
    anchors = {}
    rows = []
    for list_name, values in target.lists or [("", [])]:
        type_ = type_word(target.type, values)
        positives = {}

        def row(kind, anchor, positive, term_type, group, study=""):
            """One pair as a row, each side rendered beside its fields."""
            if anchor not in anchors:
                anchors[anchor] = render(anchor, anchor_type, "", [])
            if positive not in positives:
                name, description = positive
                positives[positive] = render(name, type_, description, values)
            return (
                anchors[anchor],
                positives[positive],
                source,
                kind,
                target.id,
                term_type,
                group,
                status,
                study,
                list_name,
                anchor,
                anchor_type,
                "",
                [],
                positive[0],
                type_,
                positive[1],
                values,
            )

        name_only = (target.canonical, "")
        described = (target.canonical, target.description)
        for text in names:
            group = term_group(text.text, text.question)
            rows.append(
                row(NAME, text.text, name_only, text.term_type, group, text.study)
            )
            if target.description:
                rows.append(
                    row(
                        NAME_DESCRIPTION,
                        text.text,
                        described,
                        text.term_type,
                        group,
                        text.study,
                    )
                )
        if target.description:
            for other in others.values():
                rows.append(
                    row(
                        DESCRIPTION_DESCRIPTION,
                        other,
                        described,
                        "description",
                        term_group(other),
                    )
                )
        differs = description_key != canonical_key
        if not names and to_description and target.description and differs:
            if is_name(target.canonical):
                rows.append(
                    row(
                        NAME_TO_DESCRIPTION,
                        target.canonical,
                        (target.description, ""),
                        "canonical",
                        term_group(target.canonical),
                    )
                )
    return rows


def distinct_pairs(pairs):
    """Each pair once: its first copy, with the studies of every copy.

    Two pairs are copies when their two texts, ``anchor`` and ``positive``,
    are the same once the tokenizer normalises them. The model cannot tell
    such pairs apart. For example, ``BMI`` -> ``Body Mass Index`` and ``bmi``
    -> ``body mass index`` are copies. Only the first copy is kept, as it is
    written. The order of ``pairs`` decides which copy comes first, such as a
    current item's copy before a retired item's. A pair from a benchmark's
    study must not be used for training. So the copy kept takes the studies of
    every copy, and the study rule sees them all.

    For example, three copies of ``ABCD_SC`` -> ``Abcd score`` have the
    studies ``phs999901``, none, and ``phs999902|phs999901``. The first copy
    is kept, and its study is updated from ``phs999901`` to
    ``phs999901|phs999902``.

    Args:
        pairs (pd.DataFrame): Pairs, with ``anchor``, ``positive`` and
            ``study``, the studies joined by ``|``.

    Returns:
        pd.DataFrame: The first copy of each pair, in the order given. Each
            keeps its index, so a caller can tell which rows were dropped.
    """
    # One number per pair of texts, the same for all its copies.
    group = pairs.groupby(
        [normalised(pairs.anchor), normalised(pairs.positive)], sort=False
    ).ngroup()
    # Every copy that holds a study. keep=False marks the first copy too.
    copies_with_study = group.duplicated(keep=False) & (pairs.study != "")
    # The joined studies of each pair, by its number.
    studies = (
        pairs.study[copies_with_study]
        .groupby(group[copies_with_study])
        .agg(joined_studies)
    )
    # One row per pair: its first copy. The later copies are dropped.
    first_copies = pairs[~group.duplicated()]
    if studies.empty:
        return first_copies
    # Each kept pair takes its pair's joined studies. A pair without copies
    # gets none there, and keeps its own study.
    merged = group[first_copies.index].map(studies)
    return first_copies.assign(study=merged.fillna(first_copies.study))


def star(items, source):
    """Every pair of a source's items, with what was left out, counted.

    Items whose restriction field does not allow use are left out. A retired
    item with a replacement gives its canonical and its texts to the
    replacement's canonical. A retired item without a replacement keeps its own
    canonical. Its texts are used only when no current or draft item has them.
    When two pairs have the same two texts, one is kept: the current item's
    copy, else the draft item's, else the retired item's. It takes the studies
    of every copy.

    Args:
        items (Iterable[Item]): The source's items.
        source (str): The source's name, written into each pair.

    Returns:
        tuple[pd.DataFrame, dict[str, int]]: The pairs, in COLUMNS, and the
            counts of items read, items not open and duplicates removed.
    """
    items = [cleaned(item) for item in items]
    by_id = {item.id: item for item in items}
    live = set()
    for item in items:
        if item.status != RETIRED:
            live.add(key(item.canonical))
            live.update(key(text.text) for text in item.texts)
    rows = []
    for item in items:
        if not item.open:
            continue
        anchor_type = type_word(item.type, [])
        replacing_items = replacements(item, by_id) if item.status == RETIRED else []
        if replacing_items:
            names = item.texts
            if item.canonical:
                names = [Text(item.canonical, "canonical"), *names]
            for replacement in replacing_items:
                rows += item_pairs(
                    replacement,
                    names,
                    item.status,
                    anchor_type,
                    [],
                    False,
                    source=source,
                )
        elif item.canonical:
            names = item.texts
            if item.status == RETIRED:
                names = [text for text in names if key(text.text) not in live]
            rows += item_pairs(
                item,
                names,
                item.status,
                anchor_type,
                item.descriptions,
                item.status != RETIRED or key(item.canonical) not in live,
                source=source,
            )
    pairs = pd.DataFrame(rows, columns=COLUMNS)
    rank = pairs.status.map({CURRENT: 0, DRAFT: 1, RETIRED: 2})
    first = pairs.loc[rank.sort_values(kind="stable").index]
    kept_pairs = distinct_pairs(first).sort_index().reset_index(drop=True)
    counts = {
        "items": len(items),
        "items_not_open": sum(not item.open for item in items),
        "duplicates_removed": len(pairs) - len(kept_pairs),
    }
    return kept_pairs, counts
