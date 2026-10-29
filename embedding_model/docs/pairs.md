# Pairs

The model learns from pairs of texts that mean the same thing. `pairs
prepare` makes them from the [sources](sources.md) that `sources fetch` wrote:

```
python -m embedding_model pairs prepare                        # every source
python -m embedding_model pairs prepare --source nhanes phenx  # some of them
```

A pair has two sides. Their names come from sentence-transformers:

- the **anchor**: a text as a study might word a variable, such as a short
  name or a question;
- the **positive**: the same thing as a data model would describe its
  property, with its name, type, description and values.

So the anchor plays the source variable of a mapping, and the positive plays
its target. Both sides are written in `compact`, the format that scoring uses
(see [scoring.md](scoring.md#running-score)). It writes a description and
values only when a side has them:

```
anchor:   SVG (string)
positive: Saphenous Vein Graft (string): A vessel graft comprised of the saphenous vein.
```

## The shape: a star around each item

Each source is read into items: a concept, a term, a code, a data element or a
variable. An item has one **canonical** text, the name that its source prefers,
such as a thesaurus's preferred name or an ontology's label. Every other text
of the item that means the same thing becomes one anchor. Its positive is the
canonical, with the item's type, description and values. Each item is the
centre of a star, and no pair joins two of its other texts.

There are four kinds of pair:

| kind | anchor | positive |
|---|---|---|
| `name` | another text of the item | the canonical, its type and values |
| `name_description` | another text of the item | the canonical, its type, description and values |
| `description_description` | another description of the item, such as a second definition | the canonical, its type, its first description and values |
| `name_to_description` | the canonical, for an item with no other text | the description alone, with the type and values |

When the item has a description, each other text gives two pairs: a `name`
pair and a `name_description` pair.

## What a pair holds

`pairs prepare` writes each source's pairs to
`embedding_model/inputs/pairs/<source>/pairs.parquet`. When `--inputs` names
another folder, they go to `pairs/<source>/` in that folder. The folder is
written whole; see [Inputs and outputs](../README.md#inputs-and-outputs). One
row is one pair:

| column | what it holds |
|---|---|
| `anchor` | the source side's text, formatted as the model sees it |
| `positive` | the target side's text, formatted as the model sees it |
| `source` | the source, such as `ncit` |
| `kind` | `name`, `name_description`, `description_description` or `name_to_description` |
| `item` | the source's own id of the item on the positive side |
| `term_type` | what the anchor's text is: the source's own label, such as a synonym type or an alternate-name type, or else its field, such as `short name`; `canonical` and `description` for a canonical and another description |
| `term_group` | `expansion` or `paraphrase`; see [Expansions and paraphrases](#expansions-and-paraphrases) |
| `status` | `current`, `retired` or `draft`: the status of the item the anchor's text comes from |
| `study` | the dbGaP studies the anchor's text comes from, joined by `\|`, if any |
| `value_list` | the name of the positive's value list, where a source names its lists |
| `anchor_name`, `anchor_type`, `anchor_description`, `anchor_values` | four columns: the fields the anchor was formatted from |
| `positive_name`, `positive_type`, `positive_description`, `positive_values` | four columns: the fields the positive was formatted from |

Each side's fields are kept beside its text, so a later step can compare
fields rather than formatted texts.

`pairs prepare` also writes a manifest for each source, `manifest.json`,
beside its pairs in the same folder. It has the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: the pairs' row count, their file's `sha256` and their table's
  `csv_sha256`;
- `sources`: the fetch that the pairs come from, with its version and its
  tables' checksums;
- `counts`:
  - the items, and the items not open;
  - the pairs by kind, term group and status;
  - the duplicates removed;
  - the texts that point to several canonicals.

## The rules every source shares

The same rules apply to every source. The [next section](#each-source) lists
the fields that each source uses for them.

### Only open items

An item is used only when its source's restriction field allows it. The
source's tables mark this in their `open` column (see
[Restricted items](licenses.md#restricted-items)).

### The canonical

The canonical is the source's own preferred field.

### Texts of the same meaning

A text is an anchor only when its source marks it as the same thing as the
canonical. So ontology synonyms that are only related, broader or narrower
are left out. LOINC's related names are left out too. Each source's section
lists the texts that it uses.

### Names

An anchor must be a name: it has two letters in a row. So ids and bare
numbers, such as `2223612v3.0:2178533v1.0` or `535`, are not anchors.

An item uses each of its texts once as a name, even when its source lists a
text twice. Texts are compared as the tokenizer
[normalises](tokenizer.md#step-1-normalize) them, and a repeat is not used.
The canonical is not used as an anchor either, since it is the positive. For
example, `BODY WEIGHT` repeats the canonical `Body Weight`, so it is not used.
But `body_weight` is an anchor, since the tokenizer reads its `_` as a token
of its own.

### Expansions and paraphrases

Each pair has a `term_group`: `expansion` or `paraphrase`. Only the anchor's
text decides it, so every kind of pair can have either group.

| `term_group` | the anchor | examples: anchor -> canonical |
|---|---|---|
| `expansion` | a short form or a code: one token with a digit, an underscore or two capitals | `BMI` -> Body Mass Index; `IL-17D` -> Interleukin-17D; `RACE_CAT_TXT` -> Race Category Text; `Val-Trp` -> Valyltryptophan |
| `paraphrase` | words, or a phrase | Heart Attack -> Myocardial Infarction; Adalat -> Nifedipine; Pre-Operative -> Preoperative; Galectin-10 -> Eosinophil Lysophospholipase |
| `paraphrase` | a question text, whatever its shape | "Are you pregnant now?" -> Current Pregnancy Indicator; "Date of Birth" -> Person Birth Date |

The examples come from NCIt, caDSR and the NIH CDE. Words joined by "-" or
"/" are a paraphrase when one of them has five letters or more, as in
`Pre-Operative` and `Galectin-10`. Otherwise they are a short form, as
`Val-Trp` is. The rule looks only at shape, so `Long-Term` is an expansion
too.

A question text is the wording that a form shows for an item. Question texts
are anchors because studies often word their variables the same way. Each
source's section names the fields that hold them.

### Retired items

Retired items are used too, since older studies use their wording:

- A source can name a replacement for a retired item. The retired item's
  canonical and texts then become anchors of the replacement's canonical.
  When the replacement is retired too, its own replacement is used, up to an
  item that is not retired.
- A retired item without a replacement keeps its own canonical. Its texts are
  used only when no current or draft item has them.

Draft items are used like current items. The `status` column keeps them
apart, so a training run can leave them out.

### One text, several canonicals

A text can name several items, such as "Other, specify". It is then
used for each of them. The manifest counts such texts.

### Duplicates

Each item is read once, by its source's id. Two pairs are copies when the
[tokenizer](tokenizer.md) normalises their texts to the same two texts. The
model cannot tell copies apart, so only the first copy is kept, as it is
written. It keeps the studies of every copy. For example:

| one pair | another pair | copies? |
|---|---|---|
| `BMI` -> `Body Mass Index (float)` | `bmi` -> `body mass index (float)` | yes: only the case differs |
| `Treatment-naïve` -> `Treatment Naive Indicator (string)` | `treatment-naive` -> `Treatment Naive Indicator (string)` | yes: only an accent differs |
| `body_wt` -> `Body Weight (float)` | `body wt` -> `Body Weight (float)` | no: the tokenizer keeps the `_` as a token |

`corpus build` combines the sources into one [corpus](corpus.md). A pair that
two sources share is then kept only in the earlier source.
[Each source](#each-source) lists the sources in this order. NCIt comes first,
since CDISC publishes its terms as NCIt subsets. `pairs prepare` prints how
many pairs each source shares with an earlier source in the same run.

### Values

A positive's values are the item's own value list, in the source's order.
Lists are never merged. Values are labels, such as "Female", not codes, such
as "F". A pair keeps the whole list. Its text shows the first 12 values, as
`ai_harmonization.formatters` does. Anchors have no values.

Some sources write a code before each label, as in "0= None" or "0 -
absent". The codes are removed only when:

- every value of the list has a code;
- no two codes are the same, and no two labels are the same;
- the list has three values or more, for codes joined by "-" alone, as in
  "0-Never".

So "0 - Never | 1 | ... | 10 - Always" keeps its numbers, since "1" has no
label: the numbers are its values. And a short list keeps "2-Dimensional" as
a name.

An item can have several lists, as an NCIt concept can. It then gives its
pairs once for each distinct list.

### Types

A source's own type word is kept, such as `Value List`. When a
source has none, the type is `string/encoded` for an item with values and
`string` for one without. `ai_harmonization.dbgap` maps dbGaP's types to
these same words. An anchor takes its item's type word, or `string` when
there is none.

### Cleaning

Every text is cleaned the same way:

- escapes such as R's `<U+00A0>` are read, and HTML entities are unescaped,
  such as `&gt;` to `>`. An entity that a source escaped twice, as `&amp;gt;`,
  becomes `>` too;
- UTF-8 text decoded as cp1252 is repaired, such as `Iâ€™m` to `I’m`, even
  when it was decoded twice;
- HTML tags are removed. Other text in angle brackets is kept, as in
  "Candida <Debaryomycetaceae> infection";
- Unicode is normalised;
- characters that hold no text are removed, as BERT's text cleaning removes
  them: control, format and private-use characters, such as zero-width
  spaces and direction marks;
- each run of whitespace becomes one space.

The steps run in this order. For example, a description from an HTML page:

```
<p>Age<U+00A0>(years)</p><p>Enter 0 if the participantâ€™s age is &lt; 1 year</p>
```

| step | what changes | the text after it |
|---|---|---|
| escapes | `<U+00A0>` becomes a no-break space | `<p>Age (years)</p><p>Enter 0 if the participantâ€™s age is &lt; 1 year</p>` |
| entities and encoding | `&lt;` becomes `<`, and `participantâ€™s` becomes `participant’s` | `<p>Age (years)</p><p>Enter 0 if the participant’s age is < 1 year</p>` |
| tags | `<p>` and `</p>` go, and each leaves a space. `< 1 year` is not a tag, so it stays | ` Age (years)  Enter 0 if the participant’s age is < 1 year ` |
| characters and spaces | the no-break space becomes a space, and each run of spaces becomes one | `Age (years) Enter 0 if the participant’s age is < 1 year` |

Unescaping turns `&lt;` into `<`, but it removes nothing. Only the tag step
removes text, and only real HTML tags.

The steps run again until the text stops changing. For example, an escape
written with entities, `Caf&lt;U+00E9&gt;`, becomes `Caf<U+00E9>` in the first
run, and `Café` in the second. A third run leaves `Café` as it is, so cleaning
stops there.

Cleaning keeps accents, since they are part of the text, as in `Café`. Later
steps use the tokenizer's normalisation, which removes them: to compare texts,
as for [duplicates](#duplicates), and to cut a text into tokens (see
[tokenizer.md](tokenizer.md#step-1-normalize)).

A question text also loses the number that a form gives it, as in "24. What
is your marital status?". Units are kept, such as "(mg/dL)".

Some NCIt and caDSR definitions follow a template with the headings
"EXAMPLE(S):", "OTHER NAME(S):" and "NOTE(S):". A heading with nothing after
it is removed from a description.

NCIt, caDSR and NIH CDE add notices about Executive Order 14168 to some
definitions. One reads: "This content is being reviewed for compliance with
Executive Order 14168 ...". CDISC copies NCIt's notices. A notice says
nothing about the concept, so it is removed from a description. A description
that was only a notice, as some retired NCIt concepts have, is dropped.

## Each source

Each source's section below has the same parts: the items, the canonical,
the anchors, the description, the values and the status. A field, such as
`preferred_name`, is a column of a table that `sources fetch` writes.
[sources.md](sources.md) lists each source's tables. A status that a section
does not name is read as current, and `pairs prepare` prints it.

| source | canonical (the positive's name) | anchors | description | values | retired items replaced by |
|---|---|---|---|---|---|
| [NCIt](#ncit) | the preferred name | synonyms of the same meaning | the first definition, NCI's where there is one | each registry's list | EVS's history |
| [Ontologies](#ontologies) | the label | EXACT synonyms | the definition | none | "term replaced by" |
| [LOINC](#loinc) | the long common name | short and display names, question texts | the definition | the default answer list | the MapTo file |
| [CDISC](#cdisc) | a term's NCI Preferred Term; a codelist's Codelist Name | submission values, synonyms, question texts | the first definition | a codelist's terms | none |
| [NIH CDE](#nih-cde) | the first designation | the other designations | the first definition | the value meanings | none |
| [caDSR](#cadsr) | the long name | the short name, alternate names, question texts | the definition | the value meanings | none |
| [NHANES](#nhanes) | the codebook's English Text | the variable's name and labels | none | the value labels | none |
| [PhenX](#phenx) | the variable's description | the variable's name, and mapped dbGaP variables | none | the value labels | none |

### NCIt

- **Items:** each concept, a row of `concepts`.
- **Canonical:** `preferred_name`.
- **Anchors:** the `name` of each synonym whose `term_type` means the same
  thing as the preferred name:
  - NCIt's own synonyms (`SY`) and other sources' preferred terms (`PT`);
  - antiquated terms (`AQ`, `AQS`) and display names (`DN`);
  - abbreviations (`AB`), and brand names (`BR`, `FB`), such as `Adalat` for
    Nifedipine;
  - country codes, such as `USA` (`CA2`, `CA3`, `CS`). The number codes
    (`CNU`), such as `250` for France, are not names, so they give no pairs;
  - left out: grammar forms (`AD`), headers (`HD`), chemical structures
    (`SN`) and drug study codes (`CN`).
- **Description:** the first `definition`. A `DEFINITION` comes before an
  `ALT_DEFINITION`, and a definition whose `source` is `NCI` comes before
  another source's. Other definitions give `description_description` pairs.
  NCIt can end a definition with its source, such as "(NCI)" or "(NCI04)",
  and this source is removed.
- **Values:** the `value`s of each registry's list. A concept can have lists
  from several registries, such as GDC, PCDC, CTDC, ICDC and SeroNet.
  `value_list` is the list's `association`, such as `Has_GDC_Value`. A
  concept gives its pairs once for each distinct list. NCIt gives a list no
  order, so its values are sorted by name.
- **Status:** from `status`. `Retired_Concept` and `Obsolete_Concept` are
  retired, and `Provisional_Concept` and `Concept_Pending_Approval` are
  drafts. The `replacements` table names a retired concept's replacement:
  the concept that a merge record of the history file names.

### Ontologies

- **Items:** each term of each [ontology](sources.md#ontologies), a row of
  `terms`.
- **Canonical:** `label`.
- **Anchors:** the synonyms whose `scope` is `EXACT`, of every synonym type,
  such as an abbreviation or a layperson's term. `RELATED`, `BROAD` and
  `NARROW` synonyms are left out.
- **Description:** the first `definition`. Other definitions give
  `description_description` pairs.
- **Values:** none.
- **Status:** a term whose `obsolete` is true is retired. The word "obsolete"
  marks its label, synonyms and definitions, and it is removed. `replaced_by`
  names its replacement.

### LOINC

- **Items:** each code, a row of `codes`.
- **Canonical:** `LONG_COMMON_NAME`.
- **Anchors:** `SHORTNAME` and `DisplayName`, and two question texts:
  `SURVEY_QUEST_TEXT`, and the `DisplayNameForForm` of each open row of
  `forms`. `RELATEDNAMES2` and `CONSUMER_NAME` are left out: LOINC says they
  are not synonyms.
- **Description:** `DefinitionDescription`.
- **Values:** the `DisplayText` of each answer in the code's default answer
  list, in the list's own order. `value_list` names the list by its
  `AnswerListId` and `AnswerListLinkType`, such as `LL9999-1 (PREFERRED)`.
- **Status:** from `STATUS`. `DEPRECATED` and `DISCOURAGED` codes are
  retired, and `TRIAL` codes are drafts. `MAP_TO`, in `map_to`, names a
  retired code's replacement. A deprecated code's names can start with the
  word "Deprecated", and it is removed.

### CDISC

- **Items:** each term and each codelist, from the rows of `terms`. The file
  has a row for each term in each codelist, so a term in several codelists
  has several rows. These rows are merged into one item, by the term's
  `Code`. For example, the term Dietary Calcium Measurement has rows in four
  codelists, such as Findings About Test Code and Findings About Test Name. A
  codelist has a row of its own, with an empty `Codelist Code`.
- **Canonical:** a term's `NCI Preferred Term`, such as Dietary Calcium
  Measurement, and a codelist's `Codelist Name`.
- **Anchors:** the `CDISC Submission Value` and `CDISC Synonym(s)` of all
  of the term's rows. So Dietary Calcium Measurement has the anchors `DCA`
  and "Dietary Calcium". A `CDISC Definition` that ends with a question mark
  is a questionnaire item's wording, so it is a question text. A note such as
  "(Lifetime)." may follow the question mark.
- **Description:** the first `CDISC Definition`. CDISC notes a definition's
  source in parentheses after its last sentence, as in "A thing. (NCI)". A
  note of six words or fewer is removed as a source. A longer note, such as a
  full citation, is kept.
- **Values:** a codelist's values are its terms' `NCI Preferred Term`s, in
  the file's order. A term has no values.
- **Status:** every item is current.

### NIH CDE

- **Items:** each data element, a row of `elements`.
- **Canonical:** the first `designation` whose `tags` mark a name.
- **Anchors:** the other designations whose `tags` are all on the reader's
  list of name and question tags, such as `Short Name` or
  `Preferred Question Text`. A designation with no tag counts as a name. A
  designation with any other tag is not used. A tag that the reader has not
  seen before is printed, so that someone can review it. Some designations
  start with the word "Deprecated", and it is removed. A designation is a
  question text when a tag names a question, or when it ends with a question
  mark.
- **Description:** the first `definition`. Other definitions give
  `description_description` pairs. The elements whose `steward` is `NCI` copy
  their definitions from caDSR, so they are cleaned in the same way as
  [caDSR](#cadsr)'s.
- **Values:** each value's `value_meaning_name`, or its `permissible_value`
  when it has no meaning. Codes are removed as [Values](#values) says. The
  type is the element's `datatype`.
- **Status:** from `registration_status`. `Retired` elements are retired.

### caDSR

- **Items:** each data element, a row of `elements`.
- **Canonical:** `long_name`. Elements of old software models end it with a
  Java type, as in "Gene Alias Name java.lang.String". The Java type is
  removed, since `datatype` gives the type.
- **Anchors:** the `short_name`, such as `RACE_CAT_TXT`, some alternate
  names, and the question texts:
  - an alternate name's `name` is an anchor when its `type` holds names, in
    words or as codes. The other types hold mostly context names, ids, model
    paths, table names or mixed text;
  - a question text is the `text` of a document whose `type` ends in
    "Question Text", such as the preferred, alternate and application
    standard ones.
- **Description:** `definition`. caDSR builds it from the definitions of the
  element's concepts. It joins them with "_" or ":", as in "Any of the
  rational or irrational numbers._The information contained in a data
  field." Each join becomes a space. A curation mark at the end, such as
  "[Manually-curated]", is removed. A phrase about a Java type is removed
  too, such as "Value Domain for java language String datatype."
- **Values:** each value's `meaning`, or its `value` when it has no meaning.
  Codes are removed as [Values](#values) says. The type is the element's
  `datatype`, such as `CHARACTER`.
- **Status:** from `workflow_status`. `RETIRED ARCHIVED`, `RETIRED PHASED
  OUT`, `RETIRED WITHDRAWN` and `RETIRED DELETED` are retired, and `DRAFT
  NEW` and `DRAFT MOD` are drafts. `RELEASED` and `RELEASED-NON-CMPLNT` are
  current.

### NHANES

- **Items:** each variable of each data file: a `name` in a `data_file` of a
  `cycle`.
- **Canonical:** the codebook's `english_text`: the question asked, without
  its number on the form, or what the variable measures. When it is empty,
  the `sas_label` is used, or else the variable list's `description`.
- **Anchors:** the variable's `name`, such as `RIDAGEYR`, its `sas_label` and
  its `description`.
- **Description:** none.
- **Values:** the `label` of each row of the codebook's value table. "Range
  of Values", "Value was recorded" and "Missing" are left out: they describe
  the codebook's rows, not a value.
- **Status:** every item is current.

### PhenX

- **Items:** the `variables` whose protocol is visible (`protocol_visible`)
  and available (`open`). A hidden protocol is an older version of a visible
  one. A variable is left out too when the export does not list its
  protocol.
- **Canonical:** `description`, without the quotes around some of them.
- **Anchors:** the variable's `name`, and each dbGaP study variable that
  PhenX maps to it with the `map_level` `comparable`: its
  `gap_variable_name` and its `gap_variable_desc`. Each keeps its studies,
  from `gap_study_accession`, in `study`. Mappings that are only `related` are
  left out.
- **Description:** none.
- **Values:** the labels of `variable_values`, without their codes. So
  `1 = Never` and `Never = 1` both give the value `Never`. The type is the
  variable's own `type`.
- **Status:** every item is current.
