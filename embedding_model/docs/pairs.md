# Pairs

The model learns from pairs of texts that mean the same thing. `pairs
prepare` makes them from the [sources](sources.md) that `sources fetch` wrote.

A pair has two sides. Their names come from
[sentence-transformers](https://sbert.net/docs/sentence_transformer/loss_overview.html#loss-table), the library that Hugging Face
maintains and that training uses:

- the **anchor**: a text as a study might word a variable, for example a
  short name or a question
- the **positive**: the same thing as a data model would describe its
  property, with its name, type, description and values

So the anchor plays the source variable of a mapping, and the positive plays
its target. `pairs prepare` writes both sides in `compact`, the
[text format](scoring.md#text-formats) that scoring uses by default. `compact`
writes a description and values only when a side has them:

```
anchor:   SVG (string)
positive: Saphenous Vein Graft (string): A vessel graft comprised of the saphenous vein.
```

`pairs prepare` writes each source's pairs and their
[manifest](manifests.md) to a separate folder.
[What `pairs prepare` writes](#what-pairs-prepare-writes) lists the files.
[Details](#details) describes how an item becomes pairs, the rules that
every source shares, and each source's own rules.

[design_decisions.md](design_decisions.md#pairs) gives the reasons for the
rules below.

## Running `pairs prepare`

```
python -m embedding_model pairs prepare                        # every source
python -m embedding_model pairs prepare --source nhanes phenx  # some of them
```

If a source is not fetched, or its tables changed since its fetch,
`pairs prepare` prints an error. The error gives the `sources fetch` command
to run. `pairs prepare` still prepares the other sources, and then exits with
status 1.

## What `pairs prepare` writes

`pairs prepare` writes each source's pairs to
`embedding_model/inputs/pairs/<source>/`, or under the folder that `--inputs`
names. It writes each folder whole. So if `pairs prepare` fails for a source,
that source's previous pairs stay as they were.
[Inputs and outputs](../README.md#inputs-and-outputs) describes how a step
writes a folder whole.

```
pairs/<source>/
    pairs.parquet    one row per pair
    manifest.json    the pairs' manifest
```

### `pairs.parquet`

One row per pair:

| column | what it holds |
|---|---|
| `anchor` | the source side's text, formatted as the model sees it |
| `positive` | the target side's text, formatted as the model sees it |
| `source` | the source, for example `ncit` |
| `kind` | `name`, `name_description`, `description_description` or `name_to_description` |
| `item` | the source's own id of the item on the positive side |
| `term_type` | what the anchor's text is: the source's own label, for example a synonym type or an alternate-name type, or else its field, for example `short name`. It is `canonical` when the anchor is a canonical, and `description` when the anchor is another description. |
| `term_group` | `expansion` or `paraphrase` (see [Expansions and paraphrases](#expansions-and-paraphrases)) |
| `status` | `current`, `retired` or `draft`: the status of the item that the anchor's text comes from |
| `study` | the [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) studies that the anchor's text comes from, joined by `\|`, if any. Only [PhenX](#phenx)'s pairs fill it. |
| `value_list` | the name of the positive's value list, where a source names its lists |
| `anchor_name`, `anchor_type`, `anchor_description`, `anchor_values` | four columns: the fields that `pairs prepare` writes the anchor's text from |
| `positive_name`, `positive_type`, `positive_description`, `positive_values` | four columns: the fields that `pairs prepare` writes the positive's text from |

`pairs.parquet` keeps each side's fields beside the side's text.
[Leakage prevention](leakage.md#the-fields-compared) compares these fields,
not the formatted texts.

### `manifest.json`

The pairs' manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: the pairs' row count, their file's `sha256` and their table's
  `content_sha256`
- `sources`: the fetch that the pairs come from, with its version and its
  tables' checksums
- `settings`: `format`, the [text format](scoring.md#text-formats) of the
  pairs' texts
- `counts`:
  - the items, and the items not open
  - the pairs by kind, term group and status
  - the duplicates removed
  - the [ambiguous names](#ambiguous-names), as `ambiguous_texts`

## Details

### The shape: a star around each item

`pairs prepare` reads each source into items: a concept, a term, a code, a
data element or a variable. An item has one **canonical** text, the name that
its source prefers, for example a thesaurus's preferred name or an ontology's
label. Every other text of the item that means the same thing becomes one
anchor. Its positive is the canonical, with the item's type, description and
values.

Each item is the centre of a star: each pair joins the canonical to one other
text. For example, `BMI` and "Quetelet Index" are both anchors of the
canonical "Body Mass Index". They are never paired with each other.

There are four kinds of pair:

| kind | anchor | positive |
|---|---|---|
| `name` | another text of the item | the canonical, its type and values |
| `name_description` | another text of the item | the canonical, its type, description and values |
| `description_description` | another description of the item, for example a second definition | the canonical, its type, its first description and values |
| `name_to_description` | the canonical, for an item with no other text | the description alone, with the type and values |

When the item has a description, each other text gives two pairs: a `name`
pair and a `name_description` pair. Another description gives no pair when
its [`text_key`](leakage.md#how-two-texts-match) equals the canonical's or the
first description's.

### The rules every source shares

The same rules apply to every source. [Each source](#each-source) lists
the fields that each source uses for them.

#### Only open items

`pairs prepare` uses an item only when its source's restriction field allows
it. The source's tables mark this in their `open` column (see
[Restricted items](licenses.md#restricted-items)).

#### The canonical

The canonical is the source's own preferred field.

#### Texts of the same meaning

A text is an anchor only when its source marks it as the same thing as the
canonical. So `pairs prepare` leaves out ontology synonyms that are only
related, broader or narrower. It also leaves out [LOINC](#loinc)'s related
names. Each source's section lists the texts that it uses.

#### Names

An anchor must be a name: it has two letters in a row. So ids and bare
numbers, for example `9999901v3.0:9999902v1.0` or `535`, are not anchors.

`pairs prepare` uses each text of an item once as a name, even when the
source lists the text twice. It compares the texts as the tokenizer
[normalises](tokenizer.md#step-1-normalize) them, and drops a repeat. It does
not use the item's canonical as a name either, because the canonical is the
positive. Only a `name_to_description` pair has the canonical as its anchor. For example, `BODY WEIGHT` repeats the canonical `Body Weight`, so
it is not an anchor. But `body_weight` is an anchor, because the tokenizer
reads its `_` as a token of its own.

#### Expansions and paraphrases

Each pair has a `term_group`: `expansion` or `paraphrase`. Only the anchor's
text decides it, so every kind of pair can have either group.

| `term_group` | the anchor | examples: anchor -> canonical |
|---|---|---|
| `expansion` | a short form or a code: one token with a digit, an underscore or two capitals | `BMI` -> Body Mass Index, `IL-17D` -> Interleukin-17D, `RACE_CAT_TXT` -> Race Category Text, `Val-Trp` -> Valyltryptophan |
| `paraphrase` | words, or a phrase | Heart Attack -> Myocardial Infarction, Adalat -> Nifedipine, Pre-Operative -> Preoperative, Galectin-10 -> Eosinophil Lysophospholipase |
| `paraphrase` | a question text, whatever its shape | "Are you pregnant now?" -> Current Pregnancy Indicator, "Date of Birth" -> Person Birth Date |

The examples come from [NCIt](#ncit), [caDSR](#cadsr) and the
[NIH CDE](#nih-cde). Words joined by "-" or "/" are a paraphrase when both
of these hold:

- each part is a number, or a word with no capital after its first letter
- one part has five letters or more, as in `Pre-Operative` and `Galectin-10`

So `HIV-Positive` is an expansion. Other joined words follow the table: `Val-Trp` has two capitals, so it is an
expansion, but `pre-op` is a paraphrase. The rule looks only at shape, so
`Long-Term` is an expansion too.

A question text is the wording that a form shows for an item. Question texts
are anchors because studies often word their variables the same way. Each
source's section names the fields that hold them.

#### Retired items

`pairs prepare` uses retired items too:

- A source can name a replacement for a retired item. The retired item's
  canonical and texts then become anchors of the replacement's canonical.
  When the replacement is retired too, `pairs prepare` uses the replacement's
  own replacement, and so on, up to an item that is not retired.
- A retired item without a replacement keeps its own canonical.
  `pairs prepare` uses its texts only when no current or draft item has them.

`pairs prepare` compares a retired item's texts with the texts of the current
and draft items by their [`text_key`](leakage.md#how-two-texts-match). So a
retired `blood_pressure` matches a current `Blood Pressure`.

`pairs prepare` uses draft items like current items. The `status` column
keeps them apart, so a training run can leave them out.

#### Ambiguous names

A name can stand for several items, as `MS` stands for multiple sclerosis and
for mitral stenosis. The name is then an anchor of each of these items. Each
source's manifest counts such names, as `ambiguous_texts`.

#### Duplicates

`pairs prepare` reads one item per id of a source. For example, it merges a
CDISC term's rows into one item. Two pairs are copies when the
[tokenizer](tokenizer.md) normalises their texts to the same two texts.
`pairs prepare` keeps only the first copy, as it is written. The copy kept
takes the studies of every copy. For example:

| one pair | another pair | copies? |
|---|---|---|
| `BMI` -> `Body Mass Index (float)` | `bmi` -> `body mass index (float)` | yes: only the case differs |
| `Treatment-naïve` -> `Treatment Naive Indicator (string)` | `treatment-naive` -> `Treatment Naive Indicator (string)` | yes: only an accent differs |
| `body_wt` -> `Body Weight (float)` | `body wt` -> `Body Weight (float)` | no: the tokenizer keeps the `_` as a token |

`corpus build` combines the sources into one [corpus](corpus.md). It then
keeps a pair that two sources share only in the earlier source.
[Each source](#each-source) lists the sources in this order, with NCIt first.
When `pairs prepare` prepares several sources, it prints how many pairs each
source shares with an earlier source that it prepared.

#### Values

A positive's values are the item's own value list, in the source's order.
`pairs prepare` never merges lists. Values are labels, not codes: "Female",
not "F". A pair keeps the whole list. The positive's text shows the first 12
values, as `ai_harmonization.formatters` does. Anchors have no values.

Some sources write a code before each label, as in "0= None" or "0 -
absent". For caDSR, the NIH CDE and LOINC, `pairs prepare` removes the codes
only when:

- every value of the list has a code
- no two codes are the same, and no two labels are the same

So "0 - Never | 1 | ... | 10 - Always" keeps its numbers, because "1" has no
code: the numbers are its values.

A code joined by "-" alone, as in "0-Never", also needs a list of three values
or more. So a short list keeps "2-Dimensional" as a name.

An item can have several lists, as an NCIt concept can. Such an item then
gives its pairs once for each distinct list.

#### Types

`pairs prepare` keeps a source's own type word, for example `Value List`.
When a source has none, the type is `string/encoded` for an item with values
and `string` for one without. `ai_harmonization.dbgap` maps
[dbGaP](https://www.ncbi.nlm.nih.gov/gap/)'s types to these same words. An
anchor takes its item's type word, or `string` when there is none.

#### Cleaning

`pairs prepare` cleans every text in the same steps:

- It reads escapes, for example R's `<U+00A0>`, and it unescapes HTML
  entities, for example `&gt;` to `>`. An entity that a source escaped twice,
  as `&amp;gt;`, becomes `>` too.
- It repairs UTF-8 text decoded as cp1252, for example `Iâ€™m` to `I’m`, even
  when the text was decoded twice.
- It normalises Unicode.
- It removes HTML tags. It keeps other text in angle brackets, as in
  "Candida <Debaryomycetaceae> infection".
- It removes the characters that hold no text, as
  [BERT's tokenizer](https://huggingface.co/docs/tokenizers/v0.23.2/en/api/normalizers#tokenizers.normalizers.BertNormalizer)
  removes them: control, format and private-use characters, for example
  zero-width spaces and direction marks.
- It turns each run of whitespace into one space.

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
pass, and `Café` in the second. A third pass leaves `Café` as it is, so
cleaning stops there.

Cleaning keeps accents, because they are part of the text, as in `Café`.
Later steps use the tokenizer's normalisation, which removes accents. They use
it to compare texts, as for [duplicates](#duplicates), and to cut a text into
tokens (see [tokenizer.md](tokenizer.md#step-1-normalize)).

A question text also loses the number that a form gives it, as in "24. What
is your marital status?". Units stay, for example "(mg/dL)".

Some NCIt and caDSR definitions follow a template with the headings
"EXAMPLE(S):", "OTHER NAME(S):" and "NOTE(S):". `pairs prepare` removes a
heading with nothing after it from a description.

NCIt, caDSR and NIH CDE add notices about Executive Order 14168 to some
definitions. One reads: "This content is being reviewed for compliance with
Executive Order 14168 ...". [CDISC](#cdisc) copies NCIt's notices. A notice
says nothing about the concept, so `pairs prepare` removes it from a
description. It drops a description that was only a notice. Some retired NCIt
concepts have such a description.

### Each source

Each source's section below has the same parts: the items, the canonical,
the anchors, the description, the values and the status. A field, for example
`preferred_name`, is a column of a table that `sources fetch` writes.
[sources.md](sources.md) lists each source's tables. Each source has a
reader, `embedding_model/pairs/<source>.py`, that reads the source's tables
into items. If a status is new to the source's reader, `pairs prepare` reads
it as current and prints it.

| source | canonical (the positive's name) | anchors | description | values | retired items replaced by |
|---|---|---|---|---|---|
| [NCIt](#ncit) | the preferred name | synonyms of the same meaning | the first definition, NCI's where there is one | each registry's list | [EVS's history](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/26.09d_Release/ReadMe_history.txt) |
| [Ontologies](#ontologies) | the label | EXACT synonyms | the definition | none | "term replaced by" |
| [LOINC](#loinc) | the long common name | short and display names, question texts | the definition | the default answer list | the MapTo file |
| [CDISC](#cdisc) | a term's NCI Preferred Term, and a codelist's Codelist Name | submission values, synonyms, question texts | the first definition | a codelist's terms | none |
| [NIH CDE](#nih-cde) | the first designation | the other designations | the first definition | the value meanings | none |
| [caDSR](#cadsr) | the long name | the short name, alternate names, question texts | the definition | the value meanings | none |
| [NHANES](#nhanes) | the codebook's English Text | the variable's name and labels | none | the value labels | none |
| [PhenX](#phenx) | the variable's description | the variable's name, and mapped [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) variables | none | the value labels | none |

#### NCIt

- **Items:** each concept, a row of `concepts`.
- **Canonical:** `preferred_name`.
- **Anchors:** the `name` of each synonym whose `term_type` means the same
  thing as the preferred name:
  - NCIt's own synonyms (`SY`) and other sources' preferred terms (`PT`)
  - antiquated terms (`AQ`, `AQS`) and display names (`DN`)
  - abbreviations (`AB`), and brand names (`BR`, `FB`), for example `Adalat`
    for Nifedipine
  - country codes, for example `USA` (`CA2`, `CA3`, `CS`). The number codes
    (`CNU`), for example `250` for France, are not names, so they give no
    pairs.
  - left out: grammar forms (`AD`), headers (`HD`), chemical structures
    (`SN`) and drug study codes (`CN`)
- **Description:** the first `definition`. A `DEFINITION` comes before an
  `ALT_DEFINITION`, and a definition whose `source` is `NCI` comes before
  another source's. Other definitions give `description_description` pairs.
  NCIt can end a definition with its source, for example "(NCI)" or
  "(NCI04)", and `pairs prepare` removes this source.
- **Values:** the `value`s of each registry's list. A concept can have lists
  from several registries, for example GDC, PCDC, CTDC, ICDC and SeroNet.
  `value_list` is the list's `association`, for example `Has_GDC_Value`. A
  concept gives its pairs once for each distinct list. NCIt gives a list no
  order, so `pairs prepare` sorts its values by name.
- **Status:** from `status`. `Retired_Concept` and `Obsolete_Concept` are
  retired, and `Provisional_Concept` and `Concept_Pending_Approval` are
  drafts. The `replacements` table names a retired concept's replacement:
  the concept that a merge record of the history file names.

#### Ontologies

- **Items:** each term of each [ontology](sources.md#ontologies), a row of
  `terms`.
- **Canonical:** `label`.
- **Anchors:** the synonyms whose `scope` is `EXACT`, of every synonym type,
  for example an abbreviation or a layperson's term. `pairs prepare` leaves
  out `RELATED`, `BROAD` and `NARROW` synonyms.
- **Description:** the first `definition`. Other definitions give
  `description_description` pairs.
- **Values:** none.
- **Status:** a term whose `obsolete` is true is retired. The word "obsolete"
  marks its label, synonyms and definitions, and `pairs prepare` removes the
  word. `replaced_by` names its replacement.

#### LOINC

- **Items:** each code, a row of `codes`.
- **Canonical:** `LONG_COMMON_NAME`.
- **Anchors:** `SHORTNAME` and `DisplayName`, and two question texts:
  `SURVEY_QUEST_TEXT`, and the `DisplayNameForForm` of each open row of
  `forms`. `pairs prepare` leaves out `RELATEDNAMES2` and `CONSUMER_NAME`,
  because LOINC says they are not synonyms.
- **Description:** `DefinitionDescription`.
- **Values:** the `DisplayText` of each answer in the code's default answer
  list, in the list's own order. `value_list` names the list by its
  `AnswerListId` and `AnswerListLinkType`, for example `LL9999-1 (PREFERRED)`.
- **Status:** from `STATUS`. `DEPRECATED` and `DISCOURAGED` codes are
  retired, and `TRIAL` codes are drafts. `MAP_TO`, in `map_to`, names a
  retired code's replacement. A deprecated code's names can start with the
  word "Deprecated", and `pairs prepare` removes the word.

#### CDISC

- **Items:** each term and each codelist, from the rows of `terms`. The file
  has a row for each term in each codelist, so a term in several codelists
  has several rows. `pairs prepare` merges these rows into one item, by the
  term's `Code`. For example, the term Dietary Calcium Measurement has rows in
  four codelists, among them Findings About Test Code and Findings About Test
  Name. A codelist has a row of its own, with an empty `Codelist Code`.
- **Canonical:** a term's `NCI Preferred Term`, for example Dietary Calcium
  Measurement, and a codelist's `Codelist Name`.
- **Anchors:** the `CDISC Submission Value` and `CDISC Synonym(s)` of all
  of the term's rows. So Dietary Calcium Measurement has the anchors `DCA`
  and "Dietary Calcium". A `CDISC Definition` that ends with a question mark
  is a questionnaire item's wording, so it is a question text. A note, for
  example "(Lifetime).", may follow the question mark.
- **Description:** the first `CDISC Definition`. CDISC notes a definition's
  source in parentheses after its last sentence, as in "A thing. (NCI)".
  `pairs prepare` removes a note of six words or fewer, as a source. It keeps
  a longer note, for example a full citation.
- **Values:** a codelist's values are its terms' `NCI Preferred Term`s, in
  the file's order. A term has no values.
- **Status:** every item is current.

#### NIH CDE

- **Items:** each data element, a row of `elements`.
- **Canonical:** the first `designation` whose `tags` mark a name.
- **Anchors:** the other designations whose `tags` are all on the reader's
  list of name and question tags, for example `Short Name` or
  `Preferred Question Text`. These rules apply:
  - A designation with no tag counts as a name.
  - `pairs prepare` does not use a designation with any other tag. If the
    reader has not seen a tag before, `pairs prepare` prints the tag, so that
    someone can review it.
  - Some designations start with the word "Deprecated", and `pairs prepare`
    removes the word.
  - A designation is a question text when a tag names a question, or when it
    ends with a question mark.
- **Description:** the first `definition`. Other definitions give
  `description_description` pairs. The elements whose `steward` is `NCI` copy
  their definitions from caDSR, so `pairs prepare` cleans them in the same
  way as [caDSR](#cadsr)'s.
- **Values:** each value's `value_meaning_name`, or its `permissible_value`
  when it has no meaning. `pairs prepare` removes codes as [Values](#values)
  says. The type is the element's `datatype`.
- **Status:** from `registration_status`. `Retired` elements are retired.

#### caDSR

- **Items:** each data element, a row of `elements`.
- **Canonical:** `long_name`. Elements of old software models end it with a
  Java type, as in "Gene Alias Name java.lang.String". `pairs prepare`
  removes the Java type, because `datatype` gives the type.
- **Anchors:** the `short_name`, for example `RACE_CAT_TXT`, some alternate
  names, and the question texts. These rules apply:
  - An alternate name's `name` is an anchor when its `type` holds names, in
    words or as codes. The other types hold mostly context names, ids, model
    paths, table names or mixed text.
  - If the reader has not seen a type before, `pairs prepare` prints the
    type, so that someone can review it.
  - A question text is the `text` of a document whose `type` ends in
    "Question Text", for example the preferred, alternate and application
    standard ones.
- **Description:** `definition`. caDSR builds it from the definitions of the
  element's concepts. It joins them with "_", ":" or "-", as in "Any of the
  rational or irrational numbers._The information contained in a data
  field." `pairs prepare` turns each join into a space, and removes each
  curation mark, for example "[Manually-curated]". It also removes a phrase
  about a Java type, for example "Value Domain for java language String
  datatype."
- **Values:** each value's `meaning`, or its `value` when it has no meaning.
  `pairs prepare` removes codes as [Values](#values) says. The type is the
  element's `datatype`, for example `CHARACTER`.
- **Status:** from `workflow_status`. `RETIRED ARCHIVED`, `RETIRED PHASED
  OUT`, `RETIRED WITHDRAWN` and `RETIRED DELETED` are retired, and `DRAFT
  NEW` and `DRAFT MOD` are drafts. `RELEASED` and `RELEASED-NON-CMPLNT` are
  current.

#### NHANES

- **Items:** each variable of each data file: a `name` in a `data_file` of a
  `cycle`.
- **Canonical:** the codebook's `english_text`: the question asked, without
  its number on the form, or what the variable measures. When it is empty,
  `pairs prepare` uses the `sas_label`, or else the variable list's
  `description`.
- **Anchors:** the variable's `name`, for example `RIDAGEYR`, its
  `sas_label` and its `description`.
- **Description:** none.
- **Values:** the `label` of each row of the codebook's value table.
  `pairs prepare` leaves out "Range of Values", "Value was recorded" and
  "Missing", because they describe the codebook's rows, not a value.
- **Status:** every item is current.

#### PhenX

- **Items:** the `variables` whose protocol is visible (`protocol_visible`)
  and available (`open`). A hidden protocol is an older version of a visible
  one. `pairs prepare` also leaves out a variable when the export does not
  list its protocol.
- **Canonical:** `description`, without the quotes around some of them.
- **Anchors:** the variable's `name`, and each dbGaP study variable that
  PhenX maps to it with the `map_level` `comparable`: its
  `gap_variable_name` and its `gap_variable_desc`. Each keeps its studies,
  from `gap_study_accession`, in `study`. `pairs prepare` leaves out mappings
  that are only `related`. It prints a `map_level` that it has not seen
  before, so that someone can review it, and it leaves out that level's
  mappings too.
- **Description:** none.
- **Values:** the labels of `variable_values`, without their codes. So
  `1 = Never` and `Never = 1` both give the value `Never`. The type is the
  variable's own `type`.
- **Status:** every item is current.
