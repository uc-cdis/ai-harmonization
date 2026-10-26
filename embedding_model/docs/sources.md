# Sources

The model learns from pairs of texts that mean the same thing, such as a
variable's name and its description. Such texts come from two kinds of
public source. Vocabularies name medical concepts in several ways. Data
dictionaries describe the variables of real studies. Eight sources are used,
and each is fetched from its publisher and kept on your machine:

```
python -m embedding_model sources fetch                    # every source
python -m embedding_model sources fetch --source ncit obo  # some of them
```

| source | what it holds | fetched from | release read |
|---|---|---|---|
| [NCIt](#ncit) | biomedical concepts, such as diseases and drugs: their synonyms, definitions and registries' value lists | EVS's OWL release archive | the version that the manifest pins, such as `26.09d` |
| [Ontologies](#ontologies) | diseases (MONDO), symptoms (HP), anatomy (Uberon), traits (OBA), measurements (CMO, MMO) and data use (DUO): terms, synonyms, definitions | each ontology's OWL release | the release date that the manifest pins; for CMO and MMO, their current state, kept after the first fetch |
| [LOINC](#loinc) | lab tests and clinical measurements: their names, question texts and answer lists | LOINC's download API, or a zip placed by hand | the version that the manifest pins, such as `2.83` |
| [CDISC](#cdisc) | the terms that clinical trials report their data with: codelists and their terms | EVS's CDISC archive | each package's release date, as the manifest pins it |
| [NIH CDE](#nih-cde) | data elements, the fields that studies collect: names, definitions and values | the repository's one-call export | its current state, kept after the first fetch |
| [caDSR](#cadsr) | NCI's data elements: names, question texts and values | the daily bulk XML snapshot | the snapshot date that the manifest pins |
| [NHANES](#nhanes) | a US health survey's variables: their questions and value labels | the variable-list and codebook pages | their current state, kept after the first fetch |
| [PhenX](#phenx) | standard study measures' variables, and their mappings to dbGaP variables | the database export | the latest export, kept after the first fetch; its release must match the manifest's pin, such as `Ver 49.0` |

The manifest records the SHA-256 of every file read, and a new fetch is
compared with it.

## What a fetch writes

A source's files go under `embedding_model/inputs/sources/`, or under the
folder that `--inputs` names:

```
sources/
    downloads/<source>/   the files as downloaded, kept for the next fetch
    <source>/             the source's tables, as <table>.parquet, and the fetch's manifest.json
```

A fetch downloads a source's files, then reads them into tables. There is one
table per kind of record, such as a concept, a name or a value. Each record
links back to its item by the source's own id.

Every item of a source is kept, including retired, draft and closed ones. The
next step, `pairs prepare`, decides which of them become training pairs (see
[pairs.md](pairs.md)). A cell with several values, such as a concept's parents,
joins them with `|`.

Each source's main table has an `open` column, set from the source's
restriction field (see [Restricted items](licenses.md#restricted-items)). It is
`true` when the item may be used, and `false` when its restriction field
restricts it. A table without `open` takes it from the item that it links to.

Each source has a manifest committed as
`embedding_model/sources/manifests/<source>.json`. It pins the release that the
source is read from. Each fetch writes its own manifest beside the tables, as
`sources/<source>/manifest.json`, with the date in `fetched`. Its `counts`
hold what the fetch prints, such as the items read and how many are not open.
A table's checksum, `csv_sha256`, is the SHA-256 of its content written as
CSV, not of the parquet file. A parquet file's bytes change with the version of the library
that writes it, even when the data are the same.

An input's `role` is `release` for a file pinned by its version, or `service`
for an address that serves only its current state. [Manifests](manifests.md)
says what a manifest holds, how a fetch is checked against it, and how to
make a new version with `sources fetch --update-manifest`.

A source that cannot be fetched is reported, and the other sources are still
fetched. LOINC, for one, needs a loinc.org account: without its credentials or
a zip placed by hand, it cannot be fetched (see [LOINC](#loinc)). The command
then exits with status 1, so a script can tell that a source is missing.

Downloads are kept, so a later fetch reads the same files. Each source's
"Newer releases" part below says how to use a newer release. For a source
that serves only its current state, that means deleting its kept files under
`downloads/`. The new fetch then differs, and is accepted as a
[new version](manifests.md#making-a-new-version).

## Licenses and references

[licenses.md](licenses.md) gives each source's license or terms, and its
reference. It also lists the fields that mark single items as restricted.

## NCIt

The NCI Thesaurus is read from its OWL release in EVS's archive, such as
`NCI_Thesaurus/archive/26.09d_Release/Thesaurus_26.09d.OWL.zip`.

| table | one row per | columns |
|---|---|---|
| `concepts` | concept | `code`, `preferred_name`, `display_names`, `status`, `semantic_types`, `subsets`, `parents`, `open` |
| `synonyms` | synonym | `code`, `name`, `term_type`, `source`, `source_code`, `subsource` |
| `definitions` | definition | `code`, `definition`, `kind` (`DEFINITION` or `ALT_DEFINITION`), `source` |
| `values` | value of a registry's value list | `code`, `association` (such as `Has_GDC_Value`), `value_code`, `value` |
| `replacements` | record of what happened to a retired concept | `code`, `action` (such as `merge` or `retire`), `date`, `replacement_code`, `replacement_name` |

The tables link by the concept's `code`. A value's `value_code`, a
concept's `parents` and a `replacement_code` are concept codes too.

Retired and obsolete concepts are kept, with their status. EVS's replacements
service says what happened to each one, such as a merge, and which concept
replaced it: that is the `replacements` table.

Reading NCIt takes a few minutes and about 4 GB of memory.

**Newer releases.** EVS publishes a new NCIt release each month. EVS's
`NCI_Thesaurus/archive/` folder lists every release, such as `26.09d_Release`.
To use a newer one, write its name as the release's `version` in the committed
manifest. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. The
replacements service serves only its current state. The fetch keeps its answer,
so delete `downloads/ncit/replacements_<version>.json` to ask again.

[licenses.md](licenses.md#ncit) gives its license and reference.

## Ontologies

Seven OBO Foundry ontologies are read. Ontologies reuse each other's terms: HP,
for example, uses Uberon's terms for body parts. Only each ontology's own terms
are read, so each term is read once, from the ontology it belongs to.

| table | one row per | columns |
|---|---|---|
| `terms` | term | `id`, `ontology`, `label`, `obsolete`, `replaced_by`, `consider`, `open` |
| `synonyms` | synonym | `id`, `ontology`, `name`, `scope` (`EXACT`, `BROAD`, `NARROW`, `RELATED`), `synonym_types` |
| `definitions` | definition | `id`, `ontology`, `definition` |

The tables link by the term's `id`. A term's `replaced_by` can be a term of
another ontology, such as a cell type in the Cell Ontology.

Obsolete terms are kept, with the terms that replace them. Each synonym
keeps its scope, such as `EXACT` or `RELATED`, and its types, such as
abbreviation.

### Files

| ontology | file read |
|---|---|
| MONDO, diseases | base release, such as `mondo/releases/<date>/mondo-base.owl` |
| HP, phenotypes | base release |
| Uberon, anatomy | base release |
| OBA, biological attributes | base release |
| CMO, clinical measurements | main file: no dated file can be downloaded, so its current state is kept |
| MMO, measurement methods | main file, as for CMO |
| DUO, data use | main release |

**Newer releases.** Each ontology's undated address, such as
`http://purl.obolibrary.org/obo/mondo.owl`, serves its newest release. The
file's header names that release in its version IRI (Internationalized Resource
Identifier), such as `.../mondo/releases/2026-10-06/mondo.owl`. The first few
kilobytes of the file hold it. A date newer than the pin means a newer release:
write it as the release's `version`. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. CMO and MMO
have no dated releases. The fetch keeps their current files, so delete
`downloads/obo/cmo.owl` and `downloads/obo/mmo.owl` to fetch the newest ones.

[licenses.md](licenses.md#ontologies) gives its license and reference.

## LOINC

LOINC's release zip needs a free loinc.org account. There are two ways to
get it:

- Set `LOINC_USERNAME` and `LOINC_PASSWORD`, and the fetch downloads it
  through LOINC's API.
- Download `Loinc_<version>.zip` from [loinc.org](https://loinc.org/downloads/)
  into `inputs/sources/downloads/loinc/`.

Either way, the zip must have the MD5 checksum that the manifest pins. A
download must also have the MD5 that LOINC publishes for the release.

| table | one row per | columns |
|---|---|---|
| `codes` | code, from `Loinc.csv` | `LOINC_NUM`, `COMPONENT`, `PROPERTY`, `TIME_ASPCT`, `SYSTEM`, `SCALE_TYP`, `METHOD_TYP`, `CLASS`, `CLASSTYPE`, `STATUS`, `LONG_COMMON_NAME`, `SHORTNAME`, `DisplayName`, `CONSUMER_NAME`, `RELATEDNAMES2`, `DefinitionDescription`, `SURVEY_QUEST_TEXT`, `SURVEY_QUEST_SRC`, `EXMPL_ANSWERS`, `EXAMPLE_UNITS`, `ORDER_OBS`, `EXTERNAL_COPYRIGHT_NOTICE`, `EXTERNAL_COPYRIGHT_LINK`, `open` |
| `map_to` | deprecated code and its replacement, from `MapTo.csv` | `LOINC`, `MAP_TO`, `COMMENT` |
| `forms` | row of a panel or form, from `PanelsAndForms.csv` | `ID`, `ParentId`, `ParentLoinc`, `SEQUENCE`, `Loinc`, `DisplayNameForForm`, `EXTERNAL_COPYRIGHT_NOTICE`, `AdditionalCopyright`, `open` |
| `answers` | answer of an answer list, from `AnswerList.csv` | `AnswerListId`, `AnswerListName`, `AnswerStringId`, `LocalAnswerCode`, `SequenceNumber`, `DisplayText`, `Description` |
| `answer_links` | link from a code to an answer list, from `LoincAnswerListLink.csv` | `LoincNumber`, `AnswerListId`, `AnswerListLinkType`, `ApplicableContext` |

The tables link by the LOINC code, named `LOINC_NUM`, `Loinc` or
`LoincNumber`. A form's rows nest by `ID` and `ParentId`, and answer lists
link by `AnswerListId`. An answer list has no restriction field of its own:
it takes the `open` of the code that uses it.

Codes of every status are kept: active, trial, discouraged and deprecated.
The answer file's SNOMED CT codes and names are not read: LOINC's license
does not cover them.

**Newer releases.** Regenstrief publishes a numbered LOINC release twice a
year, and loinc.org lists them. To use a newer one, write its number as the
release's `version`, and delete the release's `md5`: the old checksum belongs
to the old release. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. The fetch
records the new release's MD5.

[licenses.md](licenses.md#loinc) gives its license and reference.

## CDISC

CDISC's controlled terminology is read as NCI EVS publishes it, in four
packages:

- SDTM, for the data that a clinical trial collects;
- ADaM, for the data that its analysis uses;
- Protocol, for trial protocols;
- the Glossary of clinical research terms.

Each package is read from EVS's archive at a pinned release date, such as
`CDISC/SDTM/Archive/SDTM Terminology 2026-09-25.txt`. CDISC retired the
Protocol package in March 2026, so its last release is older than the others.

| table | one row per | columns |
|---|---|---|
| `terms` | codelist, or term of a codelist | `Code`, `Codelist Code`, `Codelist Extensible (Yes/No)`, `Codelist Name`, `CDISC Submission Value`, `CDISC Synonym(s)`, `CDISC Definition`, `NCI Preferred Term`, `package`, `open` |

A term links to its codelist by `package` and `Codelist Code`. A codelist's
own row has an empty `Codelist Code`. Every `Code` is an NCIt concept code.

**Newer releases.** EVS publishes each package's terminology several times a
year. A package's `Archive/` folder on EVS's server, such as
`CDISC/SDTM/Archive/`, lists every release by date. To use a newer one, write
its date as the package's release `version`. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. Each package
has its own date, so a package without a new release keeps its pin.

[licenses.md](licenses.md#cdisc) gives its license and reference.

## NIH CDE

The NIH Common Data Elements Repository serves only its current state. The
fetch uses the export that `ai_harmonization.cde` uses, with retired elements
included. The export is kept only when its count equals the total that the
repository's search reports.

The export is one JSON file, with one record per element. An element's lists,
such as its names and values, become tables of their own, linked to it by
`tiny_id`.

Its elements are current, Qualified or Standard, or retired. Their
`registration_status` says which.

| table | one row per | columns |
|---|---|---|
| `elements` | element | `tiny_id`, `steward`, `registration_status`, `administrative_status`, `copyright_status`, `no_render_allowed`, `datatype`, `open` |
| `designations` | name | `tiny_id`, `position`, `designation`, `tags` |
| `definitions` | definition | `tiny_id`, `position`, `definition`, `tags` |
| `ids` | identifier in another system | `tiny_id`, `source` (such as `BRICS Variable Name`), `id`, `version` |
| `values` | permissible value | `tiny_id`, `position`, `permissible_value`, `value_meaning_name`, `value_meaning_code`, `value_meaning_definition`, `code_system_name` |
| `properties` | property | `tiny_id`, `key`, `value` |

Some elements carry a copyright notice among their properties, such as
LOINC's. Whether an element is open depends on its `copyrightStatus` and
`noRenderAllowed`, not on its properties.

**Newer releases.** The repository serves only its current state, so there is
no release to pin. The fetch keeps the export in
`downloads/cde/SearchExport.json`. To read the current state, delete that file
and fetch again. A fetch that differs from the committed manifest is a new
version, as [Making a new version](manifests.md#making-a-new-version) says.

[licenses.md](licenses.md#nih-cde) gives its license and reference.

## caDSR

The Cancer Data Standards Registry and Repository publishes a daily snapshot
of its data elements, as a zip of XML files, such as
`CDE/XML_Archive/releasedCDEsXML-OD.zip.2026-07-01`. A snapshot can hold an
element's older versions: only the element's latest version is kept.

| table | one row per | columns |
|---|---|---|
| `elements` | element | `public_id`, `version`, `long_name`, `short_name`, `definition`, `workflow_status`, `registration_status`, `context`, `datatype`, `value_domain_type`, `open` |
| `alternate_names` | alternate name | `public_id`, `version`, `name`, `type`, `context`, `language` |
| `documents` | reference document, such as a question text | `public_id`, `version`, `type`, `name`, `text`, `language` |
| `values` | permissible value | `public_id`, `version`, `position`, `value`, `meaning`, `meaning_description`, `meaning_concepts`, `begin_date`, `end_date` |

The tables link by the element's `public_id` and `version`.

The snapshot also holds retired and draft elements. They are in `elements`
too, marked by their `workflow_status`, such as `RETIRED ARCHIVED` or
`DRAFT NEW`.

**Newer releases.** caDSR publishes snapshots of its released data elements,
each named by its date, in its download area's `CDE/XML_Archive/` folder. To
use a newer one, write its date as the snapshot's `version`. Then make a new
version, as [Making a new version](manifests.md#making-a-new-version) says.

[licenses.md](licenses.md#cadsr) gives its license and reference.

## NHANES

The National Health and Nutrition Examination Survey has no downloadable
variable list. So the fetch reads its web pages: one variable-list page per
component, such as Laboratory, and one codebook page per data file.

| table | one row per | columns |
|---|---|---|
| `variables` | variable in a data file, from the lists | `cycle`, `name`, `description`, `data_file`, `data_file_description`, `begin_year`, `end_year`, `component`, `use_constraints`, `open` |
| `codebook` | variable in a codebook | `cycle`, `data_file`, `name`, `sas_label`, `english_text`, `english_instructions`, `target`, `hard_edits`, `open` |
| `values` | row of a variable's value table | `cycle`, `data_file`, `name`, `position`, `code`, `label` |

The three tables link by `cycle`, `data_file` and `name`: a name alone
repeats across cycles and data files.

The pages are kept. Some data files, such as those restricted to the Research
Data Center, have no public codebook page. They are listed in
`downloads/nhanes/codebooks/not_found.txt`, so a later fetch does not ask for
them again. If a page fails to download, the fetch stops. Run it again: it
downloads only the pages still missing.

**Newer releases.** NHANES serves only its current pages: the variable lists
and each data file's codebook. The fetch keeps every page in
`downloads/nhanes/`. To read the current pages, delete them and fetch again. A
data file listed in `codebooks/not_found.txt` is not asked for again until that
list is deleted too. A fetch that differs from the committed manifest is a new
version, as [Making a new version](manifests.md#making-a-new-version) says.

[licenses.md](licenses.md#nhanes) gives its license and reference.

## PhenX

PhenX's database export holds its protocols, their variables and their
availability. It also maps each variable to dbGaP variables of real studies.
It is downloaded from
`https://www.phenxtoolkit.org/toolkit_content/documents/data_dictionary/Phenx_DB.zip`.
That address serves only the latest export, so the zip is kept. A new PhenX
release needs a new version of the manifest.

The export is a MySQL dump. The fetch reads it without a database server.

| table | one row per | columns |
|---|---|---|
| `variables` | variable | `id`, `name`, `code`, `description`, `protocol_id`, `type`, `units`, `min`, `max`, `variable_values`, `comment1`, `comment2`, `version_num`, `protocol_visible`, `availability`, `open` |
| `protocols` | protocol | `id`, `code`, `name`, `description`, `status`, `visible`, `version`, `release_date`, `availability`, `open` |
| `mappings` | mapping to a dbGaP variable | `id`, `variable_id`, `variable_name`, `gap_study_accession`, `gap_study_name`, `gap_phv`, `gap_variable_name`, `gap_variable_desc`, `map_level` (`comparable` or `related`) |
| `replacements` | protocol that PhenX replaced | `id`, `protocol_old_id`, `protocol_new_id` |
| `release` | the export's latest release | `name`, `release_date` |

Variables link to their protocol by `protocol_id`, and mappings to their
variable by `variable_id`. A replacement names two protocol `id`s.

A protocol that PhenX replaced by a newer one is hidden: its `visible` is
`0`, and `replacements` names the protocol that replaced it.

**Newer releases.** The export's address serves only PhenX's latest release,
and the export names it, such as `Ver 49.0`. The fetch keeps the export in
`downloads/phenx/Phenx_DB.zip`. To use a newer release, delete that file, write
the new release as the export's `version`. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. A fetch refuses
an export of another release than the pin.

[licenses.md](licenses.md#phenx) gives its license and reference.
