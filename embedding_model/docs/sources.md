# Sources

The model learns from pairs of texts that mean the same thing, for example a
variable's name and its description. Such texts come from three kinds of
public source:

- Terminologies and ontologies name medical concepts in several ways.
- Collections of standard data elements hold the fields that studies reuse.
- A survey's data dictionary describes the variables that it collected.

`sources fetch` reads eight sources. It downloads each source from its
publisher, and keeps the files on your machine:

| source | what it holds | fetched from | release read |
|---|---|---|---|
| [NCIt](#ncit) | biomedical concepts, for example diseases and drugs: their synonyms, definitions and registries' value lists | [EVS](https://evs.nci.nih.gov/)'s [OWL](https://www.w3.org/OWL/) release archive | the version that the [manifest](manifests.md#the-committed-manifest-and-the-builds-own) pins, for example `26.09d` |
| [Ontologies](#ontologies) | diseases (MONDO), symptoms (HP), anatomy (Uberon), traits (OBA), measurements (CMO, MMO) and data use (DUO): terms, synonyms, definitions | each ontology's OWL release | the release date that the manifest pins (for CMO and MMO, their current state, kept after the first fetch) |
| [LOINC](#loinc) | lab tests and clinical measurements: their names, question texts and answer lists | LOINC's download API, or a zip that you place by hand | the version that the manifest pins, for example `2.83` |
| [CDISC](#cdisc) | the terms that clinical trials report their data with: codelists and their terms | EVS's CDISC archive | each package's release date, as the manifest pins it |
| [NIH CDE](#nih-cde) | data elements, the fields that studies collect: names, definitions and values | the repository's one-call export | its current state, kept after the first fetch |
| [caDSR](#cadsr) | NCI's data elements: names, question texts and values | the daily bulk XML snapshot | the snapshot date that the manifest pins |
| [NHANES](#nhanes) | a US health survey's variables: their questions and value labels | the variable-list and codebook pages | their current state, kept after the first fetch |
| [PhenX](#phenx) | standard study measures' variables, and their mappings to [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) variables | the database export | the latest export, kept after the first fetch. Its release must match the manifest's pin, for example `Ver 49.0`. |

Each source has a
[committed manifest](manifests.md#the-committed-manifest-and-the-builds-own)
that pins what the fetch reads.

`sources fetch` writes each source's tables to a separate folder, and keeps
the source's downloads beside them.
[What `sources fetch` writes](#what-sources-fetch-writes) lists the files.
[Details](#details) links the sources' licenses, and gives each source's
files and tables.

[design_decisions.md](design_decisions.md#sources) gives the reasons for the
fetch's rules.

## Running `sources fetch`

```
python -m embedding_model sources fetch                    # every source
python -m embedding_model sources fetch --source ncit obo  # some of them
```

If `sources fetch` cannot fetch a source, it prints the error and still
fetches the other sources. For example, LOINC needs a loinc.org account.
Without its credentials or a zip that you place by hand, `sources fetch`
cannot fetch LOINC (see [LOINC](#loinc)). The command then exits with status
1, so a script can tell that a source is missing.

The committed manifest holds the
[SHA-256](https://doi.org/10.6028/NIST.FIPS.180-4) of each file read. For
NHANES' codebook pages, it holds one SHA-256 of them all. `sources fetch`
compares each new fetch with the committed manifest.

## What `sources fetch` writes

`sources fetch` writes each source's tables to
`embedding_model/inputs/sources/<source>/`, and its downloads to
`sources/downloads/<source>/`. With `--inputs`, both go under that folder
instead. It writes `<source>/` whole. So if a fetch fails, or the committed
manifest refuses it, the previous tables stay as they were.
[Inputs and outputs](../README.md#inputs-and-outputs) describes how a step
writes a folder whole.

```
sources/
    downloads/<source>/   the files as downloaded, kept for the next fetch
    <source>/             the source's tables, as <table>.parquet, and the fetch's manifest.json
```

### `downloads/<source>/`

The source's files, as downloaded. `sources fetch` keeps the downloads, so a
later fetch reads the same files. Each source's "Newer releases" part below
says how to use a newer release. For a source that serves only its current
state, that means you move its kept files aside, as its part says. If a fetch
then differs from the committed manifest, it needs a
[new version](manifests.md#making-a-new-version).

### `<source>/<table>.parquet`

A fetch reads a source's files into tables. There is one table per kind of
record, for example a concept, a name or a value. Each source has a main
table of its items, for example NCIt's `concepts` or caDSR's `elements`.
Each record links back to its item by the source's own id.
A cell with several values, for example a concept's parents, joins them with
`|`.

Each source's main table has an `open` column. It is `true` when the item may
be used, by the source's rule in
[Restricted items](licenses.md#restricted-items). A few other tables have an
`open` of their own, for example PhenX's `protocols`. In a table without one,
a row counts as open when its item is.

`sources fetch` keeps every item of a source, including retired and draft
items, and items that are not open. The next step, `pairs prepare`, decides
which of them become training pairs (see [pairs.md](pairs.md)).

### `<source>/manifest.json`

The fetch's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: each table, with its row count and two checksums: `sha256`, of its
  file, and `content_sha256`, of its content.
  [Two checksums](manifests.md#two-checksums) says what each one is for.
- `sources`: each file or address read, with its `role`. A `release` is a
  file pinned by its version, and a `service` is an address that serves only
  its current state.
- `counts`: what the fetch prints, for example the items read and how many
  are not open

The committed manifest is
`embedding_model/sources/manifests/<source>.json`. It pins what
`sources fetch` reads the source from. It pins a release by its version and
SHA-256, and a service's download by the date fetched and its SHA-256.
`sources fetch` checks each fetch against the committed manifest.
[Manifests](manifests.md) says how, and how to make a new version with
`sources fetch --update-manifest`.

## Details

### Licenses and references

[licenses.md](licenses.md) gives each source's license or terms, and its
reference. It also lists the fields that mark single items as restricted.

### NCIt

`sources fetch` reads the
[NCI Thesaurus](https://evsexplore.semantics.cancer.gov/evsexplore/welcome)
from its [OWL](https://www.w3.org/OWL/) release in
[EVS's archive](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/), for
example `NCI_Thesaurus/archive/26.09d_Release/Thesaurus_26.09d.OWL.zip`. The
replacements of retired concepts come from the release's history file, in the
same folder: `subset_cumulative_history_26.09d.zip`.

| table | one row per | columns |
|---|---|---|
| `concepts` | concept | `code`, `preferred_name`, `display_names`, `status`, `semantic_types`, `subsets`, `parents`, `open` |
| `synonyms` | synonym | `code`, `name`, `term_type`, `source`, `source_code`, `subsource` |
| `definitions` | definition | `code`, `definition`, `kind` (`DEFINITION` or `ALT_DEFINITION`), `source` |
| `values` | value of a registry's value list | `code`, `association` (for example `Has_GDC_Value`), `value_code`, `value` |
| `replacements` | merge or retire record of a retired concept, from the history file | `code`, `action` (`merge` or `retire`), `date`, `replacement_code`, `replacement_name` |

The tables link by the concept's `code`. A value's `value_code`, a
concept's `parents` and a `replacement_code` are concept codes too.

`sources fetch` keeps retired and obsolete concepts, with their status. The
history file holds a record for many of them, but not for all. A merge record
names the concept that a retired concept was merged into. That concept is
the retired concept's replacement, in `replacement_code`. A retire record
names the concept's old parent instead, as EVS's
[ReadMe_history.txt](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/26.09d_Release/ReadMe_history.txt)
says. So a retire record gives no replacement.

Reading NCIt takes a few minutes and several gigabytes of memory.

**Newer releases.** EVS publishes a new NCIt release each month. EVS's
`NCI_Thesaurus/archive/` folder lists every release, for example
`26.09d_Release`. To use a newer one, drop `_Release` from its folder name,
for example `26.09d_Release` becomes `26.09d`. Write that as the release's
`version` in the committed manifest. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says.
`sources fetch` reads the history file of the same release with it.

[licenses.md](licenses.md#ncit) gives its license and reference.

### Ontologies

`sources fetch` reads seven [OBO Foundry](https://obofoundry.org/)
ontologies. Ontologies reuse each other's terms: HP, for example, uses
Uberon's terms for body parts. `sources fetch` reads only each ontology's own
terms. So it reads each term once, from the ontology that the term belongs
to.

| table | one row per | columns |
|---|---|---|
| `terms` | term | `id`, `ontology`, `label`, `obsolete`, `replaced_by`, `consider`, `open` |
| `synonyms` | synonym | `id`, `ontology`, `name`, `scope` (`EXACT`, `BROAD`, `NARROW`, `RELATED`), `synonym_types` |
| `definitions` | definition | `id`, `ontology`, `definition` |

The tables link by the term's `id`. A term's `replaced_by` can be a term of
another ontology, for example a cell type in the Cell Ontology.

`sources fetch` keeps obsolete terms, with the terms that replace them. Each
synonym keeps its scope, for example `EXACT` or `RELATED`, and its types, for
example abbreviation.

#### Files

| ontology | file read |
|---|---|
| [MONDO](https://obofoundry.org/ontology/mondo.html), diseases | base release, for example `mondo/releases/<date>/mondo-base.owl` |
| [HP](https://obofoundry.org/ontology/hp.html), phenotypes | base release |
| [Uberon](https://obofoundry.org/ontology/uberon.html), anatomy | base release |
| [OBA](https://obofoundry.org/ontology/oba.html), biological attributes | base release |
| [CMO](https://obofoundry.org/ontology/cmo.html), clinical measurements | main file: no dated file can be downloaded, so the fetch keeps its current state |
| [MMO](https://obofoundry.org/ontology/mmo.html), measurement methods | main file, as for CMO |
| [DUO](https://obofoundry.org/ontology/duo.html), data use | main release |

**Newer releases.** Each ontology's undated address, for example
`https://purl.obolibrary.org/obo/mondo.owl`, serves its newest release. The
file's header names that release in its version IRI (Internationalized
Resource Identifier), for example `.../mondo/releases/2026-10-06/mondo.owl`.
The first few kilobytes of the file hold the version IRI. A date newer than
the pin means a newer release: write that date as the release's `version`.
Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says.

CMO and MMO have no dated releases. The fetch keeps their current files. To
fetch the newest files, move `downloads/obo/cmo.owl` and
`downloads/obo/mmo.owl` aside, for example by adding `.old` to their names.

[licenses.md](licenses.md#ontologies) gives its license and reference.

### LOINC

LOINC's release zip needs a free loinc.org account. There are two ways to
download it:

- Set `LOINC_USERNAME` and `LOINC_PASSWORD`, and the fetch downloads it
  through LOINC's API.
- Download `Loinc_<version>.zip` from [loinc.org](https://loinc.org/downloads/)
  into `inputs/sources/downloads/loinc/`.

Either way, the zip must have the [MD5](https://doi.org/10.17487/RFC1321)
checksum that the manifest pins. A download must also have the MD5 that LOINC
publishes for the release.

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

`sources fetch` keeps codes of every status: active, trial, discouraged and
deprecated. The fetch does not read the answer file's
[SNOMED CT](https://www.snomed.org/) codes and names, because LOINC's license
does not cover them. The fetch knows every column of the pinned release, read
or not. It prints a column that it does not know, so that someone can review
it.

**Newer releases.** Regenstrief publishes a numbered LOINC release twice a
year, and loinc.org lists them. To use a newer one, write its number as the
release's `version`. Also delete the release's `md5`, because the old checksum
belongs to the old release. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. The fetch
writes the new release's MD5 into the manifest.

[licenses.md](licenses.md#loinc) gives its license and reference.

### CDISC

`sources fetch` reads CDISC's
[controlled terminology](https://www.cdisc.org/standards/terminology/controlled-terminology)
as
[NCI EVS publishes it](https://www.cancer.gov/about-nci/organization/cbiit/data-standards/cdisc),
in four packages:

- [SDTM](https://www.cdisc.org/standards/foundational/sdtm), the Study Data
  Tabulation Model, for the data that a clinical trial collects
- [ADaM](https://www.cdisc.org/standards/foundational/adam), the Analysis Data
  Model, for the data that a trial's analysis uses
- [Protocol](https://www.cdisc.org/standards/foundational/protocol), for trial
  protocols
- the [Glossary](https://www.cdisc.org/standards/glossary) of clinical
  research terms

`sources fetch` reads each package from EVS's archive at a pinned release
date, for example `CDISC/ADaM/Archive/ADaM Terminology 2026-09-25.txt`. CDISC
retired the Protocol package, so its last release is older than the others.

| table | one row per | columns |
|---|---|---|
| `terms` | codelist, or term of a codelist | `Code`, `Codelist Code`, `Codelist Extensible (Yes/No)`, `Codelist Name`, `CDISC Submission Value`, `CDISC Synonym(s)`, `CDISC Definition`, `NCI Preferred Term`, `package`, `open` |

A term links to its codelist by `package` and `Codelist Code`. A codelist's
own row has an empty `Codelist Code`. Every `Code` is an [NCIt](#ncit)
concept code.

**Newer releases.** EVS publishes a current package's terminology several
times a year. A package's `Archive/` folder on EVS's server, for example
`CDISC/ADaM/Archive/`, lists every release by date. SDTM's releases are in
`CDISC/SDTM/Archive/SDTM/`. To use a newer one, write
its date as the package's release `version`. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. Each package
has its own date, so a package without a new release keeps its pin.

[licenses.md](licenses.md#cdisc) gives its license and reference.

### NIH CDE

The [NIH Common Data Elements Repository](https://cde.nlm.nih.gov/) serves
only its current state. The fetch uses the export that `ai_harmonization.cde`
uses, with retired elements included. The fetch keeps the export only when
the export holds as many elements as the repository's search reports.

The export is one JSON file, with one record per element. An element's lists,
for example its names and values, become separate tables, linked to the
element by `tiny_id`.

An element's `registration_status` says whether it is current or `Retired`.
A current status is, for example, `Qualified` or `Standard`.

| table | one row per | columns |
|---|---|---|
| `elements` | element | `tiny_id`, `steward`, `registration_status`, `administrative_status`, `copyright_status`, `no_render_allowed`, `datatype`, `open` |
| `designations` | name | `tiny_id`, `position`, `designation`, `tags` |
| `definitions` | definition | `tiny_id`, `position`, `definition`, `tags` |
| `ids` | identifier in another system | `tiny_id`, `source` (for example `BRICS Variable Name`), `id`, `version` |
| `values` | permissible value | `tiny_id`, `position`, `permissible_value`, `value_meaning_name`, `value_meaning_code`, `value_meaning_definition`, `code_system_name` |
| `properties` | property | `tiny_id`, `key`, `value` |

Some elements carry a copyright notice among their properties, for example
[LOINC](#loinc)'s. Whether an element is open depends on its
`copyrightStatus` and `noRenderAllowed`, not on its properties.
`no_render_allowed` holds `true` or `false`, as the export writes it, or
nothing where the export leaves it out. Only `false` is open.

**Newer releases.** The repository serves only its current state, so there is
no release to pin. The fetch keeps the export in
`downloads/cde/SearchExport.json`. To read the current state, move that file
aside, for example by adding `.old` to its name, and fetch again. A fetch that
differs from the committed manifest is a new version, as
[Making a new version](manifests.md#making-a-new-version) says.

[licenses.md](licenses.md#nih-cde) gives its license and reference.

### caDSR

The [Cancer Data Standards Registry and Repository](https://cadsr.cancer.gov/)
publishes a daily snapshot of its data elements, as a zip of XML files, for
example `CDE/XML_Archive/releasedCDEsXML-OD.zip.2026-07-01`. A snapshot can
hold an element's older versions. `sources fetch` keeps only each element's
latest version.

| table | one row per | columns |
|---|---|---|
| `elements` | element | `public_id`, `version`, `long_name`, `short_name`, `definition`, `workflow_status`, `registration_status`, `context`, `datatype`, `value_domain_type`, `open` |
| `alternate_names` | alternate name | `public_id`, `version`, `name`, `type`, `context`, `language` |
| `documents` | reference document, for example a question text | `public_id`, `version`, `type`, `name`, `text`, `language` |
| `values` | permissible value | `public_id`, `version`, `position`, `value`, `meaning`, `meaning_description`, `meaning_concepts`, `begin_date`, `end_date` |

The tables link by the element's `public_id` and `version`.

The snapshot also holds retired and draft elements. They are in `elements`
too, marked by their `workflow_status`, for example `RETIRED ARCHIVED` or
`DRAFT NEW`.

**Newer releases.** caDSR publishes snapshots of its released data elements,
each named by its date, in its download area's `CDE/XML_Archive/` folder. To
use a newer one, write its date as the snapshot's `version`. Then make a new
version, as [Making a new version](manifests.md#making-a-new-version) says.

[licenses.md](licenses.md#cadsr) gives its license and reference.

### NHANES

The
[National Health and Nutrition Examination Survey](https://wwwn.cdc.gov/nchs/nhanes/)
has no downloadable variable list. So the fetch reads its web pages: one
variable-list page per component, for example Laboratory, and one codebook
page per data file.

| table | one row per | columns |
|---|---|---|
| `variables` | variable in a data file, from the lists | `cycle`, `name`, `description`, `data_file`, `data_file_description`, `begin_year`, `end_year`, `component`, `use_constraints`, `open` |
| `codebook` | variable in a codebook | `cycle`, `data_file`, `name`, `sas_label`, `english_text`, `english_instructions`, `target`, `hard_edits`, `open` |
| `values` | row of a variable's value table | `cycle`, `data_file`, `name`, `position`, `code`, `label` |

The three tables link by `cycle`, `data_file` and `name`, because a name alone
repeats across cycles and data files.

`sources fetch` keeps the pages. Some data files have no public codebook
page, for example data files restricted to the Research Data Center. For such
a data file, the server returns "not found" (HTTP 404). The fetch lists these
data files in `downloads/nhanes/codebooks/not_found.txt`, so a later fetch
does not ask for them again. Any other failure to download a page, for
example a lost connection, stops the fetch. Run the fetch again: it downloads
only the pages still missing.

**Newer releases.** NHANES serves only its current pages: the variable lists
and each data file's codebook. The fetch keeps every page in
`downloads/nhanes/`. To read the current pages, move that folder aside, for
example to `downloads/nhanes.old/`, and fetch again. The fetch does not ask
again for a data file in `codebooks/not_found.txt` until that list is gone
too. A fetch that differs from the committed manifest is a new version, as
[Making a new version](manifests.md#making-a-new-version) says.

[licenses.md](licenses.md#nhanes) gives its license and reference.

### PhenX

[PhenX](https://www.phenxtoolkit.org/)'s database export holds its
protocols, their variables and their availability. It also maps some of the
variables to [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) variables of real
studies. The fetch downloads the export from
`https://www.phenxtoolkit.org/toolkit_content/documents/data_dictionary/Phenx_DB.zip`.
That address serves only the latest export, so the fetch keeps the zip. A new
PhenX release needs a new version of the manifest.

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

Some variables belong to a protocol that is not in `protocols`, for example
one of the COVID-19 library's. Such a variable has no availability, so it is
not open. The fetch prints how many there are.

**Newer releases.** The export's address serves only PhenX's latest release,
and the export names that release, for example `Ver 49.0`. The fetch keeps
the export in `downloads/phenx/Phenx_DB.zip`. To use a newer release, move
that file aside, for example by adding `.old` to its name. Write the new
release as the export's `version`. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. The fetch
refuses an export of a release other than the pin.

[licenses.md](licenses.md#phenx) gives its license and reference.
