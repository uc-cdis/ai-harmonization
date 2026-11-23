# Sources

`sources fetch` downloads the public sources of the training data, and reads
each source into tables. The next step, [`pairs prepare`](pairs.md), makes the
training pairs from these tables.

| source | `--source` | what it holds | pinned by |
|---|---|---|---|
| [NCIt](#ncit) | `ncit` | biomedical concepts, for example diseases and drugs: synonyms, definitions, and the value lists that registries, for example GDC, attach to a concept | its release, for example `26.09d` |
| [Ontologies](#ontologies) | `obo` | diseases (MONDO), phenotypes (HP), anatomy (Uberon), traits (OBA), measurements (CMO, MMO) and data use (DUO): terms, synonyms and definitions | each ontology's release date. CMO and MMO: their state at the first fetch |
| [LOINC](#loinc) | `loinc` | lab tests and clinical measurements: names, question texts and answer lists | its release, for example `2.83`, and the zip's MD5 |
| [CDISC](#cdisc) | `cdisc` | the controlled terminology of clinical trials: codelists and their terms | each package's release date |
| [NIH CDE](#nih-cde) | `cde` | data elements, the fields that studies collect: names, definitions and values | its state at the first fetch |
| [caDSR](#cadsr) | `cadsr` | NCI's data elements: names, question texts and values | its snapshot date |
| [NHANES](#nhanes) | `nhanes` | a US health survey's variables: questions and value labels | its state at the first fetch |
| [PhenX](#phenx) | `phenx` | the variables of standard study measures, and their mappings to [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) variables | its release, for example `Ver 49.0` |

Each source has a committed manifest,
`embedding_model/sources/manifests/<source>.json`. It pins each release that
a fetch reads. For a source that serves only its current state, it records
the first fetch. A later fetch must then build the same tables. A fetch on
another machine reads the current state again. If the publisher changed it,
`sources fetch` refuses the fetch: then
[make a new version](manifests.md#making-a-new-version).
[How a build is checked](manifests.md#how-a-build-is-checked) says how a fetch
is compared. [licenses.md](licenses.md) gives each source's license and reference.

## Running `sources fetch`

```
python -m embedding_model sources fetch                    # every source
python -m embedding_model sources fetch --source ncit obo  # some of them
```

- **`--source`** takes the ids in the table above.
- **LOINC** needs a free loinc.org account. [LOINC](#loinc) gives the two
  ways to download it.
- **NCIt** takes a few minutes and several gigabytes of memory to read.
- **If a source fails**, `sources fetch` prints the error and fetches the
  other sources. Then it exits with status 1.
- **`--update-manifest`** records a fetch that differs from its committed
  manifest as a new version.
  [Making a new version](manifests.md#making-a-new-version) says when. It
  writes the committed manifest in `embedding_model/sources/manifests/`, also
  with `--inputs`.

## What `sources fetch` writes

`sources fetch` writes to `embedding_model/inputs/sources/`.

```
sources/
    downloads/<source>/   the files as downloaded, kept for the next fetch
    <source>/             the source's tables, as <table>.parquet, and manifest.json
```

### `<source>/<table>.parquet`

Each source has a main table of its items, for example NCIt's `concepts` or
caDSR's `elements`. The main table has an `open` column. It is `true` when
you may use the item, by
[the source's rule for items that are not open](licenses.md#restricted-items).
In a table without an
`open` column, a row is open when its item is open.

### `<source>/manifest.json`

The fetch's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: each table, with its row count and its
  [two checksums](manifests.md#two-checksums)
- `sources`: each file or address that the fetch read, with its `role`. A
  `release` is a file that the manifest pins by its version. A `service` is
  an address that serves only its current state.
- `counts`: what the fetch prints, for example the items read and how many
  are not open

## Details

To use a newer release of a source, do what its **Newer releases** says.
Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says.

A **Newer releases** part names an entry of the committed manifest, a kept
download, or both. Find an entry by its `name` in the `sources` list of
`embedding_model/sources/manifests/<source>.json`. The `downloads/` folders
are under `embedding_model/inputs/sources/`.

### NCIt

`sources fetch` reads the
[NCI Thesaurus](https://evsexplore.semantics.cancer.gov/evsexplore/welcome)'s
OWL release from [EVS's archive](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/).
It also reads the same release's history file, for the replacements of
retired concepts.

- **Tables:** `concepts`, `synonyms`, `definitions`, `values` (the registries'
  value lists) and `replacements`.
- **Newer releases:** EVS publishes a release each month. In the archive,
  find the release's folder, for example `26.09d_Release`. Drop `_Release`
  from the folder's name: `26.09d_Release` becomes `26.09d`. Write that as
  the `version` of the entry `NCIt OWL release`. The fetch then reads the
  history file of the same release.

### Ontologies

`sources fetch` reads seven [OBO Foundry](https://obofoundry.org/) ontologies:
[MONDO](https://obofoundry.org/ontology/mondo.html),
[HP](https://obofoundry.org/ontology/hp.html),
[Uberon](https://obofoundry.org/ontology/uberon.html),
[OBA](https://obofoundry.org/ontology/oba.html),
[CMO](https://obofoundry.org/ontology/cmo.html),
[MMO](https://obofoundry.org/ontology/mmo.html) and
[DUO](https://obofoundry.org/ontology/duo.html).

- **Tables:** `terms`, `synonyms` and `definitions`.
- **Newer releases:** an ontology's undated address, for example
  `https://purl.obolibrary.org/obo/mondo.owl`, serves its newest release.
  The file's first lines name the release in its version IRI, the address of
  that release, for example `.../mondo/releases/2026-10-06/mondo.owl`. Write
  that date as the `version` of the ontology's entry, for example
  `MONDO release`. CMO and MMO have no dated releases: move
  `downloads/obo/cmo.owl` and `downloads/obo/mmo.owl` aside, for example by
  adding `.old` to their names.

### LOINC

There are two ways to download LOINC's release zip:

- Set the environment variables `LOINC_USERNAME` and `LOINC_PASSWORD`, and
  the fetch downloads the zip through LOINC's API.
- Download `Loinc_<version>.zip` from [loinc.org](https://loinc.org/downloads/)
  into `embedding_model/inputs/sources/downloads/loinc/`.

The zip must have the MD5 that the committed manifest pins.

- **Tables:** `codes`, `map_to` (the replacements of deprecated codes),
  `forms` (panels and forms), `answers` and `answer_links`.
- **Newer releases:** Regenstrief, LOINC's publisher, publishes a release
  twice a year. Write its number as the `version` of the entry
  `LOINC release`, and delete that entry's `md5`. The fetch writes the new
  release's MD5.

### CDISC

`sources fetch` reads CDISC's
[controlled terminology](https://www.cdisc.org/standards/terminology/controlled-terminology),
as [NCI EVS publishes it](https://www.cancer.gov/about-nci/organization/cbiit/data-standards/cdisc),
in four packages: SDTM, ADaM, Protocol and the Glossary.

- **Tables:** `terms`, one row per codelist or term.
- **Newer releases:** each package's `Archive/` folder in
  [EVS's CDISC folder](https://evs.nci.nih.gov/ftp1/CDISC/) lists its
  releases by date, for example `ADaM/Archive/`. SDTM's releases are in
  `SDTM/Archive/SDTM/`. Write the date as the `version` of the package's
  entry, for example `CDISC ADaM release`. A package without a new release
  keeps its pin.

### NIH CDE

`sources fetch` reads the
[NIH Common Data Elements Repository](https://cde.nlm.nih.gov/)'s export.

- **Tables:** `elements`, `designations` (names), `definitions`, `ids`,
  `values` and `properties`.
- **Newer releases:** the repository serves only its current state. Move
  `downloads/cde/SearchExport.json` aside, for example by adding `.old` to
  its name.

### caDSR

The [Cancer Data Standards Registry and Repository](https://cadsr.cancer.gov/)
publishes a daily snapshot of its data elements. `sources fetch` reads the
snapshot of the date that the manifest pins.

- **Tables:** `elements`, `alternate_names`, `documents` (for example
  question texts) and `values`.
- **Newer releases:** caDSR keeps its snapshots in the `CDE/XML_Archive/`
  folder of its download area, each named by its date. The `location` of the
  entry `caDSR released data elements` ends in that date, for example
  `releasedCDEsXML-OD.zip.2026-07-01`. Write the newer date as the entry's
  `version`.

### NHANES

`sources fetch` reads the
[National Health and Nutrition Examination Survey](https://wwwn.cdc.gov/nchs/nhanes/)'s
web pages: the variable list of each component, and the codebook of each data
file.

- **Tables:** `variables`, `codebook` and `values`.
- **Missing codebooks:** some data files have no public codebook. The fetch
  lists them in `downloads/nhanes/codebooks/not_found.txt`, and skips them.
- **If a page cannot be downloaded for another reason**, for example after a
  lost connection, the
  NHANES fetch stops. `sources fetch` goes on with the other sources. Then
  run `python -m embedding_model sources fetch --source nhanes` again. It
  reuses the pages that it kept, skips the data files in `not_found.txt`, and
  downloads the rest.
- **Newer releases:** NHANES serves only its current pages. Move
  `downloads/nhanes/` aside, for example to `downloads/nhanes.old/`.

### PhenX

`sources fetch` reads [PhenX](https://www.phenxtoolkit.org/)'s database
export, `Phenx_DB.zip`.

- **Tables:** `variables`, `protocols`, `mappings` (to dbGaP variables),
  `replacements` and `release`.
- **Newer releases:** the export's address serves only PhenX's latest
  release. Move `downloads/phenx/Phenx_DB.zip` aside, for example by adding
  `.old` to its name. Write the new release, for example `Ver 50.0`, as the
  `version` of the entry `PhenX database export`. The fetch refuses an export
  of another release.
