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
| [NCIt](#ncit) | biomedical concepts, such as diseases and drugs: their synonyms, definitions and registries' value lists | EVS's OWL release archive | the version the manifest pins, such as `26.09d` |
| [Ontologies](#ontologies) | diseases (MONDO), symptoms (HP), anatomy (Uberon), traits (OBA), measurements (CMO, MMO) and data use (DUO): terms, synonyms, definitions | each ontology's OWL release | the release date the manifest pins; for CMO and MMO, their current state, kept after the first fetch |
| [LOINC](#loinc) | lab tests and clinical measurements: their names, question texts and answer lists | LOINC's download API, or a zip placed by hand | the version the manifest pins, such as `2.83` |
| [CDISC](#cdisc) | the terms clinical trials report their data with: codelists and their terms | EVS's CDISC archive | each package's release date, as the manifest pins it |
| [NIH CDE](#nih-cde) | data elements, the fields studies collect: names, definitions and values | the repository's one-call export | its current state, kept after the first fetch |
| [caDSR](#cadsr) | NCI's data elements: names, question texts and values | the daily bulk XML snapshot | the snapshot date the manifest pins |
| [NHANES](#nhanes) | a US health survey's variables: their questions and value labels | the variable-list and codebook pages | their current state, kept after the first fetch |
| [PhenX](#phenx) | standard study measures' variables, and their mappings to dbGaP variables | the database export | the latest export, kept after the first fetch; its release must be the one the manifest pins, such as `Ver 49.0` |

Every file read has its SHA-256 recorded in the manifest, and a new fetch is
compared by it.

## What a fetch writes

A source's files go under `embedding_model/inputs/sources/`, or under the
folder `--inputs` names:

```
sources/
    downloads/<source>/   the files as downloaded, kept for the next fetch
    <source>/             the source's tables, as <table>.parquet, and the fetch's manifest.json
```

A fetch downloads a source's files, then reads them into tables. There is one
table per kind of record, such as a concept, a name or a value. The tables
link back to the item they belong to by the source's own id.

Every item a source has is kept, retired, draft and closed ones too. The next
step, `pairs prepare`, decides which of them become training pairs (see
[pairs.md](pairs.md)). A cell with several values, such as a concept's
parents, joins them with `|`.

Each source's main table has an `open` column, set from the source's
restriction field (see [Licenses](#licenses)). It is `true` when the item may
be used, and `false` when its restriction field restricts it. A table without
`open` takes it from the item it links to.

Each source has a manifest committed as
`embedding_model/sources/manifests/<source>.json`. It pins the release each
source is read from. Each fetch writes its own manifest beside the tables, as
`sources/<source>/manifest.json`, with the date in `fetched`. A table's
checksum, `csv_sha256`, is the SHA-256 of its content written as CSV, not of
the parquet file. A parquet file's bytes change with the version of the
library that writes it, even when the data are the same.

An input's `role` is `release` for a file pinned by its version, or `service`
for an address that serves only its current state. [Manifests](manifests.md)
says what a manifest holds, how a fetch is checked against it, and how to
make a new version with `sources fetch --update-manifest`.

A source that cannot be fetched is reported, and the other sources are still
fetched. LOINC, for one, needs a loinc.org account: without its credentials or
a zip placed by hand, it cannot be fetched (see [LOINC](#loinc)). The command
then exits with status 1, so a script can tell that a source is missing.

Downloads are kept, so a later fetch reads the same files. Some sources
serve only their current state. To fetch a newer state of one of them, delete
its files under `downloads/`:

| source | files to delete |
|---|---|
| NIH CDE | `cde/SearchExport.json` |
| NHANES | the whole `nhanes/` folder |
| NCIt's replacements service | `ncit/replacements_<version>.json` |
| CMO and MMO | `obo/cmo.owl` and `obo/mmo.owl` |
| PhenX | `phenx/Phenx_DB.zip` |

The new fetch then differs, and is accepted as a
[new version](manifests.md#making-a-new-version). For PhenX, the manifest's
pinned release must be raised first.

## Licenses

Each source's license was checked. Every source is free to use, under the
license or terms its section names.

Some sources also mark single items as restricted, in a field of their own.
For example, LOINC marks a code whose content belongs to a third party. The
fetch reads these restriction fields too. Each source's main table has an
`open` column: true when the item's restriction field allows its use.

Where the field holds one of a list of values, an item is open only when its
value is known to allow use. A value the fetch has not seen before means the
item is not open. The value is printed, so it can be reviewed. For caDSR, the
field is a document instead. A new kind of document is printed too, but only
a Proprietary Measure Permission closes an element.

| source | restriction field | open when |
|---|---|---|
| NCIt, ontologies, CDISC | none per item | always: the source's license covers every item |
| LOINC | `EXTERNAL_COPYRIGHT_NOTICE`, and a form row's `AdditionalCopyright` | they are empty, on the item and on the panels it sits in. With closed panels left out, a code is open when it sits in an open panel, or in no panel and has no notice. A notice marks third-party content. |
| NIH CDE | `copyrightStatus`, `noRenderAllowed` | `copyrightStatus` is "Public domain, free to use", and `noRenderAllowed` is not true |
| caDSR | a "Proprietary Measure Permission" reference document | the element has none |
| NHANES | "Use Constraints", per data file | it is "None" or "RDC Only". Both restrict access to the data, not the documentation read here. |
| PhenX | the protocol's availability | it is "Available" |

## NCIt

The NCI Thesaurus, read from its OWL release in EVS's archive, such as
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

### License

NCIt is released under CC BY 4.0, as EVS's
[Terms of Use](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.htm)
state.

### Reference

> Sioutos, N., Coronado, S. d., Haber, M. W., Hartel, F. W., Shaiu, W.-L., &
> Wright, L. W. (2007). NCI Thesaurus: A semantic model integrating
> cancer-related clinical and molecular information. *Journal of Biomedical
> Informatics*, *40*(1), 30–43. https://doi.org/10.1016/j.jbi.2006.02.013

## Ontologies

Seven OBO Foundry ontologies. Ontologies reuse each other's terms: HP, for
example, uses Uberon's terms for body parts. Only each ontology's own terms
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

### Files and licenses

Each license is the one the file states.

| ontology | file read | license |
|---|---|---|
| MONDO, diseases | base release, such as `mondo/releases/<date>/mondo-base.owl` | CC BY 4.0 |
| HP, phenotypes | base release | [HPO license](https://hpo.jax.org/app/license) |
| Uberon, anatomy | base release | CC BY 3.0 |
| OBA, biological attributes | base release | CC0 1.0 |
| CMO, clinical measurements | main file: no dated file can be downloaded, so its current state is kept | CC BY 4.0 |
| MMO, measurement methods | main file, as for CMO | CC BY 4.0 |
| DUO, data use | main release | CC BY 4.0 |

### References

> Vasilevsky, N. A., Matentzoglu, N. A., Toro, S., Flack, J. E., Hegde, H.,
> Unni, D. R., Alyea, G. F., Amberger, J. S., Babb, L., Balhoff, J. P.,
> Bingaman, T. I., Burns, G. A., Buske, O. J., Callahan, T. J., Carmody, L.
> C., Cordo, P. C., Chan, L. E., Chang, G. S., Christiaens, S. L., et al.
> (2022). Mondo: Unifying diseases for the world, by the world.
> https://doi.org/10.1101/2022.04.13.22273750

> Gargano, M. A., Matentzoglu, N., Coleman, B., Addo-Lartey, E. B.,
> Anagnostopoulos, A., Anderton, J., Avillach, P., Bagley, A. M., Bakštein,
> E., Balhoff, J. P., Baynam, G., Bello, S. M., Berk, M., Bertram, H., Bishop,
> S., Blau, H., Bodenstein, D. F., Botas, P., Boztug, K., et al. (2023). The
> Human Phenotype Ontology in 2024: phenotypes around the world. *Nucleic
> Acids Research*, *52*(D1), D1333–D1346. https://doi.org/10.1093/nar/gkad1005

> Mungall, C. J., Torniai, C., Gkoutos, G. V., Lewis, S. E., & Haendel, M. A.
> (2012). Uberon, an integrative multi-species anatomy ontology. *Genome
> Biology*, *13*(1). https://doi.org/10.1186/gb-2012-13-1-r5

> Stefancsik, R., Balhoff, J. P., Balk, M. A., Ball, R. L., Bello, S. M.,
> Caron, A. R., Chesler, E. J., de Souza, V., Gehrke, S., Haendel, M., Harris,
> L. W., Harris, N. L., Ibrahim, A., Koehler, S., Matentzoglu, N., McMurry, J.
> A., Mungall, C. J., Munoz-Torres, M. C., Putman, T., et al. (2023). The
> Ontology of Biological Attributes (OBA)—computational traits for the life
> sciences. *Mammalian Genome*, *34*(3), 364–378.
> https://doi.org/10.1007/s00335-023-09992-1

> Smith, J. R., Park, C. A., Nigam, R., Laulederkind, S. J., Hayman, G. T.,
> Wang, S.-J., Lowry, T. F., Petri, V., Pons, J. D., Tutaj, M., Liu, W.,
> Worthey, E. A., Shimoyama, M., & Dwinell, M. R. (2013). The clinical
> measurement, measurement method and experimental condition ontologies:
> expansion, improvements and new applications. *Journal of Biomedical
> Semantics*, *4*(1). https://doi.org/10.1186/2041-1480-4-26

> Lawson, J., Cabili, M. N., Kerry, G., Boughtwood, T., Thorogood, A., Alper,
> P., Bowers, S. R., Boyles, R. R., Brookes, A. J., Brush, M., Burdett, T.,
> Clissold, H., Donnelly, S., Dyke, S. O. M., Freeberg, M. A., Haendel, M. A.,
> Hata, C., Holub, P., Jeanson, F., et al. (2021). The Data Use Ontology to
> streamline responsible access to human biomedical datasets. *Cell
> Genomics*, *1*(2), 100028. https://doi.org/10.1016/j.xgen.2021.100028

## LOINC

LOINC's release zip needs a free loinc.org account. There are two ways to
get it:

- Set `LOINC_USERNAME` and `LOINC_PASSWORD`, and the fetch downloads it
  through LOINC's API.
- Download `Loinc_<version>.zip` from [loinc.org](https://loinc.org/downloads/)
  into `inputs/sources/downloads/loinc/`.

Either way, the zip must have the MD5 checksum the manifest pins. A download
must also have the one LOINC publishes for the release.

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

### License

The [LOINC License](https://loinc.org/license/): free, with
attribution. Content that belongs to a third party carries a notice, and is
not open (see [Licenses](#licenses)).

### Reference

> McDonald, C. J., Huff, S. M., Suico, J. G., Hill, G., Leavelle, D., Aller,
> R., Forrey, A., Mercer, K., DeMoor, G., Hook, J., Williams, W., Case, J.,
> Maloney, P., & Laboratory LOINC Developers. (2003). LOINC, a Universal
> Standard for Identifying Laboratory Observations: A 5-Year Update.
> *Clinical Chemistry*, *49*(4), 624–633. https://doi.org/10.1373/49.4.624

## CDISC

CDISC's controlled terminology, as NCI EVS publishes it, in four packages:

- SDTM, for the data a clinical trial collects;
- ADaM, for the data its analysis uses;
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

### License

NCI states that CDISC Terminology can be used "for free, without licensing
restrictions", on its
[CDISC terminology page](https://www.cancer.gov/research/resources/terminology/cdisc).

### Reference

> CDISC. (n.d.). *Controlled Terminology*. Retrieved October 22, 2026, from
> https://www.cdisc.org/standards/terminology/controlled-terminology

## NIH CDE

The NIH Common Data Elements Repository serves only its current state. The
fetch uses the export `ai_harmonization.cde` uses, with retired elements
included. The export is kept only when its count equals the total the
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

### License

NLM's [web policies](https://www.nlm.nih.gov/web_policies.html) state that
U.S. government works on its sites "may be freely used or reproduced without
permission". Content contributed by others may be copyrighted. Each element's
`copyrightStatus` says which: only "Public domain, free to use" is open.

### Reference

> National Library of Medicine. (n.d.). *NIH Common Data Elements (CDE)
> Repository*. Retrieved October 22, 2026, from https://cde.nlm.nih.gov/

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

### License

NCI states that "unless otherwise indicated, all text within National Cancer
Institute (NCI) products is free of copyright"
([Reuse of NCI Information](https://www.cancer.gov/policies/copyright-reuse)).
An element of a proprietary measure has a Proprietary Measure Permission
document, and is not open.

### Reference

> Covitz, P. A., Hartel, F., Schaefer, C., De Coronado, S., Fragoso, G., Sahni,
> H., Gustafson, S., & Buetow, K. H. (2003). caCORE: A common infrastructure
> for cancer informatics. *Bioinformatics*, *19*(18), 2404–2412.
> https://doi.org/10.1093/bioinformatics/btg335

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

### License

CDC states that most of the information on its websites is in the public
domain. It "may be freely used or reproduced without obtaining copyright
permission"
([Use of Agency Materials](https://www.cdc.gov/other/agencymaterials.html)).

### Reference

> Centers for Disease Control and Prevention, National Center for Health
> Statistics. (n.d.). *National Health and Nutrition Examination Survey*.
> Retrieved October 22, 2026, from https://wwwn.cdc.gov/nchs/nhanes/

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

### License

PhenX's [FAQ](https://www.phenxtoolkit.org/help/faq) states that most of its
protocols are freely available, and a small share have limited availability.
Each protocol's `availability` says which: only "Available" is open.

### Reference

> Hamilton, C. M., Strader, L. C., Pratt, J. G., Maiese, D., Hendershot, T.,
> Kwok, R. K., Hammond, J. A., Huggins, W., Jackman, D., Pan, H., Nettles, D.
> S., Beaty, T. H., Farrer, L. A., Kraft, P., Marazita, M. L., Ordovas, J. M.,
> Pato, C. N., Spitz, M. R., Wagener, D., et al. (2011). The PhenX Toolkit:
> Get the Most From Your Measures. *American Journal of Epidemiology*,
> *174*(3), 253–260. https://doi.org/10.1093/aje/kwr193
