# Benchmarks

Two benchmarks of expert mappings from study variables to data-model
properties:

- [TOPMed](#topmed): models are developed on it.
- [GDC](#gdc): held back, and scored once on each series' pick.

Both are built on your machine from their published inputs. Nothing processed
is redistributed.

```
python -m embedding_model benchmarks build --topmed
python -m embedding_model benchmarks build --gdc
```

The first build downloads the inputs into
`embedding_model/inputs/benchmarks/downloads/<benchmark>/`. It takes a few
minutes, most of it for dbGaP's variable summaries for TOPMed. A rebuild reuses
the downloads and takes about half a minute. `--inputs DIR` puts everything
that a build writes under `DIR`, in place of `embedding_model/inputs/`; see
[Inputs and outputs](../README.md#inputs-and-outputs).

Every build is checked against the benchmark's committed manifest; see
[The manifest](#the-manifest). To build a benchmark from updated sources, see
[Making a new version](manifests.md#making-a-new-version). An updated source
can be a newer commit of the TOPMed documentation, or a new GDC dictionary.

## How a benchmark is laid out

Each benchmark is two files under `embedding_model/inputs/benchmarks/`:

- **`<benchmark>_mappings.csv`**: one row per source variable, with its
  expected targets: the targets that the experts mapped it to. Any of them
  counts as correct. Each is one mapping. GDC's experts named a property
  without its node. So on GDC, each node that carries the property gives a
  target of its own: see
  [Which targets a variable maps to](#which-targets-a-variable-maps-to).
- **`<benchmark>_targets.csv`**: one row per target property. A format turns
  each row into the text that the model reads on the target side.
  [scoring.md](scoring.md#running-score) lists the formats.

[The columns](#the-columns) lists every column of both files. Both files keep
every value that a variable or property has. The formatters cap how many values
a prompt shows.

**If you load the files in your own code:** one GDC target's values run to
132,058 characters. That is over the 131,072 that Python's `csv` module reads
by default. `pandas.read_csv` reads the files as they are. `csv` needs
`csv.field_size_limit` raised first.

### The columns

Every benchmark writes the same columns. A column that a benchmark has no data
for is left empty.

`<benchmark>_mappings.csv`:

| column | what it holds | TOPMed | GDC |
|---|---|---|---|
| `source_study_id` | the dbGaP study of the variable's cohort, without its version | the variable's study: `phs000956` | the study of the paper's cohort: `phs001287` |
| `source_table_id` | the table that holds the variable | its dbGaP table, at the version cited: `pht005002.v1` | the paper, named as in GDC-SM: `Clark` |
| `source_variable_id` | the variable's own identifier | its dbGaP variable, at the version cited: `phv00252986.v1` | empty: GDC-SM gives none |
| `source_variable_name` | the variable's name in its table | `cac_agatston_score` | `Race` |
| `source_type` | the variable's type, in the source's own words | `float`; also `integer`, `string` or `string/encoded` | `string`; also `number`, from the table's column |
| `source_description` | the variable's description | from its data dictionary: "Coronary artery calcification score" | empty: GDC-SM describes no variable |
| `source_values` | the variable's values, joined by ` \| ` | the value labels that its data dictionary declares | the distinct values in the paper's table |
| `expected_target` | the targets that the experts mapped the variable to, joined by `;` | `atherosclerosis.cac_score_1` | `clinical.race;demographic.race` |

The examples are one row of each benchmark.
[Which target a variable maps to](#which-target-a-variable-maps-to) explains
the TOPMed row.

`<benchmark>_targets.csv`:

| column | what it holds | TOPMed | GDC |
|---|---|---|---|
| `target_property` | the target, as `node.property` | `atherosclerosis.cac_score_1` | `demographic.race` |
| `target_type` | the target's type, in the data model's own words | `float`, `integer` or `string/encoded` | `enum`, `string`, `integer`, `number`, `boolean` or `array` |
| `target_description` | the target's description | from its documentation file | from the dictionary |
| `target_values` | the target's permissible values, joined by ` \| ` | its documented `encoded_values` | the dictionary's permissible values |

A study is written without its version, so that every release of a study
counts as the same study. A table and a variable keep the version that the
experts cited, because their text comes from that release.

### Why the targets are a file of their own

Every source is ranked against every target in the file, not only against
the expected ones. Harmonizing a real study means searching a whole data model.
So the targets file holds the whole candidate set, and most of it is there to
be ranked below the expected targets:

| benchmark | targets | targets that are some variable's expected target |
|---|---|---|
| `topmed` | 148 | 146 |
| `gdc` | 1,217 | 87 |

Without the other targets, the task would shrink to choosing among the
expected targets. A score would then overstate how a model does on a full
schema. The table also explains part of why the two benchmarks score
differently: TOPMed's candidates are nearly all expected targets, and GDC's
mostly are not.

Kept apart, the targets are written once per benchmark rather than once per
mapping. Mappings can also be added without touching them.

Standard retrieval benchmarks such as BEIR and MTEB are laid out the same way.
They have three parts:

- a corpus to search: here, the targets file;
- queries: here, the source variables;
- which corpus entries are relevant to each query: here, `expected_target`.

They keep the queries and their relevant entries in separate files. The
mappings file keeps them together, one row per query.

### The manifest

Each benchmark has a manifest committed as
`embedding_model/benchmarks/manifests/<benchmark>.json`. It pins what the
benchmark is built from, such as the TOPMed documentation's commit or the GDC
ground truth's Zenodo record. Each build writes its own manifest beside the
files, as `<benchmark>_manifest.json`, with the date in `built`. An input's
`role` is `mappings`, `source text` or `targets`. Its `counts` hold what the
build prints:

- TOPMed: the harmonized variables, targets, component references, dbGaP
  releases and source variables, and the source variables mapped to an age;
- GDC: the papers, the source variables and the targets.

[Manifests](manifests.md) says what a manifest holds, how a build is checked
against it, and how to make a new version.

## TOPMed

dbGaP study variables mapped by the experts of the TOPMed Data Coordinating
Center to its harmonized variables. As built here, the benchmark has **1,162
source variables from 29 dbGaP studies, with 2,144 expected mappings, and
148 targets**.

It is built from the Center's harmonized phenotype documentation,
[UW-GAC/topmed-dcc-harmonized-phenotypes](https://github.com/UW-GAC/topmed-dcc-harmonized-phenotypes),
at its last commit (`14960eb`, 2021-10-04). The documentation covers 78
harmonized variables. For each one, it lists the harmonization units, each a
study or a subcohort of one. For each unit, it lists the dbGaP variables that
the unit's harmonization function reads.

Each dbGaP variable is cited with the release that the experts used. Its name,
description and values come from that release's data dictionary. dbGaP
publishes data dictionaries without controlled access. Only the cited tables
are downloaded.

### How the targets are composed

The documentation keeps one file per harmonized variable, in a folder per
dataset: `atherosclerosis/cac_score_1.json`. The harmonized data has a column
for each phenotype. For 70 of the 78, it also has a column for the age at which
the phenotype was measured: `cac_score_1` beside `age_at_cac_score_1`. Each
column is a target, written `node.property` with the dataset as the node:

- the phenotype, `atherosclerosis.cac_score_1` (78 targets), with the type,
  description and values of its documentation file (`data_type`,
  `description`, `encoded_values`);
- the age, `atherosclerosis.age_at_cac_score_1` (70 targets), described "age
  at measurement of cac_score_1 (years)", as the BioData Catalyst data model
  describes the same properties.

### Which target a variable maps to

The documentation lists a unit's variables without saying which is the
phenotype and which is the age. Its README says only that there is "at least
one representing the phenotype itself, and at least one for the age at
measurement". So the builder decides with a rule. A variable maps to the age
target when both of these hold:

- its dbGaP description says "age" as a word, so "Age when CAC score was
  obtained" counts and "Tumor stage" does not;
- the harmonized variable has an age column, as its documentation's
  `has_age_variable` says.

Otherwise it maps to the phenotype target.

For example, the
[Amish unit of `cac_score_1`](https://github.com/UW-GAC/topmed-dcc-harmonized-phenotypes/blob/14960eb088d9057f3fcd04c65f609a31e0d981a1/harmonized-variable-documentation/atherosclerosis/cac_score_1.json#L23-L24)
cites two variables of table `pht005002.v1` in dbGaP study `phs000956.v2`
([the table's data dictionary](https://ftp.ncbi.nlm.nih.gov/dbgap/studies/phs000956/phs000956.v2.p1/pheno_variable_summaries/phs000956.v2.pht005002.v1.TOPMed_WGS_Amish_Subject_Phenotypes.data_dict.xml)):

- `phv00252986.v1`, `cac_agatston_score` ("Coronary artery calcification
  score"), maps to `atherosclerosis.cac_score_1`;
- `phv00252987.v1`, `cac_score_age` ("Age when CAC score was obtained"), maps
  to `atherosclerosis.age_at_cac_score_1`.

The rule is deterministic, and about 2% of variables get a debatable label:

- Dates that feed the age map to the phenotype, for example "DATE OF TEST,
  NUMBER OF DAYS SINCE EXAM 1" and "CT scan date". Helpers that the function
  uses map to the phenotype too, such as a cohort identifier or an attendance
  flag.
- In the incident and prior event variables, the event, its date and the
  follow-up dates feed both the status and the age at event. They map to the
  phenotype.
- An age that is itself the measure maps to the age, for example "Age when
  first smoked cigarettes" for ever smoker.

### One row per variable

- A variable that several harmonized variables use appears once, with all
  its targets. 284 of the 1,162 rows have more than one target.
- A variable can be cited from more than one release. Then the text of the
  latest release cited is used. This happens for 17 variables: their text is
  the same in every release cited, and only the accession's version differs.

Each variable's type comes from its table's variable report.
`ai_harmonization.dbgap` reads it as it reads any study's, so the benchmark
sees the same text that the harmonization approaches see.

[licenses.md](licenses.md#topmed) gives its license and reference.

## GDC

Variables from the supplementary tables of ten cancer papers, mapped by
experts to the GDC data dictionary. As built here, the benchmark has **165
source variables with 298 expected mappings, and 1,217 targets**.

It is built from three published inputs:

- the ground truth of **GDC-SM**, the GDC Schema Matching Benchmark: 259
  mappings from a paper's variable to a GDC property;
- the ten papers' **supplementary tables**, downloaded from their
  publishers, for each variable's type and values. They are prepared as
  GDC-SM's own download script prepares them (`gdc_download.py` in the Zenodo
  record). GDC-SM describes no variable, so the description is empty.
- the live **GDC submission dictionary**, for the targets.

Each variable's study is the dbGaP study of its paper's cohort:

| GDC project | dbGaP study | papers |
|---|---|---|
| CPTAC-2 | `phs000892` | Krug, McDermott, Vasaikar |
| CPTAC-3 | `phs001287` | Cao, Clark, Dou, Gillette, Huang, Satpathy, Wang |

The patients in each paper's table are all cases of that GDC project. This was
checked through the GDC API on 2026-10-22, by looking up the table's case ids.

### The targets

Every data property of the dictionary is a target, as `node.property`, with its
type, description and values. `SimpleDataModel.from_gdc_model` reads the
dictionary as the harmonization approaches read it, so a target's text is the
text that they build. These properties are not targets:

- link properties, which point to other nodes rather than hold data;
- the bookkeeping fields that each node lists as `systemProperties`, such as
  `id`, `created_datetime` and `state`;
- every node's `type`, which holds the node's own name;
- the JSON meta-schema and the root, which the API serves beside the nodes.

The GDC API serves only the current dictionary. So the manifest records its
data release and the SHA-256 of its content, with keys sorted. A build on a
changed dictionary then shows as a different benchmark. After GDC publishes a
new dictionary, a build reports the difference and keeps the files that it had.
To build from the new dictionary, raise the benchmark's version; see
[Making a new version](manifests.md#making-a-new-version). Version 2's
dictionary is gdcdictionary release
[4.0.3](https://github.com/NCI-GDC/gdcdictionary/releases/tag/4.0.3) (commit
`bd56b99`). Its schemas, with their `$ref`s resolved as gdcdictionary's own
loader resolves them, are identical to what the API served (checked
2026-10-14). The dictionary that version 2 was built from on 2026-10-22 has the
same SHA-256 as the dictionary checked then.

### Which targets a variable maps to

The ground truth names a property without its node: `race`, not
`demographic.race`. The targets, though, are `node.property`. Where two nodes
carry the property and mean the same thing by it, both are expected targets:
`race` in `clinical` and in `demographic`, say. Where one of them gives the
name another meaning, that node's property is not an expected target:

| not an expected target | why |
|---|---|
| `project.primary_site` | describes the project, not a patient's tumor site |
| `project.disease_type` | describes the project, not a patient's disease |
| `portion.weight` | a sample portion's weight in milligrams, not a patient's |

The GDC benchmark built by
`jupyter/harmonization_real_benchmark_creation_gdc.ipynb`
(`harmonization_benchmark_real_GDC_v0.0.1`) maps each property to one node
only, so its scores and these are not comparable.

[licenses.md](licenses.md#gdc) gives its license and reference.
