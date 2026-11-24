# Benchmarks

`benchmarks build` builds two benchmarks of expert mappings from study
variables to data-model properties:

- [TOPMed](#topmed): [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) study
  variables, mapped to TOPMed's harmonized variables
- [GDC](#gdc): variables of cancer papers, mapped to the GDC data dictionary

Another benchmark, BDC, is in preparation for publication.

A benchmark lists source variables, each with its expected targets: the
target properties, written `node.property`, that the experts mapped it to.
`benchmarks build` builds each benchmark on your machine, from its published
files. [licenses.md](licenses.md) gives each benchmark's license and
reference.

## Running `benchmarks build`

```
python -m embedding_model benchmarks build --topmed
python -m embedding_model benchmarks build --gdc
python -m embedding_model benchmarks build --topmed --gdc
```

- **The first build** downloads the benchmark's published files into
  `embedding_model/inputs/benchmarks/downloads/<benchmark>/`. It takes a few
  minutes. A rebuild reuses the downloads.
- **The committed manifest**,
  `embedding_model/benchmarks/manifests/<benchmark>.json`, pins each
  benchmark's inputs. `benchmarks build` checks every build against it.
  [Manifests](manifests.md) says how.
- **If a benchmark cannot be built**, for example because its download fails,
  `benchmarks build` prints the error and still builds the other benchmarks.
  Then it exits with status 1.
- **`--update-manifest`** accepts a build that differs from the committed
  manifest, as [Making a new version](manifests.md#making-a-new-version)
  says. It records the build in the committed manifest, in
  `embedding_model/benchmarks/manifests/`, also with `--inputs`.

## What `benchmarks build` writes

`benchmarks build` writes to `embedding_model/inputs/benchmarks/`.

```
benchmarks/
    downloads/<benchmark>/          the published files as downloaded, kept for the next build
    <benchmark>/
        <benchmark>_mappings.csv    one row per source variable, with its expected targets
        <benchmark>_targets.csv     one row per target property
        manifest.json               the build's manifest
```

Both benchmarks have the same columns. If a benchmark has no data for a
column, the column is empty.

**If you load the files in your own code:** Python's `csv` module reads at
most 131,072 characters in a field by default. In the GDC benchmark, some
target's values can be longer. `pandas.read_csv` reads the files as they are.
With `csv`, raise `csv.field_size_limit` first.

### `<benchmark>_mappings.csv`

One row per source variable. Any of its expected targets counts as correct.

| column | what it holds |
|---|---|
| `source_study_id` | the dbGaP study of the variable's cohort, without its version |
| `source_table_id` | the table that holds the variable: its dbGaP table at the version that the experts cited, or for GDC the paper |
| `source_variable_id` | the variable's dbGaP id, at the version that the experts cited. Empty for GDC. |
| `source_variable_name` | the variable's name in its table |
| `source_type` | the variable's type: `integer`, `float`, `string`, `string/encoded` or `datetime`. For TOPMed, `ai_harmonization.dbgap` maps it from dbGaP's types. For GDC, the build finds it from the paper's column. |
| `source_description` | the variable's description. Empty for GDC. |
| `source_values` | the variable's values, joined by ` \| ` |
| `expected_target` | the targets that the experts mapped the variable to, joined by `;` |

### `<benchmark>_targets.csv`

One row per target property. [`score`](scoring.md#how-a-model-is-scored)
ranks every target of this file for each source variable.

| column | what it holds |
|---|---|
| `target_property` | the target, as `node.property` |
| `target_type` | the target's type. For GDC, the dictionary's type. For TOPMed, the documentation's type, converted as for a dbGaP variable: `decimal` becomes `float`, and `encoded` becomes `string/encoded`. An age target is `float`. |
| `target_description` | the target's description |
| `target_values` | the target's permissible values, joined by ` \| ` |

### `manifest.json`

The build's manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: the two files, each with its row count and its SHA-256
- `sources`: the benchmark's inputs, each with its `role`: `mappings`,
  `targets`, `mappings and targets`, `source text` or `corrections`
- `counts`: the numbers that the build prints

## Details

To use a newer release of a benchmark's input, follow the **Newer releases**
paragraph of that benchmark, below. Then make a new version, as
[Making a new version](manifests.md#making-a-new-version) says. When the new
version changes the benchmark's files, build the corpus again, train its
tokenizer again, and train again, as
[The workflow](../README.md#the-workflow) says.

### TOPMed

The experts of the [TOPMed](https://topmed.nhlbi.nih.gov/) Data Coordinating
Center mapped [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) study variables to
the Center's harmonized variables. `benchmarks build` reads the Center's
documentation,
[UW-GAC/topmed-dcc-harmonized-phenotypes](https://github.com/UW-GAC/topmed-dcc-harmonized-phenotypes/tree/14960eb088d9057f3fcd04c65f609a31e0d981a1),
at the commit that `topmed.json` pins, for example `14960eb`. It reads each
variable's name, description and values from the data dictionary of the
dbGaP release that the documentation cites.

Each harmonized variable gives a target, written `<dataset>.<name>`, for
example `atherosclerosis.cac_score_1`. Most also give an age target,
`<dataset>.age_at_<name>`: the age at which the phenotype was measured.

The build maps each variable that the documentation cites to a target. A
reviewed list,
[`topmed_corrections.csv`](../benchmarks/manifests/topmed_corrections.csv),
beside the committed manifest, corrects some of these mappings. Each row of
the corrections has these columns:

- `target`: the target
- `unit`: the harmonization unit, a study or a subcohort of one
- `source_variable_id` and `source_variable_name`: the variable
- `action`: `exclude` or `add`. `exclude` removes the variable's mapping to
  the target. `add` maps the variable to the age that its unit computes from
  it.
- `use`: how the unit uses the variable. With `exclude`, it is
  `selects records`, `times the event`, `checks another value`,
  `records another concept` or `not used`. With `add`, it is
  `computes the age`.
- `reason`: why

**Newer releases:** when the documentation's repository has a newer commit,
write that commit as the `version` of the entry
`TOPMed DCC harmonized phenotype documentation` in `topmed.json`.
A newer commit can change which dbGaP variables a harmonized variable reads.
So check each row of the corrections against the new commit. If a correction
does not match the documentation, the build fails, and names the first 12
corrections that do not match.

### GDC

Experts mapped the variables of cancer papers' supplementary tables to the
[GDC data dictionary](https://docs.gdc.cancer.gov/Data_Dictionary/).
`benchmarks build` reads three inputs:

- the ground truth of [GDC-SM](https://doi.org/10.5281/zenodo.14963588), the
  GDC Schema Matching Benchmark: the mappings from a paper's variable to a
  GDC property
- the papers' supplementary tables, downloaded from their publishers, for
  each variable's type and values
- the GDC data dictionary: a release of
  [gdcdictionary](https://github.com/NCI-GDC/gdcdictionary), pinned by its
  commit

**Newer releases:** a new version of GDC-SM is a new Zenodo record. In the
committed manifest, `embedding_model/benchmarks/manifests/gdc.json`, write the
record's number as the ground truth's `version`, and the record's DOI as its
`location`. For a newer dictionary, write the commit of the release's tag as
the dictionary's `version`.
