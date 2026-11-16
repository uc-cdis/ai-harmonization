# Manifests

A manifest says what a build was made from, and what it made. Benchmarks,
training sources, the sources' pairs, corpora, tokenizers, training runs and
comparisons all have a manifest, and the manifests work the same way. Here, "a
build" is any step that writes a manifest, from a source fetch to a
comparison. [design_decisions.md](design_decisions.md#all-steps) gives the
reasons for manifests and their checksums.

## The committed manifest and the build's own

Each benchmark and each source has a manifest committed in git:

- `embedding_model/benchmarks/manifests/<benchmark>.json`
- `embedding_model/sources/manifests/<source>.json`

The committed manifest pins what the build is made from, for example a commit
of a benchmark's documentation or a release of a source. It lists each input
that the build reads, by its version, its
[SHA-256](https://doi.org/10.6028/NIST.FIPS.180-4) or both. A version can be a
commit or a release. A SHA-256 is the checksum of a downloaded file, for
example a zip. The committed manifest also lists each file that the build
makes, with its row count and SHA-256.

The code reads the pins from the committed manifest. So the code says how to
build, and the manifest says from which inputs.

Each build also writes its own manifest beside the files that it makes.
[benchmarks.md](benchmarks.md#manifestjson) and
[sources.md](sources.md#sourcemanifestjson) say where.

Pairs, corpora and tokenizers have no committed manifest. The code makes
pairs from a fetch, a corpus from pairs and benchmarks, and a tokenizer from a
corpus. The fetches and the benchmarks have committed manifests. The other
builds' own manifests record their inputs.

## Its fields

Every manifest has these fields, unless a field does not apply to its build:

- `benchmark`, `source`, `corpus`, `tokenizer`, `run`, `comparison` or
  `experiment`: the build's name, under the kind of build that it is. An
  experiment's comparison has the field `experiment`, with the experiment's
  name.
- `version`: the benchmark's or the source's version
- `built` or `fetched`: the date of the build
- `files`: each file's row count, and its `sha256`, the SHA-256 of the file's
  bytes. A table kept as [parquet](https://parquet.apache.org/) also has
  `content_sha256`, the SHA-256 of its content.
  [Two checksums](#two-checksums) says what each checksum is for.
- `sources`: what the build was made from, one entry for each input, with
  these fields:
  - `role`: what the input is for, for example `mappings` or `release`
  - `name` and `location`: a URL, DOI or record
  - `version`: a release, tag or commit, where the input has one
  - `fetched`: the date, for an input that serves only its current state
  - `files`: for an input with several files, for example a fetch, each file
    as the input's own manifest holds it
  - `sha256`, where known
  - `license`: the input's license, as an
    [SPDX license expression](https://spdx.github.io/spdx-spec/v2.3.1/SPDX-license-expressions/).
    [licenses.md](licenses.md) gives each id's terms.
  - `citation`: the input's DOI, or its section of [licenses.md](licenses.md)
- `settings`: the settings that shaped the build, for example a corpus's
  validation share
- `counts`: what the build counted, for example a benchmark's source
  variables, a source's items not open, the pairs by kind, or a corpus's
  dropped pairs. A build does not compare counts, so a build with other counts
  still matches.

A table's content, for `content_sha256`, is a line with its column names and
types, then its rows as CSV. In this text, a missing value is `\N`, and each
backslash in a text value is written twice. The text is hashed, and no CSV
file is written.

SPDX, the [System Package Data Exchange](https://spdx.dev/), names each
license by a short id from its [License List](https://spdx.org/licenses/), for
example `CC-BY-4.0`. Terms that are not on the list have a `LicenseRef-` id,
for example `LicenseRef-Abcd-Terms`. An input that states no license is
`NONE`.

A training run's manifest also has four fields of its own:

- `validation`: the model's validation metrics at the last step
- `device`: the device that trained the model
- `ai_harmonization`: the commit of the code, and whether the code had
  uncommitted changes
- `libraries`: the versions of Python and of the libraries

The corpus's entry in the run's `sources` also lists the benchmarks that
`corpus build` checked the corpus against for leakage.
[training.md](training.md#what-train-writes) describes the run's manifest.

A comparison's manifest also has six fields of its own:

- `models`: the models compared, by their names in the tables
- `model_colors`: each model's color as drawn
- `reference`: the reference, or null without `--reference`
- `benchmarks`: each benchmark, with its version
- `ai_harmonization`: the commit of the code, and whether the code had
  uncommitted changes, as in a run's manifest
- `libraries`: the versions of Python and of the libraries that compared the
  models

[comparing.md](comparing.md#what-compare-writes) describes the comparison's
manifest. [experiments.md](experiments.md#manifestjson) describes the
manifest of an experiment's comparison.

## Two checksums

A checksum answers one of two questions:

- **Do two builds hold the same data?** A new build compares its files with
  its committed manifest, on any machine.
- **Is a file unchanged since its step wrote it?** The step that reads the
  file checks its `sha256`.
  [How a step checks its inputs](#how-a-step-checks-its-inputs) lists the
  checks.

For the first question, a build compares a table kept as
[parquet](https://parquet.apache.org/) by its row count and its
`content_sha256`. The `content_sha256` does not depend on how the file was
written. The bytes of a parquet file depend on the library that wrote it. So
the manifest records the file's `sha256`, but the build does not compare it.
The build compares any other file by its row count and its `sha256`.

## How a build is checked

A new build compares its manifest with the committed one:

- A file is the same when its row count and its checksum match, as the first
  question above says.
- An input is the same when its
  [SHA-256](https://doi.org/10.6028/NIST.FIPS.180-4) matches, or its version
  where it has no checksum.
- The build does not compare dates, counts, licenses or citations.

If everything matches, the build is what the manifest describes. The build's
folder then replaces the working folder, the folder of the last build of that
name. If not, the build lists each file and input that differs, and the steps
that accept the build. It then stops, and keeps the working folder as it was.
A build that differs is expected when an input changed on purpose. Then you
accept the build as a new version, as
[Making a new version](#making-a-new-version) describes.

## How a step checks its inputs

A step reads each of its inputs with that input's own manifest. For example,
`corpus build` reads each source's pairs with the pairs' manifest, and each
benchmark with the benchmark's manifest. For each input, the step checks
that:

- the manifest and the files exist, and the manifest can be read
- the manifest names the input asked for, for example the source `ncit`
- each file has the `sha256` that the manifest holds
- the pairs, a corpus's pairs and a benchmark's files have the columns that
  their step writes
- the pairs, a corpus and a run record the format of their texts, one of the
  formats that [scoring.md](scoring.md#text-formats) lists

If a check fails, the step stops. The step's message names the file and the
command that writes the file again, for example `run: python -m
embedding_model sources fetch --source ncit`.

## Making a new version

Only benchmarks and sources have versions, because only they have a committed
manifest. A source's pairs, a corpus and a tokenizer have no version. Each
build of one of them replaces the last build of that name.

When a benchmark or a source differs on purpose, for example after a new
release, make a new version of it:

1. In the committed manifest, raise `version`. For a new pin, for example
   another commit of a benchmark's documentation or a source's new release,
   also edit the input's `version`. A change that cannot be pinned, for
   example a new export from a server that serves only its current state,
   needs only the raised `version`. The build reads such an input's kept copy,
   so first move that copy aside, as the input's section in
   [sources.md](sources.md) says.
2. Run `benchmarks build` or `sources fetch` again, with `--update-manifest`.
   The build writes the new checksums into the committed manifest.
3. Commit the manifest.

A version committed in git never changes. So a build that differs asks the
local git checkout which version is committed at `HEAD`. For example, a
source's manifest is committed at version 2, and a new release makes the fetch
differ:

| the build | `version` in the manifest file | git's committed version | `--update-manifest` | result |
|---|---|---|---|---|
| matches | 2 | not asked | given or not | kept |
| differs | 2 | 2 | given or not | refused: version 2 is committed, so raise `version` first |
| differs | 3 | 2 | not given | refused: the message gives the command with `--update-manifest` |
| differs | 3 | 2 | given | accepted as version 3: the build records its checksums in the manifest file, and you then commit the file |
| differs | 3 | git cannot answer | given or not | refused: the message asks for a git checkout where `git status` works |

A build that matches needs no git, so it also works in a copy without git.
Until you commit version 3, you can build and accept it as often as you need.
A refused build is deleted, and the working folder stays as it was. An
accepted build replaces the working folder.
