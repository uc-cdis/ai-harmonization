# Manifests

A manifest says what a build was made from, and what it made. Benchmarks,
sources of training data, the sources' pairs, corpora, tokenizers, training
runs and comparisons all have a manifest, and the manifests work the same way.
Here, "a build" is any step that writes a manifest, from a source fetch to a
comparison.

## The committed manifest and the build's own

Each benchmark and each source has a manifest committed in git:

- `embedding_model/benchmarks/manifests/<benchmark>.json`
- `embedding_model/sources/manifests/<source>.json`

The committed manifest pins what the build is made from. For example, it
pins a git commit of the repository that publishes a benchmark, or a release
of a source. It lists each input that the build reads, by its version, its
[SHA-256](https://doi.org/10.6028/NIST.FIPS.180-4) or both. It also lists each
file that the build makes, with its row count and SHA-256. The code reads the
pins from the committed manifest.

Each build also writes its own manifest beside the files that it makes.
[benchmarks.md](benchmarks.md#manifestjson) and
[sources.md](sources.md#sourcemanifestjson) say where.

## Its fields

Every manifest has these fields, unless a field does not apply to its build:

- `benchmark`, `source`, `corpus`, `tokenizer`, `run`, `comparison` or
  `experiment`: the build's name. The field is named after the kind of
  build, for example `"corpus": "example"`. An experiment's comparison has
  the field `experiment`, with the experiment's name.
- `version`: the benchmark's or the source's version
- `built` or `fetched`: the date of the build
- `files`: each file's row count, and its `sha256`, the SHA-256 of the file's
  bytes. A table kept as [parquet](https://parquet.apache.org/) also has
  `content_sha256`, the SHA-256 of its content.
  [Two checksums](#two-checksums) says what each checksum is for. A
  comparison's manifest has no `files`. Its `sources` give the SHA-256 of
  each scores file that it reads.
- `sources`: what the build was made from, one entry for each input, with
  these fields:
  - `role`: what the input is for, for example `mappings` or `release`
  - `name`: the input's label, for example `NCIt OWL release`
  - `location`: the input's URL, DOI or record
  - `version`: a release, tag or commit of the input. An input of role
    `fetch` or `pairs` holds its source's `version`. An input of role
    `benchmark` holds the benchmark's `version`.
  - `fetched`: for an input of role `fetch`, the date of that fetch. For a
    service, the date when its kept download was written. A later fetch that
    reuses the download keeps that date.
  - `files`: for an input with several files, for example a fetch, each file
    as the input's own manifest holds it
  - `sha256`, where known, and `md5`, where the publisher gives one, as for
    LOINC
  - `license`: the input's license, as an
    [SPDX license expression](https://spdx.github.io/spdx-spec/v2.3.1/SPDX-license-expressions/).
    [licenses.md](licenses.md) gives each id's terms.
  - `citation`: the input's DOI, or its section of [licenses.md](licenses.md)
- `settings`: the settings that shaped the build, for example a corpus's
  validation share
- `counts`: what the build counted, for example a benchmark's source
  variables or a corpus's dropped pairs

A training run's and a comparison's manifests also have fields of their own.
[training.md](training.md#what-train-writes),
[comparing.md](comparing.md#what-compare-writes) and
[experiments.md](experiments.md#manifestjson) list them.

## Two checksums

A checksum answers one of two questions:

- **Do two builds hold the same data?** A new benchmark build or source fetch
  compares its files with its committed manifest, on any machine. It compares a parquet table by its
  row count and its `content_sha256`, and any other file by its row count
  and its `sha256`.
- **Is a file unchanged since its step wrote it?** The step that reads the
  file checks its `sha256`, as
  [How a step checks its inputs](#how-a-step-checks-its-inputs) says.

## How a build is checked

A new benchmark build or source fetch compares its manifest with the
committed one:

- Each file is compared as [Two checksums](#two-checksums) says.
- An input is the same when its SHA-256 matches, or its version where it has
  no checksum.
- An input that serves only its current state is a `service`. A service is
  compared by the content of the files built from it, not by its SHA-256 or
  its version.
- The build does not compare dates, counts, licenses or citations.

If everything matches, the build's folder replaces the previous folder, the
folder of the last build of that name. If not, the build lists each file and
input that differs, and says how to accept the build. It then stops, and
leaves the previous folder as it was.
A build that differs is expected when an input changed on purpose. Then you
accept the build as a new version, as
[Making a new version](#making-a-new-version) describes.

## How a step checks its inputs

A step reads each of its inputs with that input's own manifest, and checks
the input against it. For example, `corpus build` reads each source's pairs
with the pairs' manifest. If a check fails, the step does not use that input.
Its message names the file, and the command or the step that writes the file
again, for example `Run: python -m embedding_model sources fetch --source
ncit`. `pairs prepare` then goes on with the other sources. Every other step
stops.

## Making a new version

Only benchmarks and sources have versions, because only they have a committed
manifest. A source's pairs, a corpus and a tokenizer have no version. Each
build of one of them replaces the last build of that name.

When a benchmark or a source differs on purpose, for example after a new
release, make a new version of it:

1. Edit the committed manifest:
   - Raise the top-level `version`, for example from 1 to 2.
   - For a new pin, also edit the `version` of the input's entry in
     `sources`. A new pin is, for example, another git commit of the
     repository that publishes a benchmark, or a source's new release.
   - For an input that serves only its current state, move its download
     aside, as the input's section in [sources.md](sources.md) says.
     Otherwise the build reads the kept download again. Such an input has no
     pin to edit,
     except PhenX: its export names its release, so you also edit the
     `version` of its entry.
2. Run the build again, with `--update-manifest`: `benchmarks build --topmed`
   or `--gdc`, or `sources fetch --source <source>`. The build writes the new
   checksums into the committed manifest.
3. Commit the manifest.

A version committed in git never changes. So a build that differs asks the
local git checkout which version is committed at `HEAD`. For example, a
source's manifest is committed at version 1, and a new release makes the fetch
differ:

| the build | `version` in the manifest file | git's committed version | `--update-manifest` | result |
|---|---|---|---|---|
| matches | 1 | not asked | given or not | kept |
| differs | 1 | 1 | given or not | refused: version 1 is committed, so raise `version` first |
| differs | 2 | 1 | not given | refused: the message gives the command with `--update-manifest` |
| differs | 2 | 1 | given | accepted as version 2: the build records its checksums in the manifest file, and you then commit the file |
| differs | 2 | git cannot answer | given or not | refused: the message asks for a git checkout where `git status` works |

A build that matches needs no git, so it also works in a copy without git.
Until you commit version 2, you can build and accept it as often as you need.
