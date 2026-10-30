# Manifests

A manifest says what a build was made from, and what it made. Benchmarks,
training sources, their pairs, corpora and tokenizers all have them, and they
work the same way. Here, "a build" means a benchmark build, a source fetch,
the preparation of a source's pairs, a corpus build or a tokenizer's
training.

## The committed manifest and the build's own

Each benchmark and each source has a manifest committed in git:

- `embedding_model/benchmarks/manifests/<benchmark>.json`;
- `embedding_model/sources/manifests/<source>.json`.

It pins what the build is made from, such as a commit of a benchmark's
documentation or a release of a source. It records each input that the build
reads, by its version, its SHA-256 or both. A version can be a commit or a
release, and a SHA-256 is that of a downloaded file, such as a zip. It also
records each file that the build makes, with its row count and SHA-256. The
code reads its pins from this file. So the code says how to build, and the
manifest says from which inputs.

Each build also writes its own manifest beside the files that it makes.
[benchmarks.md](benchmarks.md#the-manifest) and
[sources.md](sources.md#what-a-fetch-writes) say where.

Pairs, corpora and tokenizers have no committed manifest. The code makes
pairs from a fetch, a corpus from pairs and benchmarks, and a tokenizer from a
corpus. The fetches and the benchmarks have committed manifests. The other
builds' own manifests record their inputs.

## Its fields

Every manifest has these fields. A field that does not apply is left out:

- `benchmark`, `source`, `corpus` or `tokenizer`: the build's name, under the
  kind of build that it is;
- `version`: the benchmark's or the source's version;
- `built` or `fetched`: the date of the build;
- `files`: each file's row count, and its `sha256`, the SHA-256 of the
  file's bytes. A table kept as parquet also has `csv_sha256`, the SHA-256 of
  the table written as CSV. [Two checksums](#two-checksums) says what each
  one is for;
- `sources`: what the build was made from, one entry each, with
  - `role`: what the input is for, such as `mappings` or `release`;
  - `name` and `location`: a URL, DOI or record;
  - `version`: a release, tag or commit, where the input has one;
  - `fetched`: the date, for an input that serves only its current state;
  - `files`: for an input with several files, such as a fetch, each file as
    the input's own manifest records it;
  - `sha256`, where known;
  - `license`: the input's license, in the format of SPDX, the
    [System Package Data Exchange](https://spdx.dev/): an
    [SPDX license expression](https://spdx.github.io/spdx-spec/v2.3.1/SPDX-license-expressions/).
    SPDX names each license by a short id from its
    [License List](https://spdx.org/licenses/), such as `CC-BY-4.0`.
    Terms that are not on the list have a `LicenseRef-` id, such as
    `LicenseRef-Abcd-Terms`. An input that states no license is `NONE`.
    [licenses.md](licenses.md) gives each id's terms;
  - `citation`: the input's DOI, or its section of
    [licenses.md](licenses.md);
- `settings`: the settings that shaped the build, such as a corpus's
  validation share;
- `counts`: what the build counted, such as a benchmark's source variables,
  a source's items not open, the pairs by kind, or a corpus's dropped pairs.
  Counts are never compared, so a build with other counts still matches.

## Two checksums

A checksum answers one of two questions:

- **Do two builds hold the same data?** A new build is compared with its
  committed manifest, on any machine and with any library version. A table
  kept as parquet is compared by its row count and its `csv_sha256`. The
  bytes of a parquet file depend on the library that wrote it, so its
  `sha256` is recorded but not compared. Any other file is compared by its
  row count and its `sha256`.
- **Is a file unchanged since its step wrote it?** The step that reads the
  file checks its `sha256`.
  [How a step checks its inputs](#how-a-step-checks-its-inputs) lists the
  checks.

## How a build is checked

A new build compares its manifest with the committed one:

- A file is the same when its row count and its checksum match, as the first
  question above says.
- An input is the same when its SHA-256 matches, or its version where it has
  no checksum.
- Dates, counts, licenses and citations are not compared.

If everything matches, the build is what the manifest describes. Its folder
then replaces the working one. If not, the build lists each file and input
that differs, and the steps that accept it. It then stops, and keeps the
working folder as it was. That is
expected when an input was updated on purpose. The build is then accepted as a
new version, as [Making a new version](#making-a-new-version) describes.

## How a step checks its inputs

A step reads each of its inputs with that input's own manifest. For example,
`corpus build` reads each source's pairs with the pairs' manifest, and each
benchmark with the benchmark's manifest. For each input, the step checks
that:

- the manifest and the files exist, and the manifest can be read;
- the manifest names the input asked for, such as the source `ncit`;
- each file has the `sha256` that the manifest records;
- the pairs, a corpus's pairs and a benchmark's files have the columns that
  their step writes.

If a check fails, the step stops. Its message names the file and the
command that writes it again, such as `run: python -m embedding_model
sources fetch --source ncit`.

## Making a new version

Only benchmarks and sources have versions, since only they have a committed
manifest. A source's pairs, a corpus and a tokenizer have none. Each build of
one of them replaces the last build of that name.

When a benchmark or a source differs on purpose, such as after a new release,
make a new version of it:

1. In the committed manifest, raise `version`. For a new pin, such as another
   commit of a benchmark's documentation or a source's new release, also edit
   the input's `version`. A change that cannot be pinned, such as a new
   dictionary from a server that serves only its current state, needs only the
   raised `version`. The build reads such an input's kept copy, so first move
   that copy aside, as the input's section in benchmarks.md or sources.md says.
2. Build again with `--update-manifest`, as `benchmarks build` or
   `sources fetch`. It records the new checksums in the committed manifest.
3. Commit the manifest.

A version committed in git never changes. So the build asks the local git
checkout which version is committed at `HEAD`. For example, a source's manifest
is committed at version 2, and a new release makes the fetch differ:

| the build | `version` in the manifest file | git's committed version | `--update-manifest` | result |
|---|---|---|---|---|
| matches | 2 | not asked | given or not | kept |
| differs | 2 | 2 | given or not | refused: version 2 is committed, so raise `version` first |
| differs | 3 | 2 | not given | refused: the message gives the command with `--update-manifest` |
| differs | 3 | 2 | given | accepted as version 3: the new build's checksums are recorded in the manifest file, which you then commit |
| differs | 3 | git cannot answer | given or not | refused: the message asks for a git checkout where `git status` works |

A build that matches needs no git, so it also works in a copy without git. Until
version 3 is committed, it can be built and accepted as often as needed. A
refused build is deleted, and the working folder stays as it was. An accepted
build replaces it.
