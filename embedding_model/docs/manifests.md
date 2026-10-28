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
- `files`: each file's row count, and its `sha256`. A table kept as parquet
  has `csv_sha256`: the SHA-256 of the table written as CSV (see
  [sources.md](sources.md#what-a-fetch-writes)). A parquet file's bytes
  depend on the library that wrote it, so the two checksums have two uses:
  - `csv_sha256` shows whether two builds hold the same table, on any
    machine. A source's tables have only `csv_sha256`, since their committed
    manifests compare builds;
  - `sha256` shows quickly that a file is unchanged since its step wrote it.
    A source's pairs and a corpus's tables have both checksums, and a step
    that reads them back checks `sha256`;
- `sources`: what the build was made from, one entry each, with
  - `role`: what the input is for, such as `mappings` or `release`;
  - `name` and `location`: a URL, DOI or record;
  - `version`: a release, tag or commit, where the input has one;
  - `fetched`: the date, for an input that serves only its current state;
  - `files`: for an input with several files, such as a fetch, each file as
    the input's own manifest records it;
  - `sha256`, `license` and `citation`, where known;
- `settings`: the settings that shaped the build, such as a corpus's
  validation share;
- `counts`: what the build counted, such as a benchmark's source variables,
  a source's items not open, the pairs by kind, or a corpus's dropped pairs.
  Counts are never compared, so a build with other counts still matches.

## How a build is checked

A new build compares its manifest with the committed one:

- A file is the same when its row count and SHA-256 match.
- An input is the same when its SHA-256 matches, or its version where it has
  no checksum.
- Dates, counts, licenses and citations are not compared.

If everything matches, the build is what the manifest describes. Its files then
replace the working ones. If not, the build lists each file and input that
differs. It then stops, and keeps the working files as they were. That is
expected when an input was updated on purpose. The build is then accepted as a
new version, as the next section describes.

## Making a new version

A difference that is meant makes a new version:

1. In the committed manifest, raise `version`. For a new pin, such as another
   commit of a benchmark's documentation or a source's new release, also edit
   the input's `version`. A change that cannot be pinned, such as a new
   dictionary from a server that serves only its current state, needs only the
   raised `version`.
2. Build again with `--update-manifest`, as `benchmarks build` or
   `sources fetch`. It records the new checksums in the committed manifest.
3. Commit the manifest.

A version committed in git never changes. `--update-manifest` refuses a build
that differs while `version` is still the committed version, so raise it first.
Until the new version is committed, it can be built and accepted as often as
needed. The new files overwrite the working ones.

The build asks the local git checkout whether a version is committed. If git
cannot answer, the build cannot tell, so it refuses a differing build.
