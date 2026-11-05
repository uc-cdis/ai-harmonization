# Design decisions

The documentation file of each step, listed in
[The workflow](../README.md#the-workflow), says what the step does. This file
says why. Each entry links to the documentation that describes the
behaviour, and has these parts:

- **Context:** the facts and forces that the decision answers, without
  judgement, in a few sentences;
- **Decision:** what was chosen, in a few sentences. The linked documentation
  says how it works;
- **Consequences:** what follows, each one good or bad, with the evidence for
  the choice;
- **Confidence:** only where it is low, with what would settle it;
- **Alternatives considered:** each option, with the facts about it and why
  it was not chosen;
- **Confirmation:** the tests that check the decision.

The entries follow the shape of an architecture decision record
([Nygard, 2011](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)),
with the alternatives and the confirmation of the
[MADR 4.0 template](https://github.com/adr/madr/tree/4.0.0/template)
([Kopp et al., 2018](https://ceur-ws.org/Vol-2072/paper9.pdf)). A decision can
change. Its entry is then rewritten, and the earlier choice joins its
alternatives, with the reason it was replaced. Git's history keeps the
earlier text.

- [All steps](#all-steps)
- [Benchmarks](#benchmarks)
- [Scoring](#scoring)
- [Sources](#sources)
- [Pairs](#pairs)
- [Corpus](#corpus)
- [Tokenizer](#tokenizer)
- [Training](#training)
- [References](#references)

## All steps

### The model is built, trained and scored with sentence-transformers

**Described in:** [README.md](../README.md#embedding-model).

**Context.** ai-harmonization's embedding approach loads a model with
sentence-transformers (`ai_harmonization/harmonization_approaches/embeddings.py`).
It embeds texts with the model's `encode`. The models here are a static
embedding table and a transformer with BERT's architecture, each trained from
pairs of texts. Training needs a loop, a loss, batches, evaluation and a saved
format. Scoring needs ranks and metrics.

**Decision.** sentence-transformers is used wherever it does the job:

- the model is a sentence-transformers model, built from `StaticEmbedding`
  or `Transformer`;
- the pairs take its column names, `anchor` and `positive`;
- training uses its `SentenceTransformerTrainer`, its loss
  `MultipleNegativesRankingLoss`, its batch sampler and its data collator;
- scoring loads the model with it, and ranks with its
  `InformationRetrievalEvaluator`.

**Consequences.**

- Good, because the code here holds only what the library does not do: the
  training data, the tokenizer, leakage prevention and the experiments.
- Good, because a trained model loads wherever a sentence-transformers model
  does, such as in the embedding approach or from the Hugging Face Hub. Using
  it needs no new code.
- Good, because the library is well established. It is the library of
  Sentence-BERT
  ([Reimers & Gurevych, 2019](https://doi.org/10.18653/v1/d19-1410)). UKP Lab
  created it, and Hugging Face maintains it ([SBERT.net](https://sbert.net)).
  Hugging Face publishes a recipe that trains a static model with it
  ([Aarsen, 2025](https://huggingface.co/blog/static-embeddings)).
- Good, because `SentenceTransformerTrainer` builds on the `Trainer` of Hugging
  Face's [`transformers`](https://huggingface.co/docs/transformers). That
  Trainer handles checkpoints, logging, seeds and devices, so none of this is
  our own code.
- Good, because the library maintains and tests its losses, samplers and
  evaluator, and its documentation cites their sources.
- Bad, because the library's defaults decide much of what happens. Each
  default that matters is checked and recorded, as in
  [A model is scored as the approaches use it](#a-model-is-scored-as-the-approaches-use-it).
- Bad, because a new major version can change behaviour or move modules.
  Version 6 moves `sentence_transformers.models` to
  `sentence_transformers.sentence_transformer.modules`, for one. `uv.lock`
  pins the version, and each score and each run writes it.

**Alternatives considered.**

- Plain PyTorch: a static model's forward pass is short, so it could be
  written by hand. But the training loop, the loss, batches without
  duplicates, checkpoints and the saved format would all be our code to
  write, test and defend. And the embedding approach could not load the
  model without a converter.
- A training loop of our own around sentence-transformers' modules: it
  would replace the `transformers` Trainer with our code, for no known gain.

### Committed manifests pin the inputs

**Described in:**
[manifests.md](manifests.md#the-committed-manifest-and-the-builds-own),
[manifests.md](manifests.md#making-a-new-version) and
[sources.md](sources.md#what-a-fetch-writes).

**Context.** The repository holds no data from the benchmarks or the sources.
Much of that data is third-party text under its own license, as
[licenses.md](licenses.md) says. So each machine builds the benchmarks and the
sources itself, perhaps months later. Some sources serve only their current
state, which can change without notice. The benchmarks' documentation and
dictionary are kept on GitHub, where a commit pins every file.

**Decision.** Each benchmark and source has a committed manifest.
It pins each input by its version, its commit or its SHA-256, and lists each
file built, with its checksum. A build that differs is refused, and the previous
build's files are kept. An intended change becomes a
[new version](manifests.md#making-a-new-version) of the benchmark or the
source. Downloads are kept for later builds.

**Consequences.**

- Good, because two builds with the same checksums hold the same data. A
  manifest works like a lock file, such as `uv.lock`. The code says how to
  build, and the manifest says from which inputs, with which checksums.
- Good, because the data never changes silently. A source that changed shows
  as a build that differs.
- Good, because a committed version never changes, as a released package
  version never does.
- Good, because a kept download is read again without the network. So a
  change to a reader can be tested on the same files.
- Good, because some inputs serve only their current state, such as
  [NHANES](sources.md#nhanes)'s pages and [NIH CDE](sources.md#nih-cde)'s
  export. Their kept files are the only copy of the state that the manifest
  pins.
- Bad, because a newer state of such an input is read only after someone
  moves its kept copy aside. The docs say to move it, not to delete it, so
  the copy that a committed version pins is not lost.
- Bad, because a build that differs needs git. When git cannot say which
  version is committed, a build that differs is refused. It cannot tell
  whether a raised version is new or already committed. A build that matches
  needs no git.
- Bad, because a new release of a source can be built only after its new
  version is made.

**Alternatives considered.**

- Committing the built files: they hold the third-party text that the
  repository does not redistribute.
- Pins in the code: a new release would need a code change, and the files
  built would have nowhere to record their checksums.
- A committed manifest for pairs, corpora and tokenizers: the code makes them
  from inputs that have one. Their own manifests record those inputs.
- Writing a differing build, with a warning, when git cannot say which
  version is committed: the build could change a committed version without
  anyone noticing.
- Downloading on every build: a source that serves only its current state
  could change between two builds. And every build would wait on the network.

**Confirmation.** The tests in
[test_embedding_model_manifests.py](../../tests/test_embedding_model_manifests.py)
check that:

- a build that differs is refused without `--update-manifest`;
- with `--update-manifest`, a build under an uncommitted version writes its
  checksums;
- a build that differs is refused when git cannot say which version is
  committed;
- a matching build needs no git.

### Licenses are SPDX license expressions

**Described in:** [manifests.md](manifests.md#its-fields) and
[licenses.md](licenses.md).

**Context.** Each input has a license, and some have terms of their own. A
manifest holds each input's license, for tools and for people. One license
can be spelled in several ways, and license pages move.

**Decision.** Each input's `license` is an
[SPDX license expression](https://spdx.github.io/spdx-spec/v2.3.1/SPDX-license-expressions/).
SPDX is the System Package Data Exchange. A license on SPDX's
[License List](https://spdx.org/licenses/) has its short id, such as
`CC-BY-4.0`. Other terms have a `LicenseRef-` id, such as
`LicenseRef-Abcd-Terms`. An input that states no license is `NONE`.
[licenses.md](licenses.md) gives each id's terms and links.

**Consequences.**

- Good, because SPDX is the standard way to name a license so that tools can
  read it. It is the ISO/IEC 5962:2021 standard, and Python's package
  metadata uses it too ([PEP 639](https://peps.python.org/pep-0639/)).
- Good, because a manifest holds only the id, not the address of the
  terms. Each address is written once, in licenses.md. So when a license page
  moves, one link in licenses.md changes, and no manifest does.
- Good, because `NONE` is SPDX's own value for no license. It is not a
  license expression, but SPDX's license fields allow it beside one.
- Bad, because terms without an SPDX id get a `LicenseRef-` id of our own,
  which only licenses.md explains.

**Alternatives considered.**

- A license's name or address in each manifest: one license is spelled in
  several ways. And a moved address would need a fix in every manifest.

**Confirmation.** The tests in
[test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)
check that:

- a stated license becomes its SPDX expression;
- an unknown license stops the fetch.

### Each output folder is written whole

**Described in:** [Inputs and outputs](../README.md#inputs-and-outputs).

**Context.** A step writes many files into one output folder, such as
`corpora/abcd/` for a corpus. A run can fail halfway through, or be stopped
by its user or by a time limit. A later step reads the folder as a whole. A
folder cannot be renamed over a folder that holds files, and a rename can
fail across two file systems.

**Decision.** `folders.staged` gives each step a new folder in a hidden
staging folder beside the destination, such as `corpora/.abcd.x1y2/abcd/` for
`corpora/abcd/`. The step writes its whole output there. Then
`folders.replace` swaps it in with two renames: the previous folder into the
staging folder, then the new folder into its place. If the second rename
fails, the previous folder is moved back.

**Consequences.**

- Good, because a later step never reads half an output. Each rename is
  atomic, for a folder as for a file. `folders.replace` renames with
  `Path.rename`, which calls `os.rename`. Its
  [documentation](https://docs.python.org/3/library/os.html#os.rename) says:
  "If successful, the renaming will be an atomic operation (this is a POSIX
  requirement)." So the destination holds the
  whole previous output or the whole new one.
- Good, because a run that fails leaves the previous output as it was.
- Bad, because the swap as a whole is not atomic. Between its two renames, no
  folder is at the destination, for a moment. The steps run one after
  another, so no step reads a folder in that moment.
- Bad, because the staging folder must sit on the destination's file
  system. So it cannot be in a temporary folder such as `/tmp`.
- Bad, because a run that is stopped leaves its staging folder behind. The
  next run warns about it and says what to do: delete it, or move the
  previous output back from it. Someone has to do that by hand.

**Alternatives considered.**

- Writing the files in place: a failed run leaves a mix of old and new files.
- Deleting the previous folder first: if the new folder then cannot move in,
  neither folder is left.
- `shutil.move`: across file systems, it copies the folder and then deletes
  the original. A run that fails during the copy leaves half a folder.
- `os.replace` in place of `Path.rename`: the two work the same here. On
  Linux and macOS, both make the same system call. On Windows, they differ
  when something already has the new name: `os.replace` replaces it, and
  `os.rename` raises an error. In the swap, nothing has the new name, so
  this case never comes up.
- Linux's `renameat2` with `RENAME_EXCHANGE`, which swaps two folders in one
  call: Python's `os` module does not offer it. It works only on Linux, and
  only on file systems that support it. A package that calls it would be one
  more dependency.
- A symbolic link to the current version's folder: each build writes a new
  version beside the link. One `os.replace` then points the link to the new
  version, and the earlier version is deleted. So the output is never
  missing, even for a moment. But every output becomes a link, and common
  tools treat a link differently from a folder. On Linux, `cp -r
  corpora/abcd` and `tar` copy only the link, so the copy points to a folder
  that is not there. `rm -r corpora/abcd` deletes only the link, and leaves
  the data. The gain is small: it closes only the moment between the two
  renames, when no step reads the folder.

**Confirmation.** The tests in
[test_embedding_model_folders.py](../../tests/test_embedding_model_folders.py)
check that:

- a finished run replaces the previous folder;
- a run that raises keeps the previous folder;
- a failed move puts the previous folder back;
- a previous folder that cannot be put back is kept, and the warning names it;
- a staging folder that a killed run left is warned about;
- the warning says whether to delete that folder or to move the previous
  folder back;
- the staging folder of a longer name, such as `abcd.v2`, is not warned about.

### Large tables are kept as parquet

**Described in:** [sources.md](sources.md#what-a-fetch-writes),
[pairs.md](pairs.md#what-a-pair-holds) and
[corpus.md](corpus.md#what-a-corpus-holds).

**Context.** The source, pairs and corpus tables are large tables of text. Some
of their columns hold lists, or true and false. Many of their texts are real
labels that look like missing values, such as `None` in the list
`Mild | Moderate | Severe | None`, or `N/A`. Training hands the pairs to the
trainer as a [`datasets`](https://huggingface.co/docs/datasets) `Dataset`.
The benchmark and score tables are small, and people open them to check
them.

**Decision.** The source, pairs and corpus tables are written as parquet, by
pandas with [pyarrow](https://arrow.apache.org/docs/python/). The benchmark and
score tables are written as CSV, which opens in a text editor or a spreadsheet.
A benchmark table is read with every cell as a text, so an empty cell stays an
empty text.

**Consequences.**

- Good, because each column keeps its type. A list stays a list, true or
  false stays a boolean, and the text `None` stays a text. Parquet stores a
  list with the encoding of Dremel
  ([Melnik et al., 2010](https://doi.org/10.14778/1920841.1920886)), as its
  [format](https://github.com/apache/parquet-format) says.
- Good, because parquet stores a table by column, and compresses each
  column. pandas compresses with snappy by default. So a large table takes
  less space than as CSV, and it is read without parsing text.
- Good, because parquet is a common standard. In data analytics, Parquet and
  [ORC](https://orc.apache.org/) are the most widely adopted open-source
  columnar formats
  ([Zeng et al., 2023](https://doi.org/10.14778/3626292.3626298)). Hugging
  Face's dataset viewer converts every dataset on the Hub to parquet
  ([Hugging Face](https://huggingface.co/docs/dataset-viewer/parquet)).
- Bad, because a parquet file's bytes depend on the library that wrote it.
  So a table has a second checksum, as
  [Two checksums, for two questions](#two-checksums-for-two-questions) says.
- Bad, because a text editor or a spreadsheet cannot open a parquet file.
  pandas, `datasets` and viewers read it, such as the VS Code extensions
  [Parquet Explorer](https://marketplace.visualstudio.com/items?itemName=AdamViola.parquet-explorer)
  and Microsoft's
  [Data Wrangler](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.datawrangler).
  Both run on Linux, macOS and Windows.

**Alternatives considered.**

- CSV for every table: a list would need a text format of its own. By
  default, pandas reads an empty cell, and texts such as `NA`, `NULL` and
  `None`, as missing values
  ([`pandas.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)).
  So every read would need `keep_default_na=False`. And a large table would
  take more space.
- [JSON Lines](https://jsonlines.org/): it keeps lists, booleans and texts. But
  it is not compressed, and every read parses each row as text.
- ORC, the other widely adopted columnar format: pandas reads and writes it
  too. Hugging Face's Hub keeps datasets as parquet, so parquet is what its
  tools expect.
- Arrow IPC (Feather): it keeps types too, and pandas reads it. The same
  holds for it: Hugging Face's tools expect parquet.
- Reading a benchmark table with pandas' defaults: each empty cell would
  become `NaN`. So would a cell that holds only a text such as `None` or
  `N/A`, such as a value list with that one label. A column whose cells are
  all empty would also get a number type, `float64`. A missing description
  would then be formatted as `nan`.

**Confirmation.** The tests check that:

- texts that look missing, such as `None` and `N/A`, an empty text, and true
  or false read back as written
  ([test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py));
- a pair's values read back from the corpus as the list written
  ([test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py)).

### Two checksums, for two questions

**Described in:** [manifests.md](manifests.md#two-checksums).

**Context.** Two builds are compared across machines, and a step checks each
file that it reads. The bytes of a [parquet](#large-tables-are-kept-as-parquet)
file depend on the libraries that wrote it. The file stores each column's
[Arrow](https://arrow.apache.org/docs/python/api/datatypes.html) type, such as
`string`, `bool` or `int64`. CSV text loses these types: it writes `True` and
the text `"True"` alike, and a missing value like an empty text. Hashing a
large table's content takes far longer than hashing the file's bytes.

**Decision.** A manifest holds each file's `sha256`, the SHA-256 of its
bytes. A parquet table also gets `content_sha256`: the SHA-256 of a line
with the column names and Arrow types, then the rows as CSV. No CSV file is
written. Builds compare parquet tables by `content_sha256`, and other
files by `sha256`. A step that reads a file checks its `sha256`.

**Consequences.**

- Good, because the same table has one `content_sha256` on every machine,
  whichever versions of [pyarrow](https://arrow.apache.org/docs/python/) and
  pandas wrote it. It came out the same under pandas 2.2 with pyarrow 17 as
  under pandas 3.0 with pyarrow 25, for every table.
- Good, because a change of type changes the checksum: `True` and the text
  `"True"` differ. A missing value differs from any text, the empty text
  included.
- Good, because a step checks the file that it reads quickly, by its
  `sha256`. The bytes settle the step's question: is this the file that the
  earlier step wrote?
- Good, because comparing two builds is fast. It compares their recorded
  `content_sha256` for a parquet table, and their `sha256` for any other file.
  No table is read again.
- Bad, because a new build hashes each parquet table twice when it writes it:
  its file's bytes for `sha256`, and its content for `content_sha256`. The
  second hash takes longer. For a large table, such as a corpus's pairs, it
  adds time to the build.
- Bad, because the checksum relies on pandas' CSV writer and on pyarrow's
  type names. pandas could change how it writes a value, or hold a type in
  another unit, such as a timestamp in microseconds. Every checksum would then
  change, though the data would not. The build would be refused, so nothing
  would pass unnoticed. No table holds a timestamp.

**Alternatives considered.**

- `sha256` alone: a build would differ only because other versions of pyarrow
  or pandas wrote it. The file's metadata names them, such as
  `parquet-cpp-arrow version 25.0.1` and pandas' version. And the bytes can
  differ even when the same versions write the same data again.
- `content_sha256` alone: every read of a large table would be slow.
- The rows as CSV, without the schema line or the missing-value marker: a
  column that turned from booleans into text would keep its checksum.
- The schema line with pyarrow's type names as given: an empty column is
  `null` under pandas 2 and `large_string` under pandas 3. So the checksum
  would change with pandas' version.
- The missing-value marker without the backslashes written twice: a cell that
  holds exactly the text `\N` would read like a missing value.
- `pandas.util.hash_pandas_object`: it cannot hash list columns, and its hash
  is pandas' internal algorithm, with no text to compare when two builds
  differ.
- Python's own `csv` module, writing the rows: it relies on no library, but
  writing millions of rows through Python is slower than pandas' writer.
- Comparing two tables row by row, as with pandas: both tables must then be
  on one machine. A checksum fits in a manifest.

**Confirmation.** The tests check that:

- a parquet table is compared by its content
  ([test_embedding_model_manifests.py](../../tests/test_embedding_model_manifests.py));
- a table's checksum does not depend on the system
  ([test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py));
- a change of type, or a missing value in place of an empty text, changes the
  checksum, and an empty column has one checksum
  ([test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py));
- a fixed table keeps its pinned checksum
  ([test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py));
- a corpus's files are checked when the corpus is read back
  ([test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py)).

## Benchmarks

### A benchmark is two files: mappings and targets

**Described in:**
[benchmarks.md](benchmarks.md#how-a-benchmark-is-laid-out).

**Context.** Harmonizing a real study means searching a whole data model. A
source variable has a few expected targets among many properties. In the TOPMed
benchmark, nearly every target is some variable's expected target. In the GDC
benchmark, only a small share of the targets are.

**Decision.** A benchmark is two files:

- `<benchmark>_mappings.csv`, one row per source variable, with its expected
  targets;
- `<benchmark>_targets.csv`, every target property.

Each source variable is ranked against every target in the targets file.

**Consequences.**

- Good, because a score measures a search of the whole data model, as
  harmonizing a real study needs.
- Good, because the layout has the three parts of standard retrieval
  benchmarks, such as BEIR
  ([Thakur et al., 2021](https://doi.org/10.48550/ARXIV.2104.08663)) and MTEB
  ([Muennighoff et al., 2023](https://doi.org/10.18653/v1/2023.eacl-main.148)):
  - a corpus to search: here, the targets file;
  - queries: here, the source variables;
  - the corpus entries that each query should find: here,
    `expected_target`.
- Good, because the targets are written once per benchmark, not once per
  mapping. Mappings can be added without touching the targets file.
- Bad, because the two files must agree: each expected target must be a
  property of the targets file. The build checks this, and refuses a
  benchmark whose files disagree.

**Alternatives considered.**

- The mappings file alone, with the candidates taken from the expected
  targets: the GDC benchmark would shrink to a choice among its few expected
  targets. Its scores would then overstate how a model does.
- BEIR's layout, with the queries apart from the entries that they should
  find: here, one row holds a source variable and all its expected targets.
  Each row is then the experts' mapping, whole.

**Confirmation.** The tests in
[test_embedding_model_benchmarks.py](../../tests/test_embedding_model_benchmarks.py)
check that:

- an expected target missing from the targets is refused;
- a target listed twice is refused.

### In the GDC benchmark, a variable maps to every property that means what the experts named

**Described in:**
[benchmarks.md](benchmarks.md#which-targets-a-variable-maps-to).

**Context.** The [GDC benchmark](benchmarks.md#gdc) starts from the mappings of
GDC-SM ([Santos et al., 2025](https://doi.org/10.5281/zenodo.14963588)). Its
ground truth names a property without its node, such as `race`. Several nodes
of the GDC data dictionary can carry a property of that name, such as
`clinical.race` and `demographic.race`. Three nodes give the name another
meaning. GDC-SM maps columns of male and female to `gender`, in `clinical`.
`demographic.sex_at_birth` has the same values.

**Decision.** Each node whose property of that name means the same thing
gives an expected target. The three properties of another meaning are not
expected targets. `demographic.sex_at_birth` is expected beside `gender`.

**Consequences.**

- Good, because the experts named a property, not a node. A model that ranks
  `clinical.race` or `demographic.race` first has found what they meant, and
  either counts.
- Good, because a model that ranks `demographic.sex_at_birth` first for a
  column of male and female counts as right.
- Good, because each variable counts once in a score, however many expected
  targets it has.
- Bad, because a variable with several expected targets has more chances:
  only its best-placed one counts.
- Bad, because the three properties of another meaning, and the second name
  of `gender`, were found by hand. The code keeps them in two fixed lists,
  `NOT_EXPECTED` and `ALSO_EXPECTED` in `benchmarks/gdc.py`. A new dictionary
  may need both reviewed.

**Alternatives considered.**

- GDC-SM's ground truth as it is, with bare property names as targets, such
  as `race`: the approaches rank `node.property` targets, so the benchmark
  would not score them as they are used. And `project.primary_site`, which
  describes a project, would merge with `case.primary_site`, a patient's.
- One node per property: a target of the same meaning in another node would
  count as wrong.

**Confirmation.** The tests in
[test_embedding_model_benchmarks.py](../../tests/test_embedding_model_benchmarks.py)
check that:

- a property that two nodes carry is expected in both;
- a node that gives the property another meaning is left out;
- a variable mapped to `gender` also expects `demographic.sex_at_birth`.

### The TOPMed benchmark's mappings are corrected by a reviewed list

**Described in:** [benchmarks.md](benchmarks.md#corrections).

**Context.** For each harmonization unit, the TOPMed documentation cites every
dbGaP variable that the unit's function reads. Some of them only select
records, check another value or time an event. Some units set a target to one
value for every subject. The documentation does not say which variable gives
the phenotype and which the age. And a variable's description does not say
what the function does with it.

**Decision.** The build applies a committed list of corrections to the
mappings that the [age rule](benchmarks.md#which-target-a-variable-maps-to)
gives. Each one excludes a mapping, or adds the age that the unit computes
from the variable. It names the unit, the variable's use and a reason. A
mapping stays when the target records the variable's value, or the unit
computes it.

**Consequences.**

- Good, because each expected target is one whose value the variable
  carries. A consent group does not count as right for race.
- Good, because each correction names its unit and its reason, so a reader
  can check it in that unit's function.
- Good, because a correction that does not match the documentation fails
  the build. A newer commit cannot leave a stale one in place.
- Bad, because the corrections come from reading each unit's function by
  hand. A newer commit of the documentation needs them reviewed.
- Bad, because some corrections rest on a judgment that could go the other
  way. A variable can decide which subjects get a value, while another
  variable gives the value. For example, "Prior VT" only drops the controls
  who had a VTE before, and the case status comes from the case-control
  status. The list excludes such a variable, and its reason says where the
  value comes from.

**Alternatives considered.**

- Every cited variable as an expected target: a model would count as right
  for ranking race first for a consent group.
- A rule on the description's words, such as "date" or "days": it would
  move "Weight at index date" to the age. A description can say when a value
  was measured.
- A rule read from the functions' code: the functions are R, written in many
  styles, and their data flow cannot be traced reliably.

**Confirmation.** The tests in
[test_embedding_model_benchmarks.py](../../tests/test_embedding_model_benchmarks.py)
check that:

- an excluded mapping is removed, and a row with no target left is dropped;
- an exclude and an add move a day count from its indicator to its age;
- a correction that the documentation does not match fails, and is named;
- the committed list reads, holds each mapping once and keeps its order.

## Scoring

### A model is scored as the approaches use it

**Described in:** [scoring.md](scoring.md#how-a-model-is-scored) and
[scoring.md](scoring.md#the-metrics).

**Context.** The harmonization approaches embed both sides with
`model.encode()`, and rank by cosine similarity
(`ai_harmonization/utils.py`). By default,
[sentence-transformers](https://sbert.net)' `InformationRetrievalEvaluator`
adds a model's `query` and `document` prompts instead. It also ranks by the
similarity function that the model's config names, such as `dot`.

**Decision.** `score` embeds both sides with the model's default prompt
only, as `model.encode()` does. It ranks by cosine similarity, whatever the
model's config names.

**Consequences.**

- Good, because a score measures a model as the approaches use it, and every
  model is ranked the same way.
- Good, because it changes nothing for a model without prompts that ranks by
  cosine, such as the models trained here. It guards only against a model
  made elsewhere.
- Bad, because a model made for the dot product, or for its own prompts, can
  score lower here than in its own setting.
- Bad, because a model with a `Router` module would still have its two sides
  routed apart. The evaluator embeds them with `encode_query` and
  `encode_document`, and the approaches' `encode` takes the default route. No
  architecture here has a `Router`, so only a model made elsewhere can have
  one.

**Alternatives considered.**

- The evaluator's defaults: a model would be scored with prompts, or a
  similarity, that the approaches never use. Its scores would not compare
  with the others'.

**Confirmation.** The tests in
[test_embedding_model_scoring.py](../../tests/test_embedding_model_scoring.py)
check that:

- targets are ranked by cosine;
- a query prompt is not added;
- the default prompt goes on both sides.

### MRR, accuracy@k and the median rank

**Described in:** [scoring.md](scoring.md#the-metrics).

**Context.** A reviewer harmonizes a source variable by choosing its target
from the suggestions, and needs one expected target near the top. In both
benchmarks, many source variables have several expected targets. By default,
[sentence-transformers](https://sbert.net)' evaluator reports:

- the mean reciprocal rank at 10, MRR@10;
- nDCG@10, the normalized discounted cumulative gain, as its main score;
- accuracy, precision and recall at 1, 3, 5 and 10;
- the mean average precision at 100, MAP@100.

**Decision.** `score` reports where each source variable's best-placed expected
target lands, in four metrics:

- MRR with no cut-off, as the main measure;
- the evaluator's MRR@10, beside it;
- accuracy@k at the evaluator's own cut-offs, 1, 3, 5 and 10;
- the median rank, which `score` computes from the ranks that the evaluator
  writes.

**Consequences.**

- Good, because MRR, accuracy@k and the median rank depend only on a source
  variable's highest expected target: MRR and accuracy@k by their
  definitions, and the median rank because `score` takes that rank. A
  reviewer needs only that one target to harmonize the variable. Precision,
  recall, MAP and nDCG count every expected target.
- Good, because MRR without a cut-off counts every rank. A model that moves an
  expected target from rank 50 to rank 20 scores better. MRR@10 scores every
  rank past 10 as 0.
- Good, because MRR@10 compares with the validation MRR@10 that training
  logs.
- Good, because accuracy@k is what a reviewer of the top k suggestions sees:
  whether an expected target is among them.
- Good, because a few very low ranks do not move the median rank, as they
  move a mean.
- Bad, because the scores do not compare with the nDCG@10 that BEIR and MTEB
  report.

**Alternatives considered.**

- nDCG@10, the evaluator's own main score, and the main score of BEIR
  ([Thakur et al., 2021](https://doi.org/10.48550/ARXIV.2104.08663)) and MTEB
  ([Muennighoff et al., 2023](https://doi.org/10.18653/v1/2023.eacl-main.148)):
  it rewards ranking every expected target high, and it gives no credit below
  rank 10.
- Recall@k or MAP: they measure how many of a variable's expected targets
  are found. One is enough to harmonize a variable.

**Confirmation.** The tests in
[test_embedding_model_scoring.py](../../tests/test_embedding_model_scoring.py)
check that:

- the ranks and the metrics are right on a fictional benchmark;
- MRR@10 counts a rank past 10 as 0, and MRR counts it.

### Models are compared in one text format

**Described in:** [scoring.md](scoring.md#running-score).

**Context.** A format writes a source variable or a target as one text, such
as `compact` or `name_description`. This pipeline's models are trained in
`compact`. Each released model was trained on texts of its own format.

**Decision.** Models are compared in `compact`. A released model can also be
scored in the format of its own training samples.

**Consequences.**

- Good, because in one format, two scores differ only by the model.
- Good, because no format is chosen on the benchmarks for each model.
- Bad, because a released model can score lower in `compact` than in its own
  format. So it is also scored in its own.

**Alternatives considered.**

- Each model in its own training format: two scores would then differ by the
  model and by the format at once. A gap between them could not be put down
  to the model alone. So a model's score in its own format is kept only as an
  additional score.
- Each model in the format that scores best for it: the scores would mix the
  model and the format as well. And each model's format would be chosen on
  the benchmarks.

**Confirmation.** The tests in
[test_embedding_model_scoring.py](../../tests/test_embedding_model_scoring.py)
check that:

- each format has its own folder of scores.

### A source variable's text leaves its node out

**Described in:** [scoring.md](scoring.md#running-score).

**Context.** `score` writes each source variable as a text for the model. A
TOPMed source variable comes from a table of a
[dbGaP](https://www.ncbi.nlm.nih.gov/gap/) study, such as `pht999999.v1`. A
GDC source variable comes from a paper. The approaches
read a study as a `SimpleDataModel`, with each table as a node of
properties. So they write the table first, as in
`pht999999.v1.body_weight (float): ...`. Neither a table nor a paper says
what the variable means.

**Decision.** When `score` writes a benchmark's texts, a source variable's
text starts at its name, in every format, as
`body_weight (float): Body weight in kilograms`.

**Consequences.**

- Good, because a score measures the model on the text that carries meaning.
- Bad, because the text differs by its node from the text that the approaches
  give a model.
- Bad, because the targets keep their node, as the approaches write them,
  while the training texts have no node on either side.
- Bad, because variables of one name in several tables of a study read the
  same, while their targets differ by table. A model that reads only the text
  cannot rank every one's targets first.

**Alternatives considered.**

- The node, as the approaches write it: `pht999999.v1.body_weight (float):
  ...`. The table's id would tell such variables apart, but it does not say
  which exam each one belongs to.

**Confirmation.** The tests in
[test_embedding_model_texts.py](../../tests/test_embedding_model_texts.py)
check that:

- a source's text has no leading dot;
- a text that starts with a dot of its own keeps it.

### Ties keep sentence-transformers' order

**Described in:** [scoring.md](scoring.md#running-score).

**Context.** A tie needs two targets with identical texts. For a static model,
which averages its tokens, the same tokens in another order tie too. Only
`stdelement` leaves a target's node out, so only it gives one property in
several nodes the same text. In the [GDC benchmark](benchmarks.md#gdc), many
targets share their `stdelement` text, but no expected target shares it with
a target that is not expected. The [TOPMed
benchmark](benchmarks.md#topmed)'s targets share no texts.

**Decision.** Targets with the same similarity keep the order that
[sentence-transformers](https://sbert.net)' evaluator gives them. Version 6
breaks a tie by target key, in alphabetical order: `clinical.race` comes
before `demographic.race`.

**Consequences.**

- Good, because every metric comes from the evaluator, with no code of our
  own.
- Good, because the order changes no score on these benchmarks. A tie changes
  a rank only when an expected target ties with a target that is not expected.
- Bad, because a future benchmark could hold such ties. Its scores would
  then depend on which tied target comes first. Version 6 puts them in a fixed
  order, so a score repeats. Earlier releases leave the order to chance, so a
  score could change from one run to the next.
- Bad, because each copy of a target that is not expected still takes a rank
  of its own. That cost belongs to the `stdelement` format.

**Alternatives considered.**

- The realistic rank, the mean of the best and the worst place among tied
  targets
  ([Berrendorf et al., 2020](https://doi.org/10.1109/wiiat50758.2020.00053)).
  Two targets tied at places 2 and 3 would each be at 2.5. Every metric would
  then be computed in code of our own, not by the evaluator. And it changes no
  score on these benchmarks.

### Scoring a run uses its corpus's leakage check

**Described in:** [training.md](training.md#scoring-a-run).

**Context.** `corpus build` checks the pairs against each benchmark built so
far. The corpus's manifest lists those benchmarks, with their versions and
their files' checksums. A benchmark built later, or rebuilt with other files,
was not checked. A model
trained on such a corpus may have learned that benchmark's mappings. A model
trained elsewhere has no such record.

**Decision.** A run's record copies its corpus's checked benchmarks, with
their files' checksums. `score` refuses a benchmark that is not among them,
or whose files differ, as `manifests.differences` compares them. A model
without a run's record is still scored, with a note that its leakage is
unknown.

**Consequences.**

- Good, because a run is never scored on a benchmark that its training data
  may hold.
- Good, because the record stays with the run. So the check still holds when
  the corpus is rebuilt or deleted.
- Good, because a benchmark rebuilt under a version not yet committed, with
  other mappings, is refused too. Its version alone would pass.
- Good, because a new version whose files did not change needs no new corpus
  and no new run. A version is a build's name, not its content.
- Bad, because a released model is scored without the check.

**Alternatives considered.**

- Checking the corpus's current manifest: a corpus rebuilt after the run
  would describe pairs that the model never saw.
- Refusing every model without a run's record: the released models could no
  longer be scored on the same benchmarks.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a benchmark that the corpus was checked against is scored into the run's
  folder;
- a benchmark that it was not checked against is refused;
- another build of a checked benchmark is refused;
- a new version with the same files is scored.

## Sources

### An item is open only when its value is known to allow use

**Described in:** [licenses.md](licenses.md#restricted-items).

**Context.** Some sources mark the items that may not be used:

- [LOINC](sources.md#loinc), by a copyright notice, in free text;
- [NIH CDE](sources.md#nih-cde), by a copyright status and by
  `noRenderAllowed`;
- [caDSR](sources.md#cadsr), by a Proprietary Measure Permission, one kind of
  an element's reference documents;
- [NHANES](sources.md#nhanes), by a data file's use constraints;
- [PhenX](sources.md#phenx), by a protocol's availability.

A source can add a new value, or a new kind of document, at any time.

**Decision.** An item is open only when its restriction field holds a value
known to allow use, for caDSR in each of its documents. A new value closes
the item, and the fetch prints it for review. A LOINC notice is free text, so
any notice closes its item, without a print.

**Consequences.**

- Good, because a new restriction value keeps an item out until someone
  reviews it. This is the principle of fail-safe defaults: "Base access
  decisions on permission rather than exclusion"
  ([Saltzer & Schroeder, 1975](https://doi.org/10.1109/proc.1975.9939)).
- Good, because the fetch keeps every item, closed ones too, and marks each in
  its `open` column. So a later change of verdict needs no new download: a
  new fetch reads the kept files again, and writes the new `open` column.
- Bad, because a harmless new value closes items until its source's
  allow-list is updated.

**Alternatives considered.**

- A deny-list of the values known to restrict: a new value would let its
  items in unseen.
- For caDSR, closing an element only on a Proprietary Measure Permission: a
  new kind of document would leave its elements open before anyone reviews
  it. That bases the decision on exclusion, against the fail-safe default
  above. A new kind arrives only with a new fetch, which is reviewed anyway.

**Confirmation.** The tests in
[test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)
check that:

- a new value is read as closed, and reported;
- LOINC's notices close items;
- NIH CDE's allow-list closes items;
- a caDSR element with a proprietary or a new kind of document is closed;
- a new NHANES use constraint closes its variables;
- a PhenX protocol that is not available closes its variables.

### Two readers of our own: OWL files and PhenX's export

**Described in:** [sources.md](sources.md#ontologies) and
[sources.md](sources.md#phenx).

**Context.** [NCIt](sources.md#ncit) and
[the ontologies](sources.md#ontologies) publish
[OWL](https://www.w3.org/OWL/) files in RDF/XML. pronto, rdflib and owlready2
read OWL files. PhenX publishes a MySQL database export. Neither Python nor the
repository's dependencies read MySQL exports.

**Decision.** `sources/owl.py` reads OWL files in RDF/XML in one streaming
pass, with defusedxml. `sources/phenx.py` reads PhenX's export as text: a
regex tokenizer reads its `INSERT` statements, and checks each row's field
count.

**Consequences.**

- Good, because one OWL reader reads NCIt and every ontology.
- Good, because defusedxml refuses entities and external references in the
  XML.
- Good, because the PhenX reader needs no MySQL server.
- Bad, because both readers are code of our own to maintain. A change in a
  file's layout needs a change in its reader.

**Alternatives considered.**

- rdflib, for the OWL files: it builds the whole graph in memory. It was many
  times slower than this reader on the base file of HP, the Human Phenotype
  Ontology. Only this file was timed with both readers.
- pronto: it is not a dependency. It knows the annotations that OBO defines,
  such as a synonym and its scope. NCIt writes its synonyms with properties of
  its own, which pronto does not know. On the ontologies' base files, pronto
  kept one definition per term and no synonym types.
- owlready2, not tried: it loads each file into a database of its own before
  it can be read.
- The ontologies' OBO files, read with pronto: EVS publishes NCIt in OWL and
  as a flat file, not in OBO. So NCIt would still need an OWL reader. pronto's
  strict OBO parser fails on the OBO file of Uberon, the anatomy ontology.
- sqlite3, for the PhenX export: it cannot run the export's `CREATE TABLE`
  statements. Turning only `\'` into SQL's own escape leaves many variables
  wrong, from `\"`, `\n` and `\r`.
- sqlparse: it is not a dependency. It splits SQL into tokens, such as one
  quoted string, and groups them. A token keeps its text as written, with its
  quotes and backslash escapes. So the code that reads each value would still
  be ours.
- mysql-to-sqlite3 and SQLAlchemy: both need a running MySQL server.

**Confirmation.** The tests in
[test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)
check that:

- the OWL reader keeps each statement's exact text;
- the PhenX reader reads rows with quotes, escapes and nulls.

### NCIt is read from its OWL release

**Described in:** [sources.md](sources.md#ncit) and
[pairs.md](pairs.md#ncit).

**Context.** EVS publishes each NCIt release as an
[OWL](https://www.w3.org/OWL/) file and as a flat file, and its archive keeps
every release. The flat file has one row per concept: its names, one definition
and its parents, as EVS's
[ReadMe](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ReadMe.txt) describes it.
It has no term types and no associations. The pairs need three things that the
OWL release holds:

- each synonym's term type and source;
- every definition, with its source;
- the associations that give value lists.

**Decision.** `sources fetch` reads NCIt from its OWL release,
`Thesaurus_<version>.OWL.zip`, in EVS's archive.

**Consequences.**

- Good, because the rule that picks names of the same meaning can run, and
  value lists can be built.
- Good, because the manifest pins a release, and every later fetch reads the
  same file. It is a release file, so another machine can rebuild the same
  version.
- Bad, because the OWL release is the larger file to download, and it needs
  the OWL reader of our own.

**Alternatives considered.**

- The flat file, `Thesaurus.FLAT.zip`: it is smaller and simpler to read. But
  without term types, the rule that picks names of the same meaning could not
  run, and no value list could be built.

**Confirmation.** The tests in
[test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)
check that:

- a fictional OWL release gives its concepts, synonyms, definitions and value
  lists.

### NCIt's replacements come from its merge records

**Described in:** [sources.md](sources.md#ncit).

**Context.** A retired NCIt concept can name the concept that replaced it.
Each release folder holds the release's history file. Each of its records has
an action, such as `merge` or `retire`. A `merge` record names the concept
that a retired concept was merged into. A `retire` record names the concept's
old parent, as EVS's
[ReadMe_history.txt](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/26.09d_Release/ReadMe_history.txt)
says. EVS's replacements service, a live service, also gives that parent as a
replacement.

**Decision.** `sources fetch` reads NCIt's replacements from the history file
of the same release, from its `merge` records only.

**Consequences.**

- Good, because a retired concept is never paired with its broader parent.
- Good, because the history file is a release file, which the manifest pins
  with the [OWL](https://www.w3.org/OWL/) release. So another machine reads the
  same replacements.
- Good, because the history file also holds merges that the service does not
  give.
- Bad, because a concept retired without a merge gets no replacement. Its
  names follow the rule for a retired item without one, as
  [Retired items are used](#retired-items-are-used) says.

**Alternatives considered.**

- EVS's replacements service: it serves only its current state. And it gives
  a retired concept's old parent as its replacement.
- `retire` records as replacements too: they name the old parent, a broader
  concept, not one of the same meaning.

**Confirmation.** The tests in
[test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)
check that:

- a merge record names a replacement, and a retire record its old parent.

### CDISC is a source of its own

**Described in:** [sources.md](sources.md#cdisc) and
[pairs.md](pairs.md#cdisc).

**Context.** NCI [EVS](https://evs.nci.nih.gov/) publishes CDISC's controlled
terminology as four [packages](sources.md#cdisc): SDTM, ADaM, Protocol and the
Glossary, each with dated releases of its own. Every CDISC term and codelist is also an
[NCIt](sources.md#ncit) concept. The packages hold what CDISC adds to the
concepts: each term's submission value, the short value that a dataset stores,
and its CDISC synonyms. They also list each codelist with its terms.

**Decision.** CDISC is fetched as its own source, from these packages. Its
pairs are read from the packages, not derived from NCIt.

**Consequences.**

- Good, because a codelist becomes an item whose values are its terms, as a
  data model's property has a value list.
- Good, because the codelists are CDISC's own addition: their pairs, with
  the terms as values, are pairs that NCIt does not give.
- Good, because the manifest pins each package's release by its date, apart
  from NCIt's version. A corpus can leave CDISC out with `--source`, as any
  other source.
- Bad, because most of CDISC's other pairs repeat an NCIt pair's anchor and
  canonical names. They differ in another field, most often the description or
  the kind.
  [Copies are kept once](#copies-are-kept-once-as-the-tokenizer-reads-them)
  removes only pairs with the same two texts. So the corpus keeps these pairs.

**Confidence.** Low for CDISC's terms beyond its codelists: no training run
has measured what they add.

**Alternatives considered.**

- Deriving CDISC from NCIt's CDISC subsets and its CDISC synonyms: code of
  our own would have to rebuild the codelists from the subsets. And CDISC
  would follow NCIt's releases, not its own.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- a fictional package gives its codelists and terms.

## Pairs

### A star around each item

**Described in:**
[pairs.md](pairs.md#the-shape-a-star-around-each-item).

**Context.** An item has a canonical and other texts, such as synonyms,
descriptions and question texts. The task pairs a study's wording with a data
model's property. Training uses in-batch negatives, as
[sentence-transformers](https://sbert.net)' `MultipleNegativesRankingLoss`
does. So two pairs of one item in one batch work against each other.

**Decision.** Each pair joins the item's canonical to one of the item's other
texts. For example, the synonyms `BMI` and "Quetelet Index" are each paired
with the canonical "Body Mass Index". They are never paired with each other.

**Consequences.**

- Good, because the shape follows the task: a study's wording on one side, a
  property on the other.
- Good, because every text is pulled toward the same canonical. So the texts
  also come close to each other.
- Good, because n texts give about 2n pairs, not n(n-1)/2. Each text gives a `name` pair, and a `name_description` pair when the item has a
  description. These are two of the
  [kinds of pair](pairs.md#the-shape-a-star-around-each-item). So an item with
  many synonyms does not outweigh the rest of the corpus.
- Good, because an item has few pairs, so two of them rarely share a batch.
  So an item's texts rarely work against each other as in-batch negatives.
- Good, because each pair is marked with what it is made of. So a later
  series can choose pairs by these marks, though `train` reads none of them
  yet:
  - `kind`, such as `name_to_description`, whose positive is a description
    alone;
  - `term_group`: an [expansion](pairs.md#expansions-and-paraphrases) of a
    short form, such as `BMI`, or a paraphrase;
  - `term_type`, the source's own label for a text, such as a synonym type;
  - `status`: `current`, `retired` or `draft`, as
    [Retired items are used](#retired-items-are-used) says;
  - `value_list`: the name of the positive's value list.
- Bad, because two of an item's other texts, such as `BMI` and "Quetelet
  Index", are never paired directly.

**Confidence.** Low: no training run has compared the star with other shapes.

**Alternatives considered.**

- Pairs among all the texts of a concept, as SapBERT builds them from UMLS
  ([Liu et al., 2021](https://doi.org/10.18653/v1/2021.naacl-main.334)).
  SapBERT caps a concept at 50 pairs, to keep the data balanced.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- a fictional item gives the pairs that the test lists;
- each of them joins a text to the canonical.

### Retired items are used

**Described in:** [pairs.md](pairs.md#retired-items).

**Context.** Sources retire items. A source can name a replacement for a
retired item, or none. Older studies use the old wording. A text can belong
to a retired item and to a current one.

**Decision.** Retired items give pairs too:

- A retired item with a replacement gives its canonical and texts as anchors
  of the replacement's canonical.
- A retired item without a replacement keeps its own canonical. It uses only
  the texts that no current or draft item has.

Each pair's `status` marks a retired item's pairs as `retired`.

**Consequences.**

- Good, because a model that knows the retired names can map older studies
  too.
- Good, because a training run can leave retired items out by their `status`.
- Good, because a retired item without a replacement gives no pair for a text
  that a current item also has. For example, a retired "Blood Pressure
  Finding" and a current "Blood Pressure Measurement" can both have the text
  `BP`. Then `BP` is paired only with "Blood Pressure Measurement". So the
  model does not learn to map a current name to a retired item.
- Bad, because every corpus holds the retired texts, and nothing reads
  `status` by default.
- Bad, because a retired item with a replacement gives the replacement all
  its texts, even one that another current item also has.

**Confidence.** Low: no training run has measured what retired items add.

**Alternatives considered.**

- Leaving retired items out: a model would not know the names that older
  studies use.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- a retired item gives its names to its replacement, marked `retired`;
- one without a replacement keeps only the names that no current item has.

### Draft items are used like current ones

**Described in:** [pairs.md](pairs.md#retired-items).

**Context.** Some sources mark items as drafts: NCIt's provisional concepts,
LOINC's trial codes and caDSR's draft elements. A later release can still
change a draft, or drop it. The source writes a draft's texts, as it writes a
current item's.

**Decision.** Draft items give pairs, as current items do. A draft can also
replace a retired item. Each pair's `status` marks a draft's pairs as `draft`.

**Consequences.**

- Good, because a draft's texts mean the same as its canonical, as a current
  item's do. A pair needs nothing more.
- Good, because a training run can leave drafts out by their `status`.
- Bad, because a later release can change a draft. The corpus can then hold
  texts that the newer release does not give.

**Confidence.** Low: no training run has measured what drafts add.

**Alternatives considered.**

- Leaving drafts out: their texts would be lost, though they mean the same as
  their canonical.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- a draft is used and marked;
- a draft can replace a retired item.

### A source's own marks choose the anchors

**Described in:** [pairs.md](pairs.md#texts-of-the-same-meaning).

**Context.** Sources mark what each text of an item is. Only some marks mean
the same thing as the item:

- NCIt's term types, such as `SY`, a synonym, or `AD`, a grammar form;
- the ontologies' synonym scopes, such as `EXACT` or `RELATED`;
- caDSR's alternate-name types, which also mark ids and table names;
- NIH CDE's designation tags, which also mark a form's instructions;
- LOINC's fields: its consumer names are not synonyms.

**Decision.** Each reader keeps an allow-list of the marks that mean the same
thing as the canonical. A text is an anchor only when its mark is on that list.
A mark that a reader has not seen is printed, and its texts are not used. For
LOINC, whose marks are its columns, the fetch prints a new column.

**Consequences.**

- Good, because an anchor means the same thing as the canonical. Instructions,
  ids and related or broader names stay out.
- Good, because every reader follows one rule. NIH CDE holds copies of LOINC's
  consumer names, and its tags mark them. So they stay out, as in LOINC.
- Good, because a new mark stays out until someone reviews it. A new
  restriction value does the same, as
  [An item is open only when its value is known to allow use](#an-item-is-open-only-when-its-value-is-known-to-allow-use)
  says.
- Bad, because a new mark of names also stays out until its allow-list is
  updated.

**Alternatives considered.**

- Every text, whatever its mark: instructions and related names would become
  anchors of the same meaning.
- A deny-list of the marks known not to be names: a new mark would let its
  texts in unseen.

**Confirmation.** The tests check that:

- NCIt and the ontologies use only their same-meaning term types and scopes
  ([test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py));
- caDSR uses only its name types
  ([test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py));
- NIH CDE uses only designations tagged as names
  ([test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py));
- the NCIt, NIH CDE and caDSR readers print a term type, a tag or a type
  that they have not seen
  ([test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py));
- the LOINC fetch prints a column that it does not know
  ([test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)).

### LOINC's example answer lists are used too

**Described in:** [pairs.md](pairs.md#loinc).

**Context.** LOINC links answer lists to its codes, each link with a type:
`NORMATIVE`, `PREFERRED` or `EXAMPLE`. A list linked without a panel is the
code's default list. Many of the codes with a default list have an
`EXAMPLE` list. Values show what a variable holds. Leakage
prevention never compares values.

**Decision.** A LOINC code's values are its default answer list, of any link
type: `NORMATIVE`, `PREFERRED` or `EXAMPLE`. `value_list` holds the type, as
in `LL9999-1 (EXAMPLE)`.

**Consequences.**

- Good, because the codes with an example list keep their values.
- Good, because an example list still shows the kind of answer that a code
  collects.
- Good, because `value_list` keeps the link type. So a training run can read
  it later, as it can read `status`.
- Bad, because an example list is not LOINC's answer for the code. So a pair
  can teach values that a study would not use.

**Confidence.** Low: no training run has measured what the example lists add
or cost.

**Alternatives considered.**

- `NORMATIVE` and `PREFERRED` lists only, the answers that LOINC requires or
  prefers: many of the codes with a list would lose their values.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- a code takes its default list, an `EXAMPLE` one, in the list's order.

### Copies are kept once, as the tokenizer reads them

**Described in:** [pairs.md](pairs.md#duplicates),
[pairs.md](pairs.md#names) and
[corpus.md](corpus.md#combining-the-sources).

**Context.** The same pair can come twice: from two items of one source, or
from two sources. [CDISC](sources.md#cdisc) publishes its terms as
[NCIt](sources.md#ncit) subsets. The model sees only token ids, and its
tokenizer normalises every text. Two texts that normalise to one text get the
same ids. The tokenizer keeps punctuation as tokens, so `body_weight` and
`Body Weight` get different ids.
[The study rule](#leakage-the-answer-key-rule-and-the-study-rule)
needs every study of a text.

**Decision.** Two pairs are copies when the tokenizer normalises their texts
to the same two texts. The first copy is kept, with the studies of all its
copies. `corpus build` reads the sources in a fixed order, with NCIt first.
An item's names are compared the same way, so a name that repeats another,
or the canonical, is used once.

**Consequences.**

- Good, because no pair counts twice. Duplicates make a model weigh some
  examples more than others. For a language model's training text,
  [Lee et al. (2022)](https://doi.org/10.18653/v1/2022.acl-long.577) show that
  removing near-duplicates can "reduce train-test overlap, which affects over
  4% of the validation set of standard datasets". Here each pair is kept
  once, before the split, so it lands on one side.
- Good, because the steps cannot disagree. `normalization.py` holds the one
  normaliser that the duplicate rule, leakage prevention and the tokenizer
  share. When the tokenizer normalises two texts to one text, the duplicate
  rule and leakage prevention treat them as one text too.
- Good, because `body_weight` stays an anchor of the canonical `Body Weight`.
  A study's variable names are written that way, and the pair teaches the
  model to read them.
- Good, because the same sources give the same corpus on every build. A copy
  is kept in the source that it comes from, since NCIt comes before CDISC.
- Good, because no copy's study is lost. The pair kept takes the studies of
  all its copies. So when any copy comes from a benchmark's study, the study
  rule drops the pair kept as well.
- Bad, because a count of pairs by source puts every shared pair under the
  earliest source.

**Alternatives considered.**

- Exact comparison of the texts: `BMI` and `bmi` would stay two pairs,
  though the model reads them as one.
- [`text_key`](#texts-are-compared-by-a-broader-text_key-for-leakage-the-split-retired-items-and-descriptions)
  for an item's names too: `body_weight` would be dropped beside
  `Body Weight`, though the model reads them apart.
- Storing the normalised texts, such as `cafe` for `Café`: the model
  normalises every text that it reads anyway, since its normaliser is part of
  `tokenizer.json`. A tokenizer with another normaliser, such as a cased one,
  could not see case and accents in them. And people read the stored texts,
  in reviews and in `dropped.parquet`.

**Confirmation.** The tests check that:

- pairs whose texts the tokenizer normalises to the same texts are kept once
  ([test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py));
- a duplicate keeps the studies of every copy
  ([test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py));
- the sources combine in their own order, whatever the order named
  ([test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py)).

### Texts are cleaned with ftfy, until nothing changes

**Described in:** [pairs.md](pairs.md#cleaning).

**Context.** Source texts hold markup, HTML entities, escapes such as R's
`<U+00A0>`, and text decoded with the wrong encoding. A source can escape a
text twice by mistake, as in `&amp;lt;5`. One cleaning step can leave work
for an earlier one: `Caf&lt;U+00E9&gt;` becomes `Caf<U+00E9>` after one
pass.

**Decision.** `pairs.common.clean` repairs encodings and unescapes HTML
entities with ftfy's `fix_text`
([Speer, 2019](https://doi.org/10.5281/ZENODO.2591652)), with
`unescape_html=True`. Its other steps read escapes, remove HTML tags, and
normalise characters and spaces. All of `clean`'s steps repeat until the text
stops changing.

**Consequences.**

- Good, because ftfy is a maintained library for text decoded with the wrong
  encoding. Its `fix_text` repeats its fixes until the text stops changing.
- Good, because cleaning is idempotent: cleaning a clean text changes nothing,
  as with Unicode's normal forms. So a step that cleans a text again does no
  harm.
- Good, because a text that a source escaped twice becomes the text that its
  author meant: `&amp;lt;5` becomes `<5`.
- Bad, because a text about HTML entities would lose them. Data dictionaries
  hold none.

**Alternatives considered.**

- Unescaping once, as a web page shows a text: `&amp;lt;5` would stay
  `&lt;5`.
- ftfy's default for HTML, `unescape_html="auto"`: it leaves the entities of
  a text that holds a `<`, such as a tag.
- ftfy's `decode_escapes`, for escapes such as `\u2122`: it also turns the
  `\n` of `C:\new` into a line break.
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) alone, for
  tags: it reads any text in angle brackets as a tag. So "Candida
  <Debaryomycetaceae> infection" would lose a word. Only known HTML tags are
  removed.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- cleaning a clean text changes nothing;
- an escape written with entities is read;
- entities are read until none is left.

## Corpus

### Texts are compared by a broader `text_key` for leakage, the split, retired items and descriptions

**Described in:** [leakage.md](leakage.md#how-two-texts-match) and
[corpus.md](corpus.md#the-split).

**Context.** The [copy rule](#copies-are-kept-once-as-the-tokenizer-reads-them)
asks whether the model reads two texts as one. So it uses the tokenizer's
normalisation, which keeps `body_wt` and `body wt` apart. Four other
comparisons ask whether two texts name the same thing: leakage prevention, the
split, retired items and descriptions.

**Decision.** `pairs.common.text_key` gives each text its `text_key`: the
text as the tokenizer normalises it, with case folded and punctuation turned
into spaces. An empty `text_key` matches nothing. Leakage prevention, the
split, retired items and descriptions compare texts by their `text_key`.

**Consequences.**

- Good, because the broader `text_key` finds texts that name the same thing, such as
  `Waist.cm` and `waist cm`. In each comparison, a match prevents a harm:

  | comparison | compares | a match prevents |
  |---|---|---|
  | leakage prevention | benchmark texts with pair texts | training on a benchmark's text |
  | the split | canonical texts | one concept in both training and validation |
  | retired items | a retired item's texts with current and draft items' texts | a current name pulled toward a retired item |
  | descriptions | an item's descriptions with its canonical | the canonical paired with its own repeat |

- Good, because `text_key` starts from the tokenizer's normalisation. So it
  matches every text that the tokenizer reads as the same.
- Bad, because `text_key` can join two texts that differ in meaning only by their
  punctuation. A pair is then dropped, or moved with its group, without need.

**Alternatives considered.**

- The broader `text_key` for copies too: `body_wt -> Body Weight` and
  `body wt -> Body Weight` would be copies, and one would be dropped. Yet the
  model reads them apart, and `body_wt` teaches it to read a variable name.
- The [copy rule](#copies-are-kept-once-as-the-tokenizer-reads-them) for the
  four comparisons too: `Waist.cm` and `waist cm` would not match. Yet a pair
  with `waist cm` leaks a benchmark's `Waist.cm`.

**Confirmation.** The tests check that:

- a `text_key` is the text as the tokenizer normalises it, with case folded and
  punctuation turned into spaces
  ([test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py));
- accents and case do not hide an answer key
  ([test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py));
- canonicals that differ in punctuation share a group
  ([test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py));
- a description that repeats the canonical, or differs from it only in
  punctuation, gives no pair of its own
  ([test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)).

### The split keeps each group on one side, by a hash

**Described in:** [corpus.md](corpus.md#the-split).

**Context.** The pairs of one item share its canonical. One concept can be two
items, in two sources or in two files of one source. Their pairs are not copies
when a description differs. A [CDISC](sources.md#cdisc) codelist's canonical is
its Codelist Name, which differs from its [NCIt](sources.md#ncit) concept's
name.

**Decision.** A group holds the items that an NCIt concept code or a
canonical text's [`text_key`](#texts-are-compared-by-a-broader-text_key-for-leakage-the-split-retired-items-and-descriptions)
links, directly or through other items. A hash of its `group_id`, the first
`item_id` of its items, puts it into one of 1,000 buckets. The groups in the first buckets go to validation:
buckets 0 to 19 by default, 2% of the groups.

**Consequences.**

- Good, because validation never tests a canonical that the model trained on. A
  repeat shares its canonical's `text_key`, and so its group. scikit-learn's guide to
  grouped data asks for the same: "all the samples in the validation fold come
  from groups that are not represented at all in the paired training fold"
  ([Cross-validation iterators for grouped
  data](https://scikit-learn.org/stable/modules/cross_validation.html#group-cv)).
- Good, because a hash needs no random draw and no seed. So the same sources
  and benchmarks give the same split on every build. Géron (2022) splits a
  test set by a hash of each row's identifier, in the
  [notebook of its chapter 2](https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb).
- Good, because the groups are the connected components of a graph of items and
  canonical texts, which
  [`scipy.sparse.csgraph.connected_components`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html)
  finds.
- Good, because an added source moves only the groups that it links.
- Good, because a larger share only adds buckets. So a group held out at a
  smaller share is held out at the larger one too.
- Bad, because the split is not stratified. Each source has many groups, so
  its share of validation pairs is close to the whole corpus's share, but not
  equal to it.
- Bad, because a share between two steps of 0.001 is refused. It would record
  one share and hold out another.

**Confidence.** Medium: grouped and hashed splits are standard practice. But a
group links only canonicals with the same `text_key`, so two canonicals that
differ by a word stay in two groups.

**Alternatives considered.**

- Groups by item alone: a concept in two sources can have pairs on both
  sides. Grouped by item, some validation pairs repeated a training pair's
  anchor and canonical.
- Groups by canonical text alone: a codelist and its NCIt concept would fall
  into two groups. So the concept code is still needed.
- Canonical texts compared as the tokenizer normalises them: a validation
  pair can then repeat a training canonical that differs only in case or
  punctuation.
- A union-find of our own: scipy's function is the standard tool, and scipy
  was already installed.
- A seeded random split, such as scikit-learn's `GroupShuffleSplit`: every
  added source would redraw every group's side.

**Confirmation.** The tests in
[test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py)
check that:

- the split follows a fixed hash of the group;
- items with one canonical text share a group;
- a larger share keeps the smaller share's groups;
- a share between two buckets stops the build.

### Leakage: the answer-key rule and the study rule

**Described in:** [leakage.md](leakage.md).

**Context.** A model trained on a benchmark's mappings has seen its expected
targets. Its score there would look too good. An anchor gives twin pairs, a
`name` pair and a `name_description` pair: two
[kinds](pairs.md#the-shape-a-star-around-each-item) of pair. Often only the
second matches a benchmark row. Some sources hold
variables of the benchmarks' own [dbGaP](https://www.ncbi.nlm.nih.gov/gap/)
studies: [PhenX](sources.md#phenx) maps its variables to dbGaP study variables.
A word such as `age` occurs in many pairs and in many benchmark rows.

**Decision.** A pair is dropped when:

- it is an answer key: its sides match a source variable and one of its
  expected targets;
- it is an answer key's twin, with the same names;
- it comes from a dbGaP variable of a benchmark's study.

A benchmark text on one side only is shared vocabulary: the pair is kept.

**Consequences.**

- Good, because no training pair holds a benchmark's mapping, as far as the
  matching rules can see. This guards against test-set contamination, which
  GPT-3's authors checked for
  ([Brown et al., 2020](https://doi.org/10.48550/ARXIV.2005.14165)).
- Good, because an answer key's twin cannot keep the source's text paired with
  the answer key's canonical.
- Good, because the study rule drops what text matching cannot see: the other
  variables of a benchmark's studies, whose wording the model would learn.
- Good, because `age` alone stays in the corpus. It is general vocabulary,
  which the model must learn.
- Good, because every benchmark built is checked, so any of them can be
  scored later.
- Bad, because a mapping reworded beyond the matching rules stays in the
  corpus.

**Alternatives considered.**

- Dropping every pair that shares a text with a benchmark: general
  vocabulary, such as `age`, would leave the corpus.

**Confirmation.** The tests in
[test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py)
plant leaks of each kind. They check that:

- each planted leak is dropped, and nothing else;
- an answer key's twin is dropped from any item or source, by its names;
- an answer key comes before a benchmark study;
- shared vocabulary is reported.

### What leakage prevention compares

**Described in:** [leakage.md](leakage.md#the-fields-compared).

**Context.** Each side of a pair is a text formatted from its fields: a name,
a type, a description and values. The same name can come with another type
or other values. Some pairs keep a description in a name field. A common
value list, such as `Unknown | Not Reported | No | Yes`, matches many
targets.

**Decision.** Leakage prevention compares four
[fields](pairs.md#what-a-pair-holds) of a pair: `anchor_name`,
`anchor_description`, `positive_name` and `positive_description`. Each is
compared with the benchmark's names and with its descriptions. It never
compares formatted texts, and it never compares values.

**Consequences.**

- Good, because a name with another type or other values still matches.
- Good, because a description kept in a name field is compared with the
  benchmark's descriptions.
- Good, because a common value list drops no pair.
- Bad, because values alone never drop a pair. A pair whose names and
  descriptions match no mapping is kept, even when it has a mapping's own
  value list.

**Alternatives considered.**

- Comparing formatted texts: a name with another type or other values would
  not match.
- Comparing values too: a common list would drop many pairs that leak
  nothing.

**Confirmation.** The tests in
[test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py)
check that:

- a dropped pair names the fields that matched.

### Two texts match when equal, by 8 words in a row, or by a name's words

**Described in:** [leakage.md](leakage.md#how-two-texts-match).

**Context.** A benchmark's descriptions often repeat a source's descriptions,
with small edits. For example, many target descriptions of
[GDC](benchmarks.md#gdc) repeat [caDSR](sources.md#cadsr)'s or
[NCIt](sources.md#ncit)'s definitions. Most names are shorter than 8 words. A
study's variable names are often variants of a benchmark name, such as
`Waist in cm` for `Waist.cm`.

**Decision.** Two texts match by their
[`text_key`](#texts-are-compared-by-a-broader-text_key-for-leakage-the-split-retired-items-and-descriptions)
in one of three ways:

- `=`: they are equal;
- `~`: they share at least 8 words in a row (`min_overlap_words`);
- `>=`: a side holds every word of a benchmark name of 2 words or more
  (`min_name_words`), and the other side matches that row's description.

The source and the target must match on different sides.

**Consequences.**

- Good, because a copied description with a changed word still matches. The
  run of words is the published check. GPT-3's authors write: "We define a
  'dirty' example as one with any N-gram overlap with any training document"
  ([Brown et al., 2020](https://doi.org/10.48550/ARXIV.2005.14165), Appendix
  C). EleutherAI's evaluation tool uses the same check, as its
  [decontamination docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md)
  say. GPT-3 sets N per benchmark, from the 5th percentile of its examples'
  lengths, between 8 and 13 words. Applied to each compared text, that rule
  gives 8 here, since the benchmarks' descriptions are short.
- Good, because equal texts match for every text, as Dodge et al. "measure
  exact matches, normalized for capitalization and punctuation"
  ([Dodge et al., 2021](https://doi.org/10.18653/v1/2021.emnlp-main.98)).
  Tutubalina et al. match short mentions only when equal
  ([Tutubalina et al., 2020](https://doi.org/10.18653/v1/2020.coling-main.588)).
- Good, because a name's variants match, next to a description of the same
  row.
- Good, because a shared phrase alone never drops a pair. An answer key needs
  both sides to match the same benchmark row.
- Good, because each of the three ways finds answer keys that the other two
  miss.
- Bad, because the name rule is our own combination of published parts, with
  no published evaluation. Its closest precedent is GPT-4's check of a short
  example whole. Its authors "use the entire example if it's less than 50
  characters"
  ([OpenAI et al., 2023](https://doi.org/10.48550/ARXIV.2303.08774), Appendix
  C).
- Bad, because a side must hold every word of a benchmark name. A pair whose
  side drops one word, such as a unit, is kept, though it can hold the
  mapping. For example, `Waist Circumference` lacks the `cm` of the source
  name `Waist_Circumference_cm`. Paired with the target's description, it
  stays in the corpus.

**Confidence.** Low for the two thresholds, 8 words and 2 words: they were
checked by counting answer keys, not by their effect on scores.

**Alternatives considered.**

- Skipping texts shorter than 8 words, as GPT-3 skipped "examples less than
  8 words long" ([Brown et al., 2020](https://doi.org/10.48550/ARXIV.2005.14165)):
  most names are shorter. So no copied name would match.
- One text containing the other whole, from 10 words: one changed word hides
  a copy. So it misses many copies that 8 words in a row find.
- A longer run, such as the 13 words of EleutherAI's tool: it misses copies
  that 8 words in a row find. GPT-3 caps N at 13 for speed.
- A benchmark name contained in a side, of any length: it drops many times
  more pairs. Most come through one-word names, such as `weight`, which are
  general vocabulary.
- A name's words for one-word names too: it drops more pairs, mostly through
  common words, such as `Stage` beside a stock staging description.
- A similarity search with a threshold, such as TF-IDF or embeddings: it
  finds rephrased copies, but
  [Yang et al. (2023)](https://doi.org/10.48550/ARXIV.2311.04850) find that
  "choosing an appropriate similarity threshold ... is often challenging". It
  ranks candidates for a person to judge, rather than drop pairs by itself.

**Confirmation.** The tests in
[test_embedding_model_corpus.py](../../tests/test_embedding_model_corpus.py)
check that:

- texts overlap when they share eight words in a row;
- a long name matches that way too;
- a one-word name matches only an equal text;
- a text that matches the source and the target on one side is kept.

## Tokenizer

### The tokenizer is built with Hugging Face's tokenizers

**Described in:** [tokenizer.md](tokenizer.md).

**Context.** A later step trains the model, a static embedding table of
[sentence-transformers](#the-model-is-built-trained-and-scored-with-sentence-transformers).
Its `StaticEmbedding` module reads the tokenizer. It takes a tokenizer of
Hugging Face's [`tokenizers`](https://github.com/huggingface/tokenizers)
library, or a `transformers` tokenizer that wraps one. Its docstring says the
tokenizer "Must be a fast tokenizer". The tokenizer is trained from scratch,
on the corpus's training texts. It needs a normaliser, a pre-tokenizer, a vocabulary
trainer and a file that other tools read.

**Decision.** The tokenizer is built from the parts of `tokenizers`: its
normalizers, its `BertPreTokenizer`, a `BPE` model that its `BpeTrainer`
trains, and its `tokenizer.json` file.
[The vocabulary's entry](#the-vocabulary-is-learned-by-bpe-not-wordpiece)
says why BPE. Our own code, in [`tokenizer.py`](../tokenizer.py), chooses the
[training texts](tokenizer.md#what-it-learns-from) and the settings. It also
counts the [diagnostics](tokenizer.md#diagnostics) from the tokens that the
library gives.

**Consequences.**

- Good, because `StaticEmbedding` reads the tokenizer directly, with no
  conversion.
- Good, because `tokenizer.json` loads anywhere: with `Tokenizer.from_file`,
  or in `transformers`, whose fast tokenizers wrap the same library.
- Good, because none of the tokenizer's algorithms is our own code. Hugging
  Face maintains the library.
- Bad, because the library's defaults decide part of the result.
  `tokenizer train` sets each setting that matters, and the manifest holds
  the library's version.
- Bad, because `tokenizers` 1.0 cannot yet train or save a tokenizer from
  Python, in its first release candidate. So the project requires a version
  below 1.0 for now. Hugging Face's
  [roadmap for 1.0](https://github.com/huggingface/tokenizers#v100-roadmap)
  brings back the whole Python API before the final release.

**Alternatives considered.**

- [SentencePiece](https://github.com/google/sentencepiece), Google's tokenizer
  library: `StaticEmbedding` cannot read its model file, so each tokenizer
  would need a conversion to `tokenizer.json`.
- [TkTkT](https://github.com/bauwenst/TkTkT) for the diagnostics: this toolkit
  of tokenizers and of measures for them would give more measures, such as
  Rényi entropy and morphological boundaries. But it adds many other
  dependencies.

**Confirmation.** The tests in
[test_embedding_model_tokenizer.py](../../tests/test_embedding_model_tokenizer.py)
check that:

- the `tokenizer.json` that `tokenizer train` writes loads with
  `Tokenizer.from_file`.

### The vocabulary is learned by BPE, not WordPiece

**Described in:** [tokenizer.md](tokenizer.md#how-bpe-learns-its-vocabulary).

**Context.** The model that a later step trains keeps one row of its table
per token id. With one seed, the same ids give the same starting rows. Hugging Face's
[`tokenizers`](https://github.com/huggingface/tokenizers) library trains
[BPE](https://huggingface.co/learn/llm-course/chapter6/5),
[WordPiece](https://huggingface.co/learn/llm-course/chapter6/6) and
[Unigram](https://huggingface.co/learn/llm-course/chapter6/7) vocabularies. In
separate processes on the same texts, BPE gives the same `tokenizer.json` each
time. WordPiece gives different files, with other ids.

**Decision.** `tokenizer train` learns its vocabulary by byte-pair encoding,
with the library's `BpeTrainer`.

**Consequences.**

- Good, because the same texts give the same tokens, with the same ids, in
  every run.

**Alternatives considered.**

- WordPiece: it can give the same tokens with other ids, and so another
  starting point.
- WordPiece with its ids renumbered in a fixed order: each run would give the
  same file. But this is not a standard practice, and the ids would lose the
  rough frequency order that the trainer gives them.
- Unigram, as in [SentencePiece](https://github.com/google/sentencepiece): not
  compared.

**Confirmation.** The tests in
[test_embedding_model_tokenizer.py](../../tests/test_embedding_model_tokenizer.py)
check that:

- the same texts give the same tokenizer.

### The normalisation of BERT's tokenizer, with NFD

**Described in:** [tokenizer.md](tokenizer.md#step-1-normalize).

**Context.** Variable names come in any case, and their case carries no
meaning. Texts hold accents, invisible characters, and compatibility
characters such as `µ`, `²` and `½`.

**Decision.** The tokenizer normalises a text as
[BERT's tokenizer](https://huggingface.co/docs/tokenizers/v0.23.2/en/components) does for BERT's
uncased models. It takes four steps: text cleaning, Unicode's
[NFD](https://www.unicode.org/reports/tr15/#Norm_Forms), lowercase, and no
accents. It leaves out only the spaces that BERT's tokenizer puts around
Chinese characters. `normalization.py` builds the steps from `BertNormalizer`,
`NFD`, `Lowercase` and `StripAccents` of `tokenizers`. Every step that
compares texts uses them.

**Consequences.**

- Good, because `BMI`, `Bmi` and `bmi` get the same tokens. `Café` and
  `Cafe` get the same tokens too.
- Good, because the four steps are a standard: BERT's uncased models use
  them, and `tokenizers` provides each one.
- Bad, because `µ` and `μ` stay two characters, and `x²` and `x2` two texts,
  though they often mean the same thing. Leakage prevention's `text_key` folds
  `µ` into `μ` anyway, by case folding.

**Alternatives considered.**

- NFKD, or NFKC as in
  [SentencePiece](https://github.com/google/sentencepiece)'s default rule,
  `nmt_nfkc`: they fold the micro sign `µ` into the Greek `μ`, which mean the
  same thing. But they also fold `x²` into `x2`, which can name another
  variable. Unicode warns: "Normalization Forms KC and KD must not be blindly
  applied to arbitrary text"
  ([Unicode Consortium, 2026](https://www.unicode.org/reports/tr15/)). On a
  corpus of all the sources, NFKD would change very few training texts. Most
  changes are `µ`, `…`, `´` and `‑`. A few texts would become equal to another
  training text, each a true duplicate, such as `SpO²` and `SpO2`.
- Keeping case: `BMI` and `bmi` would get different tokens, though their case
  carries no meaning.

**Confirmation.** The tests in
[test_embedding_model_tokenizer.py](../../tests/test_embedding_model_tokenizer.py)
check that:

- a text is cleaned, lowercased, stripped of accents and split.

### Only training texts shape the vocabulary

**Described in:** [tokenizer.md](tokenizer.md#what-it-learns-from) and
[tokenizer.md](tokenizer.md#diagnostics).

**Context.** A tokenizer cuts the texts that it learned from into fewer tokens
than new texts. A rare word in the texts that it learned from can become one
whole token. A rare word that only new texts hold is usually cut into pieces.
The validation loss and the benchmark scores judge a model.
[BPE](https://huggingface.co/learn/llm-course/chapter6/5) counts how often each
word occurs.

**Decision.** The tokenizer learns from the training pairs' distinct texts,
each once after normalisation. Four [diagnostics](tokenizer.md#diagnostics)
are measured on the same texts:

- the fertility, the tokens per word;
- the share of words split;
- the unknown rate;
- the share of the vocabulary used.

They are a sanity check, not a measure of quality.

**Consequences.**

- Good, because the tokenizer cuts validation and benchmark texts as it cuts
  any new text.
- Good, because nothing that makes a choice looks at the data that judges the
  result.
- Good, because
  [Rust et al. (2021)](https://doi.org/10.18653/v1/2021.acl-long.243) found,
  across languages, that the fertility and the share of words split matter.
  A lower fertility and a smaller share of continued words, their name for
  words split, "correlate with an increase in downstream performance".
- Bad, because the diagnostics are measured on the training texts, which the
  tokenizer learned from. So they look a little better than on new texts. A
  broken tokenizer still gives bad numbers on them. Examples are an unknown
  rate above 0, and a high fertility when words are cut into single letters.

**Alternatives considered.**

- Learning the vocabulary from the validation and benchmark texts too: the
  vocabulary would then depend on the data that judges the model. And a word that only
  validation texts hold would get a token whose row training never updates.

**Confirmation.** The tests in
[test_embedding_model_tokenizer.py](../../tests/test_embedding_model_tokenizer.py)
check that:

- only training texts shape the vocabulary;
- spellings of one text count once.

### Five special tokens

**Described in:** [tokenizer.md](tokenizer.md#special-tokens).

**Context.** A static model uses only `[UNK]`:
[sentence-transformers](https://sbert.net)' `StaticEmbedding` turns padding
off, and pools each text's tokens as they are. A
[BERT](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/bert)
model pads its batches, so it needs `[PAD]`. `BertTokenizer` of `transformers`
also names `[CLS]`, `[SEP]` and `[MASK]`, which BERT's own pretraining uses.
Removing a token moves the id of every token after it.

**Decision.** Every tokenizer has `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]` and
`[MASK]`, with the first five ids.

**Consequences.**

- Good, because one vocabulary serves both architectures. A later BERT
  setting that needs `[CLS]`, `[SEP]` or `[MASK]`, such as masked-language
  pretraining, finds them in place.
- Bad, because a static model carries four rows that it never uses.
- Bad, because the BERT here uses only `[PAD]`. It adds no `[CLS]` or `[SEP]`
  to a text, and trains no masked tokens.
- Bad, because the `[UNK]` row never trains. Every character of the training
  texts is in the vocabulary, so `[UNK]` never occurs in them, and its row
  keeps its random start.

**Alternatives considered.**

- `[PAD]` and `[UNK]` alone: a later BERT setting that needs `[CLS]`,
  `[SEP]` or `[MASK]` would need another tokenizer, with every id moved.

## Training

### A static run turns its pairs into token ids once, in memory

**Described in:** [training.md](training.md#what-a-run-does).

**Context.** By default, `StaticEmbedding` tokenizes a batch's texts in each
training step, in the trainer's data collator. So every epoch encodes the
same texts to the same ids again. The collator runs in the dataloader's
workers, which are forked processes. The tokenizer's threads do not survive a
fork, so each worker tokenizes on one thread. The `Transformer` module of
sentence-transformers cuts, pads and masks each batch in its own `preprocess`.

**Decision.** A static run encodes its training and validation pairs once,
before training, with `Dataset.map` of Hugging Face's
[`datasets`](https://huggingface.co/docs/datasets). The ids have no special
tokens, and they stay in memory. The standard
`SentenceTransformerDataCollator` takes `features` as its `preprocess_fn`,
which builds the batch from the ids. A BERT run trains on the texts, with its
model's own `preprocess`.

**Consequences.**

- Good, because the training steps do not wait for the tokenizer. The
  encoding runs once, in the main process, on all the tokenizer's threads.
- Good, because the ids cannot be stale. They come from the corpus and the
  tokenizer that the run has just read and checked.
- Good, because the code is small: one function of a few lines, and no
  collator class. Hugging Face's course encodes a dataset the same way, with
  `Dataset.map` before the `Trainer`
  ([LLM Course, chapter 3.3](https://huggingface.co/learn/llm-course/chapter3/3)).
- Good, because the pairs are encoded once only where tokenizing is a large
  share of a step. A static model's step is a table lookup. A BERT step
  spends most of its time in the model's layers. So a BERT keeps the
  standard path, with no code of ours to cut and pad its texts.
- Bad, because each run encodes the pairs again, and holds its own copy of
  the ids in memory. A series trains many runs at once, each with its copy.

**Alternatives considered.**

- Encoding in each training step, as `StaticEmbedding` does by default: every
  epoch would encode the same texts again. The steps would also wait for the
  dataloader's workers.
- A folder of ids shared by every run of a corpus and a tokenizer: the pairs
  would be encoded once for many runs. But the folder needs a lock for runs
  that start together. It also needs a check that it still matches its corpus
  and tokenizer, and a command that builds it.
- Ids for a BERT too: its `features` would copy the cutting, the padding and
  the attention mask of `Transformer.preprocess`, for a small gain.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- `features` on the ids gives what `StaticEmbedding.preprocess` gives on the
  texts;
- the data collator gives the same batch from the ids as from the texts;
- a BERT run never turns its pairs into ids.

### A config in sections, with a flag for every setting

**Described in:** [training.md](training.md#the-config).

**Context.** A run has settings of its own: the corpus, the tokenizer, the
architecture, its size and the loss's scale. The trainer has many more, in
`SentenceTransformerTrainingArguments`. The `HfArgumentParser` of
[`transformers`](https://huggingface.co/docs/transformers) makes a flag for
every field of a dataclass. Its YAML reader takes one flat list of settings.
PyYAML reads `1e-3` as text, since YAML 1.1 writes a number with a dot.

**Decision.** A config has five sections: `data`, `model`, the architecture's
own section, `loss` and `training`. `training` holds the trainer's arguments,
under their own names. Each section's settings become the defaults of one
`HfArgumentParser` over the sections' dataclasses. A flag then overrides any setting. A section or a setting
that does not exist is refused.

**Consequences.**

- Good, because the trainer's settings work without code of ours, under the
  names that the trainer's documentation gives them. A run
  [refuses](#a-run-refuses-the-trainers-prompts-and-router_mapping) two of
  them: `prompts` and `router_mapping`.
- Good, because a flag changes one setting for one run, without a new
  config. The run's `config.yaml` holds the final values.
- Good, because argparse reads a default that is text by the setting's type.
  So `1e-3` in a config becomes a number.
- Good, because each architecture's settings have a section of their own.
  So a static model's config cannot carry BERT's by mistake
  ([Each architecture's settings are its library's](#each-architectures-settings-are-its-librarys-in-a-section-of-their-own)).
- Bad, because the flags have no sections, so each name must be unique
  across them. A test checks it.

**Alternatives considered.**

- `HfArgumentParser.parse_yaml_file`, the reader of Hugging Face's example
  scripts: it takes a flat file, so the sections would be lost.
- A flag of our own for each setting: each setting of the trainer would need
  code of ours, under a name of ours.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a flag overrides the config, and the config overrides the defaults;
- `1e-3` in a config becomes a number;
- an unknown section or setting is refused;
- no two sections share a setting's name.

### A run refuses the trainer's prompts and router_mapping

**Described in:** [training.md](training.md#the-config).

**Context.** Two of the trainer's settings go to its data collator. `prompts`
puts a text before a column's texts. `router_mapping` sends each column to
one side of a `Router` module. The trainer uses both in training only, and
does not save the prompts as the model's own. The approaches embed the source
through `embed_query` and the targets through `embed_documents`. Both make
the same `encode` call: it adds only the default prompt, and takes the
default route.

**Decision.** For now, `read_config` refuses `prompts` and `router_mapping`,
in the config or as a flag, before training starts. The message names the
setting and the reason.

**Consequences.**

- Good, because a model is trained the way that the approaches use it. A
  prompt or a route that they do not add would train the model one way and
  use it another.
- Good, because both sides are entries of data dictionaries, written in one
  format. Sentence-BERT embeds the two texts that it compares with one
  encoder, without a prompt
  ([Reimers & Gurevych, 2019](https://doi.org/10.18653/v1/d19-1410)).
- Good, because a config that sets either fails at once, with the reason,
  before any training.
- Bad, because neither is tested. A prompt or a Router branch for each side
  could teach a model the two roles: a source variable and a target
  property. A test needs the approaches to pass the side: `encode_query` in
  `embed_query`, and `encode_document` in `embed_documents`.

**Alternatives considered.**

- Passing both settings, for both architectures: a run could save the
  prompts as the model's own. But the approaches would add only its default
  prompt, one text on both sides. A prompt or a route for each side needs
  the approaches to call sentence-transformers' `encode_query` and
  `encode_document`, in place of `encode`. A static run would also need its
  prompt as token ids. A route would also need a `Router` module around the
  architecture's modules, with a branch for each side.
- Passing them for BERT only: the approaches would still add no prompt or
  route for each side, as above. The two architectures would also differ in
  what a config can set.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that `prompts` and `router_mapping` are refused, in the config and as a
flag.

### The defaults start from Hugging Face's recipe for a static model

**Described in:** [training.md](training.md#the-config).

**Context.** The trainer's defaults are made for fine-tuning a pretrained
transformer: a learning rate of 5e-5, batches of 8 and 3 epochs. A static
table starts from random values. Hugging Face publishes a recipe that trains
`StaticEmbedding` from scratch with MultipleNegativesRankingLoss. The recipe
trains one epoch on far more pairs, in bf16, with `MatryoshkaLoss`. It
evaluates, logs and saves every 250 steps.

**Decision.** The defaults take from the recipe:

- batches of 2,048 pairs, and its warmup;
- the static model's 1,024 dimensions and learning rate of 0.2.

The others are this pipeline's own:

- five epochs, in random batches;
- an evaluation each epoch, a log every 50 steps, no checkpoint;
- four data loader workers.

**Consequences.**

- Good, because the recipe is standard practice for this very model
  ([Aarsen, 2025](https://huggingface.co/blog/static-embeddings)).
- Good, because every architecture shares these defaults, since none of
  them depends on the model. Only
  [its learning rate](#each-architecture-has-its-own-learning-rate) differs.
- Good, because an evaluation after each epoch gives the validation curve one
  point per epoch. Each evaluation ranks all the validation pairs, so it is
  not repeated within an epoch.
- Good, because a log every 50 steps gives the loss curve many points in a
  run of a few epochs.
- Good, because four data loader workers build the next batches while the
  model trains on the current one, as
  [PyTorch's multi-process loading](https://docs.pytorch.org/docs/stable/data.html#multi-process-data-loading)
  does.
- Good, because no checkpoint is needed:
  [the model kept is the model at the last step](#the-model-kept-is-the-model-at-the-last-step).
- Good, because random batches suit the corpus, and train faster.
  [The star](#a-star-around-each-item) keeps an item's pairs apart, so a
  random batch rarely holds a text twice.
- Good, because the defaults work on every machine. Only a GPU that supports
  bf16 can use it, so it stays off, as in the library, until a config turns
  it on.
- Bad, because five epochs is a starting point, not a measured choice. The
  validation curve in the run's
  [`trainer_state.json`](training.md#trainer_statejson) shows whether a corpus
  needs more, and a series tunes the number.
- Bad, because a run on a GPU trains in 32 bits unless its config turns bf16
  on.
- Bad, because the recipe was made for general text, not for metadata. A
  series may find better settings, and tests them as arms.
- Bad, because the recipe's 1,024 dimensions come with `MatryoshkaLoss`, which
  trains shorter embeddings too. Without it, a model here is used at its full
  length.

**Alternatives considered.**

- The library's defaults: they are made for fine-tuning a pretrained model. A
  learning rate of 5e-5 would barely move a random table.
- The recipe's `NO_DUPLICATES` sampler: it keeps a text from appearing twice
  in a batch. The loss would treat the second copy as a negative, a text that
  should not match. Here it removes a rare repeat, but slows each run. Its
  hashed form also forks processes from a process that runs threads.
- The recipe's single epoch: the recipe trains on far more pairs.
- The recipe's bf16: only a GPU that supports it can use it. A config turns
  it on.
- The recipe's `MatryoshkaLoss`: it also trains shorter embeddings, and a
  model here is used at its full length.
- The recipe's evaluation, logging and checkpoints every 250 steps: they
  suit one long epoch, and a run here keeps no checkpoint.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a config that names only its data trains at these defaults.

### A second architecture: BERT, trained from scratch

**Described in:** [training.md](training.md#bert-a-transformer-embedding-model-with-berts-architecture).

**Context.** A static embedding model gives a token the same vector in every
context. A transformer encoder lets each token's vector depend on the other
tokens of the text. The BERT paper says: "BERT's model architecture is a
multi-layer bidirectional Transformer encoder"
([Devlin et al., 2019](https://doi.org/10.18653/v1/N19-1423)). With its
`Transformer` and `Pooling` modules, sentence-transformers trains such an
encoder with the same trainer and loss. A pretrained BERT learned
from texts that the leakage check never sees.

**Decision.** `architecture: bert` builds `BertModel` of
[`transformers`](https://huggingface.co/docs/transformers) from random weights,
followed by mean pooling. It trains on the same corpus and tokenizer, with the
same loss, trainer and defaults, except its learning rate. No special token is
added to a text, as for the static model.

**Consequences.**

- Good, because the corpus is the model's only training data, as for the
  static model. So the leakage check covers all that the model learned from.
- Good, because the trainer, the validation and `score` work unchanged. A
  BERT run is a sentence-transformers model, which a plain
  `SentenceTransformer` loads.
- Good, because both architectures form a text's embedding the same way:
  the average of its token vectors. So a comparison of the two measures only
  what BERT's layers add.
- Good, because the average, mean pooling, is the default of
  sentence-transformers' `Pooling` and of Sentence-BERT
  ([Reimers & Gurevych, 2019](https://doi.org/10.18653/v1/d19-1410)).
- Bad, because a BERT is much larger and slower than a static model, to train
  and to use.

**Alternatives considered.**

- A pretrained BERT, fine-tuned on the corpus: its pretraining texts are
  outside the leakage check.
- CLS pooling, as BERT's own classifiers use: a `[CLS]` token goes before
  each text, and its output vector is the embedding. A static model cannot
  pool this way: its `[CLS]` vector is the same for every text. So a BERT
  would read other token ids than a static model, and form its embedding
  another way.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a BERT trains on a CPU, and writes its run's folder;
- two runs of one config and one seed give the same weights, on a CPU;
- `score` ranks a benchmark with a BERT run.

### Each architecture's settings are its library's, in a section of their own

**Described in:** [training.md](training.md#architectures).

**Context.** `StaticEmbedding` names its width `embedding_dim`.
[`BertConfig`](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig)
names it `hidden_size`. Its documentation
says that its defaults "will yield a similar configuration to that of the
google-bert/bert-base-uncased" model: 12 layers of 768 dimensions. Google's
release of smaller BERT models names BERT-Small: 4 layers of 512. A batch is padded to its longest text, up to
`max_position_embeddings`, 512 in `BertConfig`. The BERT paper notes that
"attention is quadratic to the sequence length". `HfArgumentParser` cannot read
`BertConfig` itself.

**Decision.** Each architecture has a section named after it, under its
library's names. `static` holds `embedding_dim`. `bert` holds eight fields of
`BertConfig`. Its shape defaults to BERT-Small's, its length to 128 tokens,
and its other settings to `BertConfig`'s defaults. The `model` section names
the architecture. A config or a flag that gives another architecture's settings
is refused.

**Consequences.**

- Good, because each setting has the name that its library's documentation
  gives it.
- Good, because BERT's default shape is a standard one: BERT-Small, from
  Google's release ([Turc et al., 2019](https://doi.org/10.48550/ARXIV.1908.08962)).
  Its feed-forward part is 4 times the width, as the BERT paper sets it.
- Good, because the BERT paper pretrains with "sequence length of 128 for
  90% of the steps" (Appendix A.2). At 128 tokens, BERT-Small trains at a
  batch of 2,048 pairs on one GPU.
- Good, because BERT's other defaults are read from `BertConfig` itself, so
  the code copies none of its numbers.
- Good, because a static config cannot carry BERT's settings by mistake, and
  a BERT config cannot carry the static width.
- Bad, because BERT-Small is a guess at a size that suits a corpus of short
  texts. A series tests other shapes as arms.
- Bad, because a text longer than 128 tokens is cut, in training and in
  scoring.

**Alternatives considered.**

- `BertConfig`'s own shape, BERT-base: 12 layers of 768 dimensions are large
  and slow to train from scratch, at a batch of 2,048 pairs.
- `BertConfig`'s own length, 512 tokens: a batch of 2,048 pairs padded to
  512 does not fit on one GPU, even for BERT-Small.
- One width for both, `embedding_dim`: BERT's width would not have its
  library's name, `hidden_size`. Its default of 1,024 would also give BERT
  twice BERT-Small's width.
- Every field of `BertConfig`: many of them, such as `is_decoder`, do not
  apply to an encoder trained here.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a BERT config reads its own section, at BERT-Small's shape, 128 tokens
  and `BertConfig`'s other defaults;
- a setting of the other architecture, in its section or as a flag, is
  refused;
- no two sections share a setting's name.

### Each architecture has its own learning rate

**Described in:** [training.md](training.md#architectures).

**Context.** Hugging Face's recipe trains a static model at a learning rate of
0.2 ([Aarsen, 2025](https://huggingface.co/blog/static-embeddings)). The BERT
paper pretrains at 1e-4: "We use Adam with learning rate of 1e-4"
([Devlin et al., 2019](https://doi.org/10.18653/v1/N19-1423), Appendix A.2).
The trainer's default, 5e-5, is made for fine-tuning a pretrained model.

**Decision.** The other training defaults are shared, since they concern the
data and the loss. The learning rate is the architecture's own: 0.2 for
`static`, and 1e-4 for `bert`. A config or a flag overrides it.

**Consequences.**

- Good, because each rate comes from a public source for its architecture.
- Bad, because the BERT paper set its rate for pretraining with masked
  language modelling, not for a contrastive loss. A series tests other rates
  as arms.

**Alternatives considered.**

- One rate for both: the recipe's 0.2 is far above every rate of the BERT
  paper, and 1e-4 would barely move a random table.
- Other training defaults for each architecture too: the batch sets the loss's
  in-batch negatives, and the validation pairs set the epochs, for both
  architectures alike.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a static config trains at 0.2, and a BERT config at 1e-4;
- a learning rate that a flag gives is kept.

### The model kept is the model at the last step

**Described in:** [training.md](training.md#what-a-run-does).

**Context.** The trainer can save checkpoints during a run and load the best
of them at the end, by a validation metric. By default, it keeps the model at
the last step. Its learning rate decays linearly to zero over the run. A
series compares its runs by their validation metrics.

**Decision.** A run keeps the model at the last step: a config that sets
`load_best_model_at_end` is refused, and no checkpoint is saved. The run
writes that model's validation metrics.

**Consequences.**

- Good, because the validation pairs are used once: to compare runs. Picking
  the best of many evaluations within a run would favour a lucky one. The same
  pairs would then compare it with other runs.
- Good, because the last steps run at a learning rate near zero, so the model
  changes little at the end.
- Good, because nothing is saved during training, so a run writes only its
  model.
- Bad, because a run that peaks early keeps a later, worse model. The loss
  curve shows it, and a series can then try fewer epochs.

**Alternatives considered.**

- The best checkpoint by validation loss: the choice would use the
  validation pairs twice. And the loss measures the training objective, not
  the ranking.
- The best checkpoint by validation MRR@10: the choice would use the
  validation pairs twice too.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a run writes the model at the last step, and no checkpoint;
- a config that sets `load_best_model_at_end` is refused.

### Validation pairs are ranked, with MRR@10

**Described in:** [training.md](training.md#what-a-run-does).

**Context.** The validation loss measures MultipleNegativesRankingLoss on
held-out pairs: an anchor's positive against the positives of its batch. The
benchmarks measure a ranking: a source variable's expected targets among all
the targets. sentence-transformers' `InformationRetrievalEvaluator` ranks a
corpus for each query. It keeps each query's hits in Python lists, up to the
largest cutoff asked for. Its default cutoff for MRR is 10.

**Decision.** At each evaluation, each validation anchor ranks every
validation positive, each counted once as the tokenizer normalises it. The
evaluator gives MRR@10, and accuracy@1, @3,
@5 and @10. The trainer logs them with the validation loss. The evaluator is
built as `score` builds its own, with a cutoff of 10.

**Consequences.**

- Good, because a series can tune on a ranking, as the benchmarks score one,
  without opening a benchmark.
- Good, because every metric comes from the evaluator, with no code of our
  own.
- Good, because the model reads `BMI` and `bmi` as one text, so they are one
  query, with the positives of both. As two queries, each would count only
  its own positive as relevant, and even a perfect model would miss one.
- Bad, because MRR@10 counts a positive below the tenth rank as 0, while the
  benchmarks' main measure, MRR, counts every rank. `score` also reports
  MRR@10, which compares.
- Bad, because each evaluation encodes the validation texts again.

**Alternatives considered.**

- Full MRR, as `score` computes it: the evaluator would hold every positive
  for every anchor in Python lists. That does not fit for a full corpus's
  validation pairs.
- The validation loss alone: it measures the training objective, not the
  ranking that the model is used for.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a run writes the validation loss, MRR@10 and accuracy@k at the last step;
- two anchors that the tokenizer normalises to one text are one query, with
  the positives of both;
- the trainer's log holds MRR@10 at each evaluation.

### A run trains on one GPU, the first one it sees

**Described in:** [training.md](training.md#gpus-and-cpus).

**Context.** With several GPUs, the `transformers` Trainer wraps the model in
`nn.DataParallel`, which splits each input tensor between them.
`StaticEmbedding` reads a batch as one flat tensor of ids, and the offsets
where each text starts. Building the trainer's settings, `TrainingArguments`,
picks the device and starts CUDA. Until then, PyTorch
counts the visible GPUs with NVIDIA's management library, which reads
`CUDA_VISIBLE_DEVICES` each time. Without that library, the CUDA driver
counts them, and reads the list once.

**Decision.** When a run sees several GPUs and `use_cpu` is off, it keeps the
first one. Before it builds its training settings, it sets
`CUDA_VISIBLE_DEVICES` to that GPU, for its own process. It names the visible
GPUs, and says how to choose another. If the list is already fixed, the run
stops, and asks for `CUDA_VISIBLE_DEVICES` to name one GPU.

**Consequences.**

- Good, because a run's batch, and so its loss's in-batch negatives, does not
  depend on the number of GPUs.
- Good, because a user changes nothing. Other processes keep their own
  `CUDA_VISIBLE_DEVICES`.
- Bad, because the first GPU may be busy with another process. The run does
  not look for a free one; its message says how to choose another.

**Alternatives considered.**

- DataParallel, the trainer's default: the batch would grow with the number
  of GPUs, and the flat ids and their offsets would be split apart.
- Stopping, with a message to set `CUDA_VISIBLE_DEVICES`: a user would have
  to change it, and change it back for other work.
- Looking for a free GPU: "free" has no clear definition, and the run would
  have to ask NVIDIA's management library before CUDA starts.
- Setting the trainer's `_n_gpu` to 1: `transformers` does not document that
  attribute, so a release could change it.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- with `CUDA_VISIBLE_DEVICES=2,3`, the run keeps GPU 2;
- without a list, it keeps GPU 0;
- one GPU changes nothing;
- with `use_cpu`, every GPU stays visible;
- reading a config builds no training settings;
- several GPUs after CUDA has started stop the run;
- a count that stays above one stops the run.

## References

> Aarsen, T. (2025, January 15). *Train 400x faster Static Embedding Models
> with Sentence Transformers*. Hugging Face.
> https://huggingface.co/blog/static-embeddings

> Berrendorf, M., Faerman, E., Vermue, L., & Tresp, V. (2020). Interpretable
> and Fair Comparison of Link Prediction or Entity Alignment Methods. *2020
> IEEE/WIC/ACM International Joint Conference on Web Intelligence and
> Intelligent Agent Technology (WI-IAT)*, 371–374.
> https://doi.org/10.1109/wiiat50758.2020.00053

> Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P.,
> Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S.,
> Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A.,
> Ziegler, D. M., Wu, J., Winter, C., … Amodei, D. (2020). *Language Models
> are Few-Shot Learners* (Version 4). arXiv.
> https://doi.org/10.48550/ARXIV.2005.14165

> Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT:
> Pre-training of Deep Bidirectional Transformers for Language
> Understanding. *Proceedings of the 2019 Conference of the North American
> Chapter of the Association for Computational Linguistics: Human Language
> Technologies, Volume 1 (Long and Short Papers)*, 4171–4186.
> https://doi.org/10.18653/v1/n19-1423

> Dodge, J., Sap, M., Marasović, A., Agnew, W., Ilharco, G., Groeneveld, D.,
> Mitchell, M., & Gardner, M. (2021). Documenting Large Webtext Corpora: A
> Case Study on the Colossal Clean Crawled Corpus. *Proceedings of the 2021
> Conference on Empirical Methods in Natural Language Processing*,
> 1286–1305. https://doi.org/10.18653/v1/2021.emnlp-main.98

> Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and
> TensorFlow* (3rd ed.). O'Reilly Media.

> Kopp, O., Armbruster, A., & Zimmermann, O. (2018). Markdown Architectural
> Decision Records: Format and Tool Support. In N. Herzberg, C. Hochreiner, O.
> Kopp, & J. Lenhard (Eds.), *Proceedings of the 10th Central European
> Workshop on Services and their Composition* (pp. 55–62). CEUR-WS.org.
> https://ceur-ws.org/Vol-2072/paper9.pdf

> Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C.,
> & Carlini, N. (2022). Deduplicating Training Data Makes Language Models
> Better. *Proceedings of the 60th Annual Meeting of the Association for
> Computational Linguistics (Volume 1: Long Papers)*, 8424–8445.
> https://doi.org/10.18653/v1/2022.acl-long.577

> Liu, F., Shareghi, E., Meng, Z., Basaldella, M., & Collier, N. (2021).
> Self-Alignment Pretraining for Biomedical Entity Representations.
> *Proceedings of the 2021 Conference of the North American Chapter of the
> Association for Computational Linguistics: Human Language Technologies*,
> 4228–4238. https://doi.org/10.18653/v1/2021.naacl-main.334

> Melnik, S., Gubarev, A., Long, J. J., Romer, G., Shivakumar, S., Tolton,
> M., & Vassilakis, T. (2010). Dremel: Interactive Analysis of Web-Scale
> Datasets. *Proceedings of the VLDB Endowment, 3*(1-2), 330–339.
> https://doi.org/10.14778/1920841.1920886

> Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive
> Text Embedding Benchmark. *Proceedings of the 17th Conference of the
> European Chapter of the Association for Computational Linguistics*,
> 2014–2037. https://doi.org/10.18653/v1/2023.eacl-main.148

> Nygard, M. (2011, November 15). *Documenting architecture decisions*.
> Cognitect.
> https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions

> OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman,
> F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., Avila, R.,
> Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., Bao, H., Bavarian, M.,
> Belgum, J., … Zoph, B. (2023). *GPT-4 Technical Report* (Version 6). arXiv.
> https://doi.org/10.48550/ARXIV.2303.08774

> Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
> using Siamese BERT-Networks. *Proceedings of the 2019 Conference on
> Empirical Methods in Natural Language Processing and the 9th International
> Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3980–3990.
> https://doi.org/10.18653/v1/d19-1410

> Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How
> Good is Your Tokenizer? On the Monolingual Performance of Multilingual
> Language Models. *Proceedings of the 59th Annual Meeting of the Association
> for Computational Linguistics and the 11th International Joint Conference on
> Natural Language Processing (Volume 1: Long Papers)*, 3118–3135.
> https://doi.org/10.18653/v1/2021.acl-long.243

> Santos, A., Wu, E., Lopez, R., Keegan, S., Pena, E., Liu, W., Liu, Y.,
> Fenyo, D., & Freire, J. (2025). *GDC-SM: The GDC Schema Matching Benchmark*
> (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.14963588

> Saltzer, J. H., & Schroeder, M. D. (1975). The protection of information in
> computer systems. *Proceedings of the IEEE, 63*(9), 1278–1308.
> https://doi.org/10.1109/proc.1975.9939

> Speer, R. (2019). *ftfy* (Version 5.5.1) [Computer software]. Zenodo.
> https://doi.org/10.5281/ZENODO.2591652

> Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).
> *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information
> Retrieval Models* (Version 4). arXiv.
> https://doi.org/10.48550/ARXIV.2104.08663

> Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *Well-Read Students
> Learn Better: On the Importance of Pre-training Compact Models* (Version 2).
> arXiv. https://doi.org/10.48550/ARXIV.1908.08962

> Tutubalina, E., Kadurin, A., & Miftahutdinov, Z. (2020). Fair Evaluation in
> Concept Normalization: a Large-scale Comparative Analysis for BERT-based
> Models. *Proceedings of the 28th International Conference on Computational
> Linguistics*, 6710–6716. https://doi.org/10.18653/v1/2020.coling-main.588

> Unicode Consortium. (2026). *Unicode Normalization Forms* (Unicode Standard
> Annex #15, Revision 58). https://www.unicode.org/reports/tr15/

> Yang, S., Chiang, W.-L., Zheng, L., Gonzalez, J. E., & Stoica, I. (2023).
> *Rethinking Benchmark and Contamination for Language Models with Rephrased
> Samples* (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2311.04850

> Zeng, X., Hui, Y., Shen, J., Pavlo, A., McKinney, W., & Zhang, H. (2023). An
> Empirical Evaluation of Columnar Storage Formats. *Proceedings of the VLDB
> Endowment, 17*(2), 148–161. https://doi.org/10.14778/3626292.3626298
