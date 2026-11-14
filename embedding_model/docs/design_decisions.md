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
- **Confirmation:** the tests that check the decision, if any.

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
- [Comparing](#comparing)
- [Sources](#sources)
- [Pairs](#pairs)
- [Corpus](#corpus)
- [Tokenizer](#tokenizer)
- [Training](#training)
- [Experiments](#experiments)
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
  written by hand. But the training loop, the loss, the batches, the
  evaluation and the saved format would all be our code to write, test and
  defend. And the embedding approach could not load the
  model without a converter.
- A training loop of our own around sentence-transformers' modules: it
  would replace the `transformers` Trainer with our code, for no known gain.

### Committed manifests pin the inputs

**Described in:**
[manifests.md](manifests.md#the-committed-manifest-and-the-builds-own),
[manifests.md](manifests.md#making-a-new-version) and
[sources.md](sources.md#what-sources-fetch-writes).

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
source.

**Consequences.**

- Good, because two builds with the same checksums hold the same data. A
  manifest works like a lock file, such as `uv.lock`. The code says how to
  build, and the manifest says from which inputs, with which checksums.
- Good, because the data never changes silently. A source that changed shows
  as a build that differs.
- Good, because a version names one build. A build that differs needs a new
  version, as a released package does.
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
**Confirmation.** The tests in
[test_embedding_model_manifests.py](../../tests/test_embedding_model_manifests.py)
check that:

- a build that differs is refused without `--update-manifest`;
- with `--update-manifest`, a build under an uncommitted version writes its
  checksums;
- a build that differs is refused when git cannot say which version is
  committed;
- a matching build needs no git.

### Downloads are kept for later builds

**Described in:** [sources.md](sources.md#what-sources-fetch-writes) and
[manifests.md](manifests.md#making-a-new-version).

**Context.** Each benchmark build and each source fetch downloads its inputs
from their publishers. Some inputs serve only their current state, such as
[NHANES](sources.md#nhanes)'s pages and [NIH CDE](sources.md#nih-cde)'s
export. That state can change without notice. A change to a reader is tested
on the files that it reads.

**Decision.** Each download is kept in the inputs folder. A later build reads
the kept file again, and downloads only what it does not have.

**Consequences.**

- Good, because a kept download is read again without the network. So a
  change to a reader can be tested on the same files.
- Good, because the kept files of an input that serves only its current
  state are the only copy of the state that the manifest pins.
- Bad, because a newer state of such an input is read only after someone
  moves its kept copy aside. The docs say to move it, not to delete it, so
  the copy that a committed version pins is not lost.
- Bad, because the kept downloads take disk space.

**Alternatives considered.**

- Downloading on every build: a source that serves only its current state
  could change between two builds. And every build would wait on the network.

**Confirmation.** The tests in
[test_embedding_model_download.py](../../tests/test_embedding_model_download.py)
check that:

- a file is downloaded once, and no partial file is left;
- a failed check keeps nothing.

### Two checksums, for two questions

**Described in:** [manifests.md](manifests.md#two-checksums).

**Context.** Two builds are compared across machines, and a step checks each
file that it reads. The bytes of a [parquet](https://parquet.apache.org/)
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
[benchmarks.md](benchmarks.md#what-benchmarks-build-writes).

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
  Each row then holds all the experts' mappings of that variable.

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

### GDC's data dictionary is read by a reader of our own

**Described in:** [benchmarks.md](benchmarks.md#the-targets).

**Context.** The GDC benchmark's targets come from a release of
[gdcdictionary](https://github.com/NCI-GDC/gdcdictionary). The benchmark's
manifest pins it by its commit.
Each node is a YAML file, which refers to shared definitions by `$ref`. The
release's own loader, `GDCDictionary`, needs `jsonschema` and `pyyaml`. On
PyPI, the name `gdcdictionary` holds another package, a "Gen3 generic data
dictionary", whose last release came in 2019.

**Decision.** The build reads the schemas from the downloaded archive, as
data. It resolves each `$ref` with the
[`referencing`](https://referencing.readthedocs.io/) library, and merges each
definition over the node's own keys, as `GDCDictionary` merges it.

**Consequences.**

- Good, because the dictionary is pinned like every other input, by its
  committed manifest. A newer release is its commit, written in the
  manifest, and a build with `--update-manifest` accepts it.
- Good, because the build runs no code from the download.
- Good, because `referencing` adds no package: `uv.lock` holds it already, as
  a dependency of `jsonschema`, which
  [resolves its references with it](https://python-jsonschema.readthedocs.io/en/stable/referencing/).
- Bad, because the merge rule is copied from `GDCDictionary`. A change to it
  there must be copied here.

**Alternatives considered.**

- gdcdictionary as a dependency of the `embedding-model` group. Its releases
  are only on GitHub, so it would need a git source in `tool.uv.sources`,
  which only uv reads. Installed with pip, the group would get PyPI's
  package of that name, Gen3's dictionary of 2019, instead. A new dictionary
  would change both the dependency and the manifest.
- The loader of the downloaded release, imported from the archive. The build
  would run code that `uv.lock` does not cover. Each release's loader can
  need other packages, and can change how it is called.
- `DataDictionary` of dictionaryutils, the loader of Gen3's data
  dictionaries. It reads each `$ref` as a single reference, the form that
  Gen3's dictionaries write, such as BioData Catalyst's. In the pinned
  release, a node's `$ref` can be a list of references, so version 3.5.1
  stops with an error. It would also add three packages, dictionaryutils,
  gen3dictionary and cdislogging, for a few lines of our code. And it requires
  `jsonschema` 4.23.0 or below, which the repository's lock would then hold.
- The dictionary that GDC's API serves, at `/v0/submission/_dictionary/_all`
  ([GDC's submission API](https://docs.gdc.cancer.gov/API/Users_Guide/Submission/)).
  It serves only the dictionary that GDC uses now, which can change without
  notice. So a later build could not read the pinned release. Its schemas
  still hold `$ref`s, which the build would resolve all the same.

**Confirmation.** The tests in
[test_embedding_model_benchmarks.py](../../tests/test_embedding_model_benchmarks.py)
check that:

- only the nodes are read as nodes, and not the shared schemas;
- a list of references, as GDC's nodes give, is merged over the node's own
  keys, so that the definition's value wins.

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
  the build. A newer commit cannot leave a stale one in place. The build
  matches a correction by its variable and its target. It does not check
  the correction's harmonization unit.
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

### MRR, accuracy@k, the median rank and nDCG

**Described in:** [scoring.md](scoring.md#the-metrics).

**Context.** A data curator harmonizes a source variable by choosing its
target from the suggestions, and needs one expected target near the top. In
both benchmarks, many source variables have several expected targets. By default,
[sentence-transformers](https://sbert.net)' evaluator reports:

- the mean reciprocal rank at 10, MRR@10;
- nDCG@10, the normalized discounted cumulative gain, as its main score;
- accuracy, precision and recall at 1, 3, 5 and 10;
- the mean average precision at 100, MAP@100.

**Decision.** `score` reports where each source variable's best-placed
expected target lands:

- MRR with no cut-off, as the main measure;
- MRR@10, beside it;
- accuracy@k at the evaluator's own cut-offs, 1, 3, 5 and 10;
- the median rank, from the evaluator's ranks.

It also reports the evaluator's overall nDCG@10, the retrieval score of BEIR and
MTEB.

**Consequences.**

- Good, because MRR, accuracy@k and the median rank depend only on a source
  variable's highest expected target. MRR and accuracy@k do so by their
  definitions, and the median rank because `score` takes that rank. A
  curator needs only that one target to harmonize the variable. Precision,
  recall, MAP and nDCG count every expected target.
- Good, because MRR without a cut-off counts every rank. A model that moves an
  expected target from rank 50 to rank 20 scores better. MRR@10 scores every
  rank past 10 as 0.
- Good, because MRR@10 compares with the validation MRR@10 that training
  logs.
- Good, because accuracy@k is what a curator who reviews the top k
  suggestions sees: whether an expected target is among them.
- Good, because a few very low ranks do not move the median rank, as they
  move a mean.
- Good, because nDCG@10 is the main score of BEIR
  ([Thakur et al., 2021](https://doi.org/10.48550/ARXIV.2104.08663)) and of
  MTEB's retrieval tasks
  ([Muennighoff et al., 2023](https://doi.org/10.18653/v1/2023.eacl-main.148)).
  So the scores compare with theirs.
- Bad, because nDCG@10 and MRR can disagree on a variable. A variable whose
  expected targets come first and third has an MRR of 1. Its nDCG@10 is below
  1, though its first target is enough to harmonize it.

**Alternatives considered.**

- nDCG@10 as the main measure, as the evaluator, BEIR and MTEB use it. It
  rewards ranking every expected target high, and gives no credit below rank
  10. One expected target is enough to harmonize a variable.
- Leaving nDCG@10 out: readers who know BEIR and MTEB look for it, and it
  costs one column.
- Recall@k or MAP: they measure how many of a variable's expected targets
  are found. One is enough to harmonize a variable.

**Confirmation.** The tests in
[test_embedding_model_scoring.py](../../tests/test_embedding_model_scoring.py)
check that:

- the ranks and the metrics are right on a fictional benchmark;
- MRR@10 and nDCG@10 count a rank past 10 as 0, and MRR counts it;
- nDCG@10 counts every expected target in the first ten, and MRR only the
  best placed.

### Models are compared in one text format

**Described in:** [scoring.md](scoring.md#text-formats).

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

**Described in:** [scoring.md](scoring.md#text-formats).

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

## Comparing

### `compare` reads the scores that `score` wrote

**Described in:** [comparing.md](comparing.md#running-compare).

**Context.** For each source variable, `score` ranks every target of a
benchmark. It writes the rank of the best-placed
[expected target](#a-benchmark-is-two-files-mappings-and-targets) to
`per_variable.csv`, and each metric's mean to `metrics.csv`. Ranking needs
the model, and a large model ranks slowly on a CPU. `score` names a scores
folder after the model folder's own name. A [run](training.md) of `train`
keeps its scores in its own folder. `score` records the checksums of the
benchmark's files.

**Decision.** `compare` reads each model's scores, and loads no model. Each
mean comes from `metrics.csv`. Each variable's rank, in `per_variable.csv`,
is used for the intervals and the tests. `compare` refuses another model's
scores, scores of another build of the benchmark, and two models' scores of
other variables.

**Consequences.**

- Good, because a comparison needs no GPU. It runs again at once, with other
  scored models or another reference.
- Good, because each mean is the number that `score` reports, and its
  interval comes from the same ranking.
- Good, because a benchmark is matched by its files' checksums, as when
  `score` [checks a run's corpus](#scoring-a-run-uses-its-corpuss-leakage-check).
  So a benchmark built again with other files, under the same version, is
  refused.
- Good, because a run's scores sit inside the run's folder. So `compare`
  finds them after the folder moves.
- Bad, because each model must be scored first, on the same benchmarks and in
  the same format.

**Alternatives considered.**

- `compare` scores each model itself: it would repeat the work of `score`
  each time a comparison changes.
- `compare` scores only the models that have no scores: it would still load
  models, and a large one needs a GPU.
- A warning for scores that do not match, and the comparison continues: its
  tests would match up variables that differ, or builds that differ.

**Confirmation.** The tests check that:

- the scores come back as `score` wrote them, in
  [test_embedding_model_scoring.py](../../tests/test_embedding_model_scoring.py);
- a model that is not scored is refused, with the command that scores it, in
  [test_embedding_model_scoring.py](../../tests/test_embedding_model_scoring.py);
- the scores of another model of the same name are refused, in
  [test_embedding_model_scoring.py](../../tests/test_embedding_model_scoring.py);
- scores of another build of the benchmark are refused, in
  [test_embedding_model_scoring.py](../../tests/test_embedding_model_scoring.py);
- two models' scores that list other variables, or list them in another
  order, are refused, with ids and without them, in
  [test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py);
- a run's scores are read from the run's own folder, also after the run has
  moved, in
  [test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py).

### One reference, which each other model is measured against

**Described in:** [comparing.md](comparing.md#running-compare).

**Context.** A comparison often asks how models compare with one model: the
model in use, or the best setting so far. Testing each of n models against
every other one makes n(n - 1) / 2 tests per metric. The bar plot,
`accuracy_and_mrr`, marks each difference found with a comparison bracket: a
line over the two bars that it compares. A difference between two models can
be taken either way round.

**Decision.** `--reference` names one model. Each other model is tested
against it, and each difference is that model minus the reference. The
reference's bar comes first in each group. Without `--reference`, `compare`
gives each model's intervals and makes no tests.

**Consequences.**

- Good, because the tests ask what a comparison asks: how each model compares
  with the reference. A comparison makes n - 1 tests per metric, not
  n(n - 1) / 2.
- Good, because each difference reads as its own model's gain: a positive
  difference means that the model does better than the reference.
  Estimation plots define a difference so, as the test group minus the
  control ([Ho et al., 2019](https://doi.org/10.1038/s41592-019-0470-3)).
- Good, because each group of bars in `accuracy_and_mrr` has at most n - 1
  comparison brackets, which a reader can follow. Each one starts at the
  reference's bar, the first of its group.
- Bad, because a test between two other models needs a second comparison,
  with one of them as the reference.

**Alternatives considered.**

- Each model against every other one: many tests, and more comparison
  brackets than the bar plot can show clearly.
- A default reference, such as the first model of `--model`: the tests would
  then measure against a model that nobody chose.

**Confirmation.** The tests in
[test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py)
check that:

- the differences are each other model minus the reference;
- the reference is drawn first;
- without a reference, no differences are written, and `compare` says so;
- a reference named by a relative path is the model given by its full path;
- a reference that is not one of the models, even in a folder of the same
  name, or that has no other model, is refused.

### accuracy@k: Wilson intervals, Newcombe's paired interval and McNemar's exact test

**Described in:** [comparing.md](comparing.md#what-it-shows).

**Context.** For accuracy@k, a source variable gives a model a 1 if an
expected target is among its first k suggestions, else 0. So accuracy@k is the
share of variables with a 1. A benchmark is a sample of variables, so the
share has a sampling error, which a confidence interval shows. Two models on
one benchmark rank the same variables. They often agree: when both rank the
expected target first, both get 1 on accuracy@1.

**Decision.** A model's accuracy@k gets a 95% Wilson score interval. Each
other model's difference from the reference gets Newcombe's interval for
paired shares. It combines the two Wilson intervals with the two models'
correlation. The test of that difference is McNemar's exact test: a
binomial test on the variables where the two models differ.

**Consequences.**

- Good, because the Wilson score interval
  ([Wilson, 1927](https://doi.org/10.1080/01621459.1927.10502953)) is the
  interval recommended for a share
  ([Brown et al., 2001](https://doi.org/10.1214/ss/1009213286)). Recent work
  on evaluating models confirms it on small benchmarks
  ([Bowyer et al., 2025](https://proceedings.mlr.press/v267/bowyer25a.html)).
- Good, because the Wilson interval keeps a width at a share of 0 or 1.
- Good, because Newcombe's interval uses the pairing. Two models that agree
  on many variables get a narrower interval than their two Wilson intervals
  would give apart. It is among the intervals recommended for paired shares
  ([Newcombe, 1998](https://doi.org/10.1002/(SICI)1097-0258(19981130)17:22%3C2635::AID-SIM954%3E3.0.CO;2-C);
  [Fagerland et al., 2014](https://doi.org/10.1002/sim.6148)).
- Good, because McNemar's test
  ([McNemar, 1947](https://doi.org/10.1007/bf02295996)) is the standard test
  for two paired shares. It counts only the variables where the two models
  differ, so it can find a difference that two overlapping intervals hide.
  Its exact form, a binomial test on those variables, needs no large-sample
  approximation.
- Good, because `scipy.stats.binomtest` computes both the Wilson interval and
  McNemar's exact test.
- Bad, because neither scipy nor statsmodels computes Newcombe's interval.
  So `compare` computes it by Newcombe's formula, from scipy's Wilson
  limits.
- Bad, because the exact test is conservative when few variables differ: its
  p tends to be larger than it should be
  ([Fagerland et al., 2013](https://doi.org/10.1186/1471-2288-13-91)).
- Bad, because a difference's interval and its test are two methods, so near
  0 they can disagree. Far from 0, the two agree. When an end of the interval
  is close to 0, the interval can just miss 0 while p is just above 0.05, or
  the reverse.

**Alternatives considered.**

- The BCa bootstrap: one method for every metric. But at a share of 0 or 1
  its interval has no width, and on small benchmarks it covers too little
  ([Bowyer et al., 2025](https://proceedings.mlr.press/v267/bowyer25a.html)).
- Clopper and Pearson's exact interval: it always covers at least 95%, but it
  is wider than needed ([Brown et al.,
  2001](https://doi.org/10.1214/ss/1009213286)).
- A Bayesian interval, from a Beta distribution: about as good as the Wilson
  interval for one share ([Bowyer et al.,
  2025](https://proceedings.mlr.press/v267/bowyer25a.html)). For a paired
  difference, it needs sampling code of our own.
- The Wald interval: the share plus or minus 1.96 standard errors. MRR's t
  interval gives it on values of 1 and 0. It can leave the range from 0 to
  1, and it has no width at 0 or 1 ([Brown et al.,
  2001](https://doi.org/10.1214/ss/1009213286)).
- Two Wilson intervals combined as if the models were independent: wider
  than needed, since the two models agree on many variables.
- The paired t-test, as for MRR: one test for every metric. On values of 1
  and 0, it approximates McNemar's test, which the binomial test gives
  exactly.
- `scipy.stats.permutation_test`: on a benchmark of real size, scipy samples
  the swaps instead of trying them all. Its p is then an estimate, with a
  floor set by the number of resamples. The binomial test gives the exact p
  of the same test.
- statsmodels' `mcnemar`: the same exact test, from a library that the
  repository does not otherwise need.
- McNemar's mid-p test, which Fagerland et al. recommend: its p keeps closer
  to the stated rate. But it is not an exact test, so its error rate is not
  kept at or below the stated rate. And `scipy.stats.binomtest` gives no
  mid-p.
- Comparing the two models' intervals: it misses differences that a paired
  test finds.

**Confirmation.** The tests in
[test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py)
check that:

- `metrics.csv` gives a share of 1 a Wilson interval of some width;
- Newcombe's interval is narrower for two models that agree than for two
  that agree only by chance, with the same shares. It keeps some width when
  they agree on every variable;
- Newcombe's interval treats the two models alike: swapping the model and
  the reference turns [low, high] into [-high, -low];
- `differences.csv` gives each difference of accuracy@k Newcombe's interval;
- McNemar's p is right on an example computed by hand: three variables that
  only the reference finds give p = 2 × 0.5³ = 0.25;
- two models that never differ have a p of 1;
- the stars mark the p below each level.

### MRR and MRR@10: the t interval and the paired t-test

**Described in:** [comparing.md](comparing.md#what-it-shows).

**Context.** For MRR, each source variable gives a model a value between 0
and 1: 1 / rank, or 0 past rank 10 for
[MRR@10](#mrr-accuracyk-the-median-rank-and-ndcg). Many of the values are 1.
A benchmark is a sample of variables, so the mean has a sampling error,
which a confidence interval shows. Two models' values match up, variable by
variable. Two good models often give a variable the same rank, so many of
the differences are 0.

**Decision.** A model's MRR gets the 95% t interval of its mean, cut at 0
and 1, the range of MRR. A difference from the reference gets the paired
t-test, and the test's own 95% t interval: the interval of the variables'
differences. scipy computes both.

**Consequences.**

- Good, because the interval and the test come from one t distribution. A
  difference's interval excludes 0 exactly when the test's p is below 0.05,
  so the two cannot disagree.
- Good, because the paired t-test is reliable in IR evaluation, and is its
  most common test. When two systems do not differ, it gives p < 0.05 about
  1 time in 20. This holds across measures and sample sizes
  ([Urbano et al., 2019](https://doi.org/10.1145/3331184.3331259);
  [Ihemelandu & Ekstrand, 2023](https://doi.org/10.1145/3539618.3592004)).
- Good, because the t interval and the t-test are the standard methods for
  a mean ([Student, 1908](https://doi.org/10.2307/2331554)).
- Bad, because the t interval assumes that the mean is about normally
  distributed. Values that pile up at 1 need enough variables for that.
- Good, because the cut never loses the true MRR, which lies between 0 and
  1. An interval that held it before the cut still holds it. The cut
  interval always holds the mean, so it is never empty.
- Bad, because a cut interval is not symmetric around the mean. Its cut
  end is MRR's limit, not a t limit.
- Bad, because the t-test loses power when the differences are skewed. On a
  small benchmark, a permutation test can find a difference that the t-test
  misses.

**Alternatives considered.**

- The BCa bootstrap: it assumes nothing about the shape of the values, and
  adjusts for skew ([Efron,
  1987](https://doi.org/10.1080/01621459.1987.10478410)). But when two models
  give most variables the same rank, its interval of their difference covers
  too little. On a small benchmark, a bootstrap interval covers too little
  too ([Bowyer et al., 2025](https://proceedings.mlr.press/v267/bowyer25a.html)).
- The percentile bootstrap: simpler, but less accurate than BCa when the
  distribution is skewed.
- A paired permutation test: as reliable as the t-test ([Urbano et al.,
  2019](https://doi.org/10.1145/3331184.3331259)), and more powerful when the
  differences are skewed. But it gives no interval, so a second method would
  give the interval, and near 0 the two could disagree.
- The Wilcoxon signed-rank test: it drops the variables where both models
  tie, and it does not test the mean. It is unreliable for means in IR
  evaluation, more so as samples grow
  ([Urbano et al., 2019](https://doi.org/10.1145/3331184.3331259);
  [Urbano, 2026](https://doi.org/10.1145/3805712.3808540)).
- The sign test: unreliable for the same reasons ([Urbano et al.,
  2019](https://doi.org/10.1145/3331184.3331259)).
- A bootstrap test: its p leans toward small values ([Urbano et al.,
  2019](https://doi.org/10.1145/3331184.3331259)).

**Confirmation.** The tests in
[test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py)
check that:

- on 200 variables, the interval is the mean plus or minus t × sd / √200;
- a model's MRR interval that passes 1 ends at 1, and keeps its low end;
- when every variable has an MRR of 1, the interval is [1, 1];
- a difference's interval excludes 0 exactly when the paired t-test's p is
  below 0.05;
- MRR is compared by a paired test: it finds a difference that the spread of
  tied ranks hides from an unpaired test;
- when every variable differs by the same amount, p is 0, or 1 when that
  amount is 0.

### Each paired test is reported alone, with no correction for the number of tests

**Described in:** [comparing.md](comparing.md#what-it-shows).

**Context.** Each test asks: does this model differ from the reference, on
this metric and benchmark? Models are not tested against each other. Two
models and a reference, on 6 metrics and 2 benchmarks, make 2 x 6 x 2 = 24
tests. By chance, a test gives p < 0.05 about 1 time in 20.
A correction for multiple comparisons, such as Holm's, keeps the chance of a
false finding among the 24 below 0.05.

**Decision.** Each p is reported as its test gives it, with no correction. A
comparison bracket marks each comparison with p < 0.05.

**Consequences.**

- Good, because each comparison bracket is its own claim: this model differs
  from the reference on this metric. A claim about one comparison needs no
  correction for the others
  ([Rothman, 1990](https://doi.org/10.1097/00001648-199001000-00010)).
- Good, because a claim that a model beats the reference on every metric
  needs no correction either. It holds at 0.05 when each of its tests gives
  p < 0.05 ([Berger, 1982](https://doi.org/10.1080/00401706.1982.10487790)).
- Good, because a small benchmark keeps its power to find a real difference.
- Bad, because `compare` leaves a claim about the whole set to its reader.
  Picking the best of several models from its tables is such a claim, and
  needs a correction. `differences.csv` gives every p, so that a reader can
  apply one.

**Alternatives considered.**

- Bonferroni's correction: each test is held to 0.05 divided by the number
  of tests, such as 0.05 / 24. It is the simplest correction, and it keeps
  the chance of any false finding in the set below 0.05. But it costs the
  most power, and a small benchmark then finds few differences.
- Holm's correction: the same guarantee as Bonferroni's, by a step-down rule,
  and it never finds fewer differences. It fits a claim about the whole set,
  but still costs power on a small benchmark.
- Dunnett's procedure, made for several treatments against one control
  ([Dunnett, 1955](https://doi.org/10.1080/01621459.1955.10501294)): it fits a
  claim about the whole set too. It assumes normally distributed means, which
  a share near 1 on a small benchmark is not.
- Benjamini and Hochberg's procedure: it bounds the expected share of false
  findings among the differences found. It suits a search through many
  tests, not a claim about each one.

**Confirmation.** The tests in
[test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py)
check that:

- the p of accuracy@1 in `differences.csv` is the p of its own test.

### Four figures

**Described in:** [comparing.md](comparing.md#what-it-shows).

**Context.** The user compares models to answer a few questions
quickly:

- How often, and how high, does each model rank an expected target?
- How does that change as a
  [data curator](#mrr-accuracyk-the-median-rank-and-ndcg) reviews more
  suggestions?
- How does a model's quality relate to its size?
- How much better or worse than the reference is each model?

A figure shows such answers at a glance. `metrics.csv` and
`differences.csv` hold the values.

**Decision.** `compare` draws one figure for each question:

1. `accuracy_and_mrr`: accuracy@1, @5 and @10 as grouped bars, with MRR;
2. `accuracy_by_k`: accuracy@k for every k from 1 to 50;
3. `mrr_by_size`: MRR by the model's number of parameters;
4. `differences_from_reference`, with a reference: each model's difference
   from it.

The bars carry no numbers. A comparison holds at most 8 models.

**Consequences.**

- Good, because `accuracy_and_mrr` answers the first question with grouped
  bars. A bar shows well the share of source variables that
  a model finds. Comparison brackets mark which differences hold.
- Good, because `accuracy_by_k` shows where a model gains as a curator
  reviews more suggestions, beyond the three k of the bars.
- Good, because `mrr_by_size` places each model by its number of
  parameters, on a log scale.
- Good, because `differences_from_reference` shows by how much each model
  differs from the reference, with an interval, on an axis of its own. On the
  0-to-1 axis of `accuracy_and_mrr`, a difference of 0.01 is hard to see.
  This is an estimation plot
  ([Ho et al., 2019](https://doi.org/10.1038/s41592-019-0470-3)). It shows
  how large each difference is, not only whether a test finds it.
- Good, because the figures give the answers at a glance, and the tables
  keep every value. So the bars need no numbers.
- Bad, because a comparison of more than 8 models is refused. Its bars would
  grow too thin to read, and its comparison brackets would fill half of each
  panel.

**Alternatives considered.**

- Tables alone, with no figures: exact, but slow to read. A curve's shape,
  or a difference of 0.01, is hard to see in a column of numbers.
- Numbers on the bars: past a few models they
  crowd the bars, and `metrics.csv` holds every value.
- Points with intervals in place of bars, for the first question: they suit
  any mean, but show a share less plainly than a bar.
- Comparison brackets alone, for the last question, without
  `differences_from_reference`: they say whether two models differ, but not
  by how much.

**Confirmation.** The tests in
[test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py)
check that:

- each figure is written as a PNG, a PDF and an EPS file;
- `differences_from_reference` puts each model at its difference from the
  reference;
- a comparison of 9 models is refused.

### Colors from matplotlib's petroff10, not its default tab10

**Described in:** [comparing.md](comparing.md#colors).

**Context.** The figures give each model a color. The default color sequence
of matplotlib is tab10. It also ships petroff10, a sequence made for readers
with a color vision deficiency. The repository requires matplotlib
3.10 or later, so that it runs on Google Colab.

**Decision.** By default, the figures take their colors from petroff10, in
the order of `--model`.

**Consequences.**

- Good, because petroff10 was chosen for readers with a color vision
  deficiency, and for how people rated its look
  ([Petroff, 2021](https://doi.org/10.48550/ARXIV.2107.02270)).
- Bad, because the bars of `accuracy_and_mrr` carry no marker or pattern,
  only a color. In black-and-white print, a reader must go by the bars'
  order, which the legend follows.

**Alternatives considered.**

- tab10, matplotlib's default: under simulated protanopia
  ([Machado et al., 2009](https://doi.org/10.1109/tvcg.2009.113)), two of
  its colors look nearly the same.
- A pattern on each bar, by matplotlib's hatching: it would keep the bars
  apart in black-and-white print. But up to 8 patterns crowd the bars, and
  matplotlib names no sequence of patterns, as it does of colors.
- [Okabe and Ito's](https://jfly.uni-koeln.de/color/) eight colors, and
  Petroff's eight: matplotlib ships them only from 3.11, above the version
  that the repository requires. Okabe and Ito's also hold black, the color of
  the error bars and the text.

**Confirmation.** The tests in
[test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py)
check that:

- each model's bars take petroff10's color of its place in `--model`.

### A model's color can be given with `--colors`, one per model

**Described in:** [comparing.md](comparing.md#colors).

**Context.** By default, each model takes the
[petroff10](#colors-from-matplotlibs-petroff10-not-its-default-tab10) color
of its place in `--model`. Two comparisons that list other models, or list
them in another order, can give one model two colors. A reader who sees
both, as in one report, may take it for two models. `--model` and
`--benchmark` each take a list.

**Decision.** `--colors` gives one color per `--model`, in the same order, in
any form that matplotlib reads. compare refuses a count that differs from
the models', a color that matplotlib does not read, a fully transparent
color, and one color for two models. Without `--colors`, each model keeps
its place's petroff10 color. The manifest records each color as drawn, in
hex.

**Consequences.**

- Good, because a model can keep one color in comparisons of other models,
  so a reader can follow it from one figure to the next.
- Good, because the colors come as a list, as `--model` and `--benchmark`
  do, and in matplotlib's own forms
  ([matplotlib's guide to colors](https://matplotlib.org/stable/users/explain/colors/colors.html)).
- Good, because the colors are checked before any score is read, so a
  misspelled color stops compare at once.
- Good, because no two models share a color, and none is fully transparent.
  The bars of `accuracy_and_mrr` carry only a color, so either would hide a
  model there.
- Good, because the manifest says what the figures show, whatever form
  each color was given in. `'tab:blue'` and `'#1f77b4'` are both recorded
  as `#1f77b4ff`.
- Bad, because a color can still be hard to see. White is lost on the white
  background. Two colors can look alike, to every reader or only under a
  color vision deficiency, which petroff10 was chosen to avoid. compare
  checks neither.
- Bad, because the list must be kept in step with `--model` by hand. A model
  added in the middle shifts the colors after it, and compare cannot tell,
  since any color fits any model.

**Alternatives considered.**

- A color named per model, as `--color MODEL=COLOR`. seaborn's dictionary
  palette
  ([seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html))
  and plotly's `color_discrete_map`
  ([plotly's discrete colors](https://plotly.com/python/discrete-color/))
  take colors this way. It needs no order, and a model left out keeps its
  default. But each model must be written again exactly as in `--model`,
  and a model id or a run's folder can be long.
- `--palette`, naming another of matplotlib's color sequences: it changes
  every color at once, but a model still takes the color of its place. So
  two comparisons of other models still give it two colors.

**Confirmation.** The tests in
[test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py)
check that:

- in all four figures, each model takes the color that `--colors` gives its
  place, whichever model is the reference;
- `--colors` with fewer or more colors than models, or with one color for
  two models, is refused before anything is written;
- a misspelled color, and a fully transparent one, are refused too;
- the manifest records each model's color as drawn, in hex, with
  `--colors` and without it;
- the command passes `--colors` on to the comparison, and stops with a
  message when the colors do not fit.

### nDCG@10 and the median rank get no interval, test or figure

**Described in:** [comparing.md](comparing.md#where-the-numbers-come-from).

**Context.** `score` reports nDCG@10, as BEIR and MTEB do, and the median
rank, each as one number. A variable can have several expected targets. MRR
and accuracy@k count the best-placed one, and nDCG@10 counts every one. An
interval or a test of nDCG@10 needs each variable's own nDCG@10, from the rank
of every expected target. `per_variable.csv` keeps only the best-placed one's
rank. A curator needs one expected target to map a variable.

**Decision.** `compare` copies nDCG@10 and the median rank into its
`metrics.csv` as plain numbers, with no interval or test, and draws neither.

**Consequences.**

- Good, because readers who know MTEB find nDCG@10.
- Good, because the figures show what a curator needs: where the first
  expected target lands.
- Bad, because a difference in nDCG@10 is not tested.

**Alternatives considered.**

- Each variable's nDCG@10: `score` would keep the rank of every expected
  target, and a library of IR measures would compute it. It adds files and a
  dependency, for a metric that a curator does not need.
- nDCG@10 in the figures: it marks a model down for missing a second
  expected target, which a curator does not need.
- The median rank in the figures, with an interval: it often shows nothing.
  When more than half of the variables rank their expected target first, the
  median rank is 1. It stays 1 when a model moves its worst variables up, and
  its bootstrap interval is [1, 1].

**Confirmation.** The tests in
[test_embedding_model_compare.py](../../tests/test_embedding_model_compare.py)
check that:

- `metrics.csv` copies nDCG@10 from the scores.

## Sources

### Eight public sources, of three kinds

**Described in:** [sources.md](sources.md) and
[licenses.md](licenses.md#training-sources).

**Context.** The model learns from pairs of texts that mean the same thing.
Three kinds of public source hold such texts. Terminologies and ontologies
name medical concepts in several ways: synonyms, abbreviations and
definitions. Collections of standard data elements hold the fields that
studies reuse: their names, questions and values. A survey's data dictionary
describes the variables that it collected. Each source has its own license or
terms. Some sources also mark single items as restricted.

**Decision.** Eight sources are used, of all three kinds. The terminologies
and ontologies are NCIt, seven OBO Foundry ontologies, LOINC and CDISC. The
collections of data elements are NIH CDE, caDSR and PhenX. NHANES is a
survey's data dictionary. Each source is fetched from its publisher, and is
free to use under its own license or terms.

**Consequences.**

- Good, because the three kinds give what the task needs. Terminologies and
  ontologies give the many names of one concept. Data elements and a
  survey's variables give the short names, questions and value lists that
  studies use.
- Good, because each source comes from its publisher, under one license or
  terms. [licenses.md](licenses.md#training-sources) gives each one, and the
  manifests record it.
- Good, because a source's restricted items are left out, as
  [An item is open only when its value is known to allow use](#an-item-is-open-only-when-its-value-is-known-to-allow-use)
  says.
- Bad, because a large terminology whose license is not open to every user
  stays out, though it holds many synonyms. SNOMED CT, for one, needs a
  license for each user, with fees in some countries.
- Bad, because some sources need a free registration before a download, such
  as LOINC.

**Alternatives considered.**

- The UMLS Metathesaurus, which joins many terminologies, NCIt and LOINC
  among them. Downloading it needs a UMLS license. Its
  [license agreement](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement.html)
  prohibits distributing the Metathesaurus or subsets of it, except as an
  integral part of an application. Some of its sources allow only internal
  use. So each source's terms would have to be checked within it, by its
  restriction category.
- SNOMED CT, a large clinical terminology with many synonyms. Each person who
  downloads it needs a license of their own. NLM's
  [licensing page](https://www.nlm.nih.gov/healthit/snomedct/snomed_licensing.html)
  says that it is free in IHTSDO member countries, such as the US. Fees may
  apply elsewhere, so not every user could rebuild the corpus for free.
- dbGaP's public data dictionaries, of studies that no benchmark uses. They
  describe real study variables, in the shape that the task reads. They are
  not used yet. The benchmarks come from dbGaP studies too, so each added
  study must be checked against every benchmark first. A later version may
  add them, first as texts for comparing tokenizers.

**Confirmation.** The tests in
[test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)
check that:

- each source records its license as an SPDX license expression.

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
- Good, because a restriction on a parent covers its children. An NHANES
  data file's constraints cover its variables, and a PhenX protocol's
  availability covers its variables. A LOINC code that sits only in closed
  forms is closed too.
- Good, because each allow-list holds only values whose meaning was checked.
  NHANES's "RDC Only", for one, restricts access to a data file's data, not
  to the documentation that the fetch reads. So it allows use.
- Bad, because a harmless new value closes items until its source's
  allow-list is updated.

**Alternatives considered.**

- A deny-list of the values known to restrict: a new value would let its
  items in, undetected.
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

### OWL files are read by a reader of our own

**Described in:** [sources.md](sources.md#ontologies).

**Context.** [NCIt](sources.md#ncit) and
[the ontologies](sources.md#ontologies) publish
[OWL](https://www.w3.org/OWL/) files in RDF/XML. pronto, rdflib and owlready2
read OWL files.

**Decision.** `sources/owl.py` reads OWL files in RDF/XML in one streaming
pass, with defusedxml.

**Consequences.**

- Good, because one OWL reader reads NCIt and every ontology.
- Good, because defusedxml refuses entities and external references in the
  XML.
- Bad, because the reader is code of our own to maintain. A change in the
  files' layout needs a change in the reader.

**Alternatives considered.**

- rdflib: it builds the whole graph in memory. It was many times slower than
  this reader on the base file of HP, the Human Phenotype Ontology. Only this
  file was timed with both readers.
- pronto: it is not a dependency. It knows the annotations that OBO defines,
  such as a synonym and its scope. NCIt writes its synonyms with properties of
  its own, which pronto does not know. On the ontologies' base files, pronto
  kept one definition per term and no synonym types.
- owlready2, not tried: it loads each file into a database of its own before
  it can be read.
- The ontologies' OBO files, read with pronto: EVS publishes NCIt in OWL and
  as a flat file, not in OBO. So NCIt would still need an OWL reader. pronto's
  strict OBO parser fails on the OBO file of Uberon, the anatomy ontology.
- The ontologies' OBO Graphs JSON files, read with `json`: EVS publishes no
  such file for NCIt. So NCIt would still need an OWL reader.

**Confirmation.** The tests in
[test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)
check that:

- the OWL reader keeps each statement's exact text.

### PhenX's export is read by a reader of our own

**Described in:** [sources.md](sources.md#phenx).

**Context.** PhenX publishes a MySQL database export. Neither Python nor the
repository's dependencies read MySQL exports.

**Decision.** `sources/phenx.py` reads PhenX's export as text: a regex
tokenizer reads its `INSERT` statements, and checks each row's field count.

**Consequences.**

- Good, because the reader needs no MySQL server.
- Bad, because the reader is code of our own to maintain. A change in the
  export's layout needs a change in the reader.

**Alternatives considered.**

- sqlite3: it cannot run the export's `CREATE TABLE` statements. Turning only
  `\'` into SQL's own escape leaves many variables wrong, from `\"`, `\n`
  and `\r`.
- sqlparse: it is not a dependency. It splits SQL into tokens, such as one
  quoted string, and groups them. A token keeps its text as written, with its
  quotes and backslash escapes. So the code that reads each value would still
  be ours.
- mysql-to-sqlite3 and SQLAlchemy: both need a running MySQL server.

**Confirmation.** The tests in
[test_embedding_model_sources.py](../../tests/test_embedding_model_sources.py)
check that:

- the PhenX reader reads rows with quotes, escapes and nulls.

### NCIt's replacements come from its merge records

**Described in:** [sources.md](sources.md#ncit).

**Context.** A retired NCIt concept can name the concept that replaced it.
Each release folder holds the release's history file. Each of its records has
an action, such as `merge` or `retire`. A `merge` record names the concept
that a retired concept was merged into. A `retire` record names the concept's
old parent, as EVS's
[ReadMe_history.txt](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/archive/26.09d_Release/ReadMe_history.txt)
says.

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

- EVS's replacements service, a live service: it serves only its current
  state. And it gives a retired concept's old parent as its replacement.
- `retire` records as replacements too: they name the old parent, a broader
  concept, not one of the same meaning.
- `split` records as replacements: a split names the concepts that each take
  a part of the old concept's meaning. Each of them would get all its names.

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
  experiment can choose pairs by these marks, though `train` reads none of them
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
- A cap on each item's pairs, as SapBERT caps a concept: a few items give far
  more pairs than most. Many of their names are short variable names that
  forms give one caDSR element. Each is a real variable name, so a cap would
  drop true pairs.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- a fictional item gives the pairs that the test lists;
- each of them joins a text to the canonical.

### Four kinds of pair, once for each value list

**Described in:**
[pairs.md](pairs.md#the-shape-a-star-around-each-item) and
[pairs.md](pairs.md#values).

**Context.** An item has a canonical, and often other names and a
description. Some items have no other name, only a description. Some have a
second description, such as another definition. A target's text holds a
description only when its data model gives one. An item can have several
value lists, as an NCIt concept can.

**Decision.** Each other name gives a `name` pair, with the canonical alone,
and a `name_description` pair when the item has a description. Each other
description gives a `description_description` pair. An item with no other
name pairs its canonical with its description, as `name_to_description`. An
item with several value lists repeats its pairs for each, with that list's
values in the positive.

**Consequences.**

- Good, because a target comes with a description or without one, and the
  model learns to match both.
- Good, because an item with no other name still gives a pair,
  `name_to_description`: its canonical, matched with its description.
- Good, because two descriptions of one meaning teach the model that two
  wordings of a definition match.
- Good, because each positive shows values that the item really has. Lists
  are never merged, so a positive never mixes two lists.
- Bad, because an item with a description gives each name twice. So items
  with descriptions weigh more in the corpus.
- Bad, because an item with several lists repeats its names once for each
  list.

**Alternatives considered.**

- Only `name` pairs: the model would never see a description, though many
  targets have one.
- One pair for each name, with the description when there is one: an item
  with a description would never be a positive without it. Yet a data model
  can name the same thing without a description.
- One pair for each name, with the item's value lists merged: a positive
  would show values that no single list has.
- Pairs for the first value list only: the other lists' values would be
  lost.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- each name gives a `name` and a `name_description` pair, and another
  description gives a `description_description` pair;
- an item with no other name pairs its canonical with its description;
- an item with two lists gives its pairs once for each, with each list
  named.

### An ambiguous name is paired with each item that it names

**Described in:** [pairs.md](pairs.md#ambiguous-names).

**Context.** Some names are ambiguous: one name stands for several items. An
abbreviation can stand for several concepts, as `MS` stands for multiple
sclerosis and for mitral stenosis. A study can use such a name for either
meaning. Training uses in-batch negatives: in one batch, each pair's positive
is a negative for the other pairs' anchors.

**Decision.** An ambiguous name is an anchor of each item that it names. None
of its pairs is dropped, and no item is chosen over the others. Each source's
pairs manifest counts these names, as `ambiguous_texts`.

**Consequences.**

- Good, because each of these pairs is true: the name stands for each item.
- Good, because studies use ambiguous names too. A model that never saw them
  could not rank any of their meanings.
- Good, because `ambiguous_texts` shows how many such names a source has. So
  a new release that adds many of them is seen.
- Bad, because a name has one embedding. The model places it between its
  items, so none of them ranks clearly first.
- Bad, because two of its pairs can share a batch. Each one's positive then
  counts as a negative for the other.
- Bad, because `ambiguous_texts` counts within one source. A name that two
  sources give to different items is not counted.

**Alternatives considered.**

- Dropping an ambiguous name: true pairs would be lost, and studies use such
  names.
- Keeping one of its items, such as the first: the choice would be
  arbitrary, and the other pairs would be lost.
- The `NO_DUPLICATES` sampler, which keeps a text from appearing twice in a
  batch.
  [The defaults start from Hugging Face's recipe for a static model](#the-defaults-start-from-hugging-faces-recipe-for-a-static-model)
  gives why it is not used.

**Confirmation.** The tests in
[test_embedding_model_pairs.py](../../tests/test_embedding_model_pairs.py)
check that:

- an ambiguous name that two items give keeps a pair for each, and counts
  once in `ambiguous_texts`.

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
  its texts, even one that another current item has. That text then becomes
  an [ambiguous name](#an-ambiguous-name-is-paired-with-each-item-that-it-names),
  an anchor of both items. Removing copies does not help, since the two pairs
  have different positives.

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
  texts in, undetected.

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
- Good, because within a source, a current item's copy comes first, then a
  draft's, then a retired item's. So a pair is marked `retired` only when no
  current or draft item gives it.
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
targets. Its score there would look too good. An anchor with a described
canonical gives twin pairs, a `name` pair and a `name_description` pair.
Often only the second matches a benchmark row. Some sources hold
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
[fields](pairs.md#pairsparquet) of a pair: `anchor_name`,
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
in three ways:

- `=`: they are equal;
- `~`: they share at least 8 words in a row (`min_overlap_words`);
- `>=`: a side holds every word of a benchmark name of 2 or more words
  (`min_name_words`), and the other side matches the row's description.

The source and the target match on different sides.

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
each once after normalisation. A run refuses a tokenizer of another corpus
or build. Four [diagnostics](tokenizer.md#diagnostics) are measured on the
same texts:

- the fertility, the tokens per word;
- the share of words split;
- the unknown rate;
- the share of the vocabulary used.

**Consequences.**

- Good, because the tokenizer cuts validation and benchmark texts as it cuts
  any new text.
- Good, because nothing that makes a choice looks at the data that judges the
  result.
- Good, because a tokenizer of another corpus, or of another build, is
  refused. Its training pairs can hold this build's validation pairs. The
  tokenizer's manifest records the pairs that it learned from. They are
  compared by their rows and content, as every manifest compares a table. So
  the same pairs, written again on another machine, are still accepted.
- Good, because
  [Rust et al. (2021)](https://doi.org/10.18653/v1/2021.acl-long.243) found,
  across languages, that the fertility and the share of words split matter.
  A lower fertility and a smaller share of continued words, their name for
  words split, "correlate with an increase in downstream performance".
- Bad, because the diagnostics are measured on the training texts, which the
  tokenizer learned from. So they look a little better than on new texts, and
  are a sanity check, not a measure of quality. A broken tokenizer still
  gives bad numbers on them. Examples are an unknown
  rate above 0, and a high fertility when words are cut into single letters.

**Alternatives considered.**

- Learning the vocabulary from the validation and benchmark texts too: the
  vocabulary would then depend on the data that judges the model. And a word that only
  validation texts hold would get a token whose row training never updates.
- A pretrained tokenizer, such as a released model's: its vocabulary was
  learned from other texts, and its size is fixed. A rare word of the corpus,
  such as a variable's short name, would more often be cut into pieces.
- A note, rather than a refusal, for a tokenizer of another corpus or
  build. The run would then train on a vocabulary that this build's
  validation texts may have shaped.

**Confirmation.** The tests check that:

- only training texts shape the vocabulary
  ([test_embedding_model_tokenizer.py](../../tests/test_embedding_model_tokenizer.py));
- spellings of one text count once
  ([test_embedding_model_tokenizer.py](../../tests/test_embedding_model_tokenizer.py));
- a run refuses a tokenizer of another corpus or another build. Its message
  gives the command that trains one on this build, with the same vocabulary
  size and minimum frequency
  ([test_embedding_model_training.py](../../tests/test_embedding_model_training.py));
- a run keeps a tokenizer whose pairs were only written again
  ([test_embedding_model_training.py](../../tests/test_embedding_model_training.py)).

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
- Bad, because the BERT here uses only `[UNK]` and `[PAD]`. It does not yet
  add `[CLS]` or `[SEP]` to a text, or train masked tokens.
- Bad, because the `[UNK]` row never trains. Every character of the training
  texts is in the vocabulary, so `[UNK]` never occurs in them, and its row
  keeps its random start.

**Alternatives considered.**

- `[PAD]` and `[UNK]` alone: a later BERT setting that needs `[CLS]`,
  `[SEP]` or `[MASK]` would need another tokenizer, with every id moved.

**Confirmation.** The tests in
[test_embedding_model_tokenizer.py](../../tests/test_embedding_model_tokenizer.py)
check that:

- a trained tokenizer gives `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]` and `[MASK]` the
  ids 0 to 4.

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
  the ids in memory. An experiment trains many runs at once, each with its copy.

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

**Decision.** A config has four sections: `data`, `model`, `loss` and
`training`. `model` holds the
[architecture and its settings](#the-model-section-holds-the-architecture-and-its-settings).
`training` holds the trainer's arguments, under their own names. The settings
become the defaults of one `HfArgumentParser` over their dataclasses. A flag
then overrides any setting. A section or a setting that does not exist is
refused.

**Consequences.**

- Good, because the trainer's settings work without code of ours, under the
  names that the trainer's documentation gives them. A run refuses three of
  them: [`prompts` and `router_mapping`](#a-run-refuses-the-trainers-prompts-and-router_mapping),
  and [`load_best_model_at_end`](#the-model-kept-is-the-model-at-the-last-step).
- Good, because a flag changes one setting for one run, without a new
  config. The run's `config.yaml` holds the final values.
- Good, because argparse reads a default that is text by the setting's type.
  So `1e-3` in a config becomes a number.
- Good, because the sections follow the parts that the trainer of
  [sentence-transformers](https://sbert.net/docs/sentence_transformer/training_overview.html)
  takes: the data, the model, the loss and the training arguments.
  Hugging Face's transformers also keeps a model's config, saved with its
  weights, apart from the `TrainingArguments` that only the trainer reads.
  So the model's shape and the training's settings never mix.
- Bad, because the flags have no sections, so each name must be unique
  across them. A test checks it.

**Alternatives considered.**

- `HfArgumentParser.parse_yaml_file`, the reader of Hugging Face's example
  scripts: it takes a flat file, so the sections would be lost.
- A flag of our own for each setting: each setting of the trainer would need
  code of ours, under a name of ours.
- [Hydra](https://hydra.cc/) or [OmegaConf](https://omegaconf.readthedocs.io/),
  the common config tools of machine learning: they would replace only the
  reading of the config and its overrides. The checks of each architecture
  would stay our code. Overrides would be written as
  `training.learning_rate=1e-2`, not as the trainer's own flags. And their
  typed configs refuse some of the trainer's settings. OmegaConf 2.4 refuses
  `batch_sampler`. Hydra 1.3 needs an older OmegaConf, and also refuses
  `report_to`.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a flag overrides the config, and the config overrides the defaults;
- `1e-3` in a config becomes a number;
- an unknown section or setting is refused;
- no two settings share a name, in any section or architecture.

### The model section holds the architecture and its settings

**Described in:** [training.md](training.md#the-config).

**Context.** A run trains one of two architectures, `static` or `bert`. A
static model has one setting, `embedding_dim`. A BERT has eight, from
`BertConfig`. A run uses only its own architecture's settings.

**Decision.** The `model` section holds `architecture` and that
architecture's settings. A setting of the other architecture is refused, in
the config or as a flag.

**Consequences.**

- Good, because the model's kind and its shape sit in one section. A
  transformers `config.json` keeps them together too:
  [BERT's](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json)
  has `model_type` beside `hidden_size` and the other settings.
- Good, because the section takes only its own architecture's settings. So
  a static model's config cannot carry BERT's by mistake.
- Bad, because the settings that the section takes depend on its
  `architecture`. A reader learns them from that architecture's table in
  [training.md](training.md#architectures), not from the section's name.

**Alternatives considered.**

- A section of its own for each architecture's settings, beside `model`.
  A config would then choose the kind of model in one section, and set its
  shape in another.
- No `architecture` setting: the architecture follows from the section that
  a config holds, `static` or `bert`. A BERT with every default would then
  need an empty `bert` section.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a BERT config reads its settings from the model section;
- a setting of the other architecture is refused, in the model section or
  as a flag;
- a setting in the wrong section names the section where it belongs, with
  its architecture;
- a run's `config.yaml` holds its architecture's settings in the model
  section, and reads back as the settings that trained it.

### A run of the same name with other settings is refused

**Described in:** [training.md](training.md#running-train).

**Context.** Every step's new folder replaces an earlier one of the same
name. `corpus build` and `compare` take that name on the command line. A run
takes it from its config's file and its seed. So a flag, or an edit to the
config, changes a run's settings but not its name. A run's `config.yaml`
records the settings that trained it, as `TrainingArguments.to_dict` gives
them.

**Decision.** Before it trains, a run compares its settings with the
`config.yaml` of an earlier run of the same name. With the same settings,
the new run replaces the earlier one. With other settings, or a
`config.yaml` that cannot be read, the run stops and names the settings that
differ.

**Consequences.**

- Good, because a run with a flag cannot replace a run that other settings
  trained, without a word.
- Good, because the check comes before training, so it costs no training
  time.
- Good, because a run trained again with the same settings replaces the
  earlier one, as a build of the same name does in every step.
- Good, because the settings are compared as built, in the form that
  `config.yaml` records. So `report_to: wandb`, which the trainer turns into
  a list, still matches.
- Bad, because any changed setting stops the run, even one that leaves the
  model the same, such as `logging_steps`.

**Alternatives considered.**

- Only the docs say that a flag keeps the name. A run trained with a flag
  would still replace the earlier one, without a word.
- A run named after its flags, such as `my_run_s1_learning_rate0.01`. Names
  would grow long and depend on the flags' order, unlike every other name of
  the pipeline.
- An option such as `--overwrite`, without which any earlier run is refused.
  A run trained again with the same settings would need it too, unlike the
  other steps' builds.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a run with other settings stops before it trains, and keeps the earlier
  run;
- a run with the same settings trains and replaces the earlier one, though
  `config.yaml` records `report_to` as a list;
- an earlier run whose `config.yaml` cannot be read is kept.

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
- Good, because the defaults work on every machine. transformers refuses
  bf16 on a GPU without it, and on a CPU without `use_cpu`. So it stays off,
  as in the library, until a config turns it on.
- Bad, because five epochs is a starting point, not a measured choice. The
  validation curve in the run's
  [`trainer_state.json`](training.md#trainer_statejson) shows whether a corpus
  needs more, and an experiment tunes the number.
- Bad, because a run on a GPU trains in 32 bits unless its config turns bf16
  on.
- Bad, because the recipe was made for general text, not for metadata. An
  experiment may find better settings, and tests them as arms.
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
- The recipe's bf16: transformers refuses it on a GPU without it, and on a
  CPU without `use_cpu`. A config turns it on.
- The recipe's `MatryoshkaLoss`: it also trains shorter embeddings, and a
  model here is used at its full length.
- The recipe's evaluation, logging and checkpoints every 250 steps: they
  suit one long epoch, and a run here keeps no checkpoint.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a config that names only its data trains at the recipe's batch size, warmup,
  width and learning rate, for five epochs.

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
  the average of its token vectors. So a comparison of the two does not
  measure pooling. It measures the other differences: BERT's layers and, at
  the defaults, the width, the learning rate and dropout.
- Good, because the average, mean pooling, is the default of
  sentence-transformers' `Pooling` and of Sentence-BERT
  ([Reimers & Gurevych, 2019](https://doi.org/10.18653/v1/d19-1410)).
- Bad, because a BERT is much larger and slower than a static model, to train
  and to use.

**Alternatives considered.**

- A pretrained BERT, fine-tuned on the corpus: its pretraining texts are
  outside the leakage check.
- Masked-language pretraining on the corpus's own texts before the pair
  loss, as the BERT paper pretrains: not built yet. Once it is built, an
  experiment can compare it as an arm. The tokenizer keeps `[MASK]` for it.
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

### BERT defaults to BERT-Small's shape and 128 tokens

**Described in:** [training.md](training.md#bert-a-transformer-embedding-model-with-berts-architecture).

**Context.** `BertConfig`'s documentation says that its defaults "will yield a
similar configuration to that of the google-bert/bert-base-uncased" model: 12
layers of 768 dimensions. Google's release of smaller BERT models names
BERT-Small: 4 layers of 512. The default batch is 2,048 pairs. A batch is
padded to its longest text, up to `max_position_embeddings`, 512 in
`BertConfig`. The BERT paper notes that "attention is quadratic to the
sequence length".

**Decision.** A BERT's shape defaults to BERT-Small's, and its length to 128
tokens. Its other settings default to `BertConfig`'s.

**Consequences.**

- Good, because BERT's default shape is a standard one: BERT-Small, from
  Google's release ([Turc et al., 2019](https://doi.org/10.48550/ARXIV.1908.08962)).
  Its feed-forward part is 4 times the width, as the BERT paper sets it.
- Good, because the BERT paper pretrains with "sequence length of 128 for
  90% of the steps" (Appendix A.2). Attention's work is quadratic in the
  length, so at 128 tokens it is a sixteenth of its work at 512.
- Good, because BERT's other defaults are read from `BertConfig` itself, so
  the code copies none of its numbers.
- Bad, because BERT-Small is a guess at a size that suits a corpus of short
  texts. Other shapes still need to be tested.
- Bad, because a text longer than 128 tokens is cut, in training and in
  scoring.

**Alternatives considered.**

- `BertConfig`'s own shape, BERT-base: 12 layers of 768 dimensions are large
  and slow to train from scratch, at a batch of 2,048 pairs.
- `BertConfig`'s own length, 512 tokens. One long text pads its whole batch
  of 2,048 pairs to 512 tokens, so a run's memory would depend on the
  corpus's longest text.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- a BERT config gets BERT-Small's shape and 128 tokens.

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
  language modelling, not for a contrastive loss. An experiment tests other rates
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
the last step. Its learning rate decays linearly to zero over the run. An
experiment compares its runs by their validation metrics.

**Decision.** A run keeps the model at the last step, and saves no
checkpoint. A config that sets `load_best_model_at_end` or
`resume_from_checkpoint`, or a `save_strategy` other than `no`, is refused.
The run writes that model's validation metrics.

**Consequences.**

- Good, because the validation pairs are used once: to compare runs. Picking
  the best of many evaluations within a run would favour a lucky one. The same
  pairs would then compare it with other runs.
- Good, because the last steps run at a learning rate near zero, so the model
  changes little at the end.
- Good, because nothing is saved during training, so a run writes only its
  model.
- Good, because a setting that the run cannot honour stops it at once,
  rather than doing nothing. The trainer does not read
  `resume_from_checkpoint` itself
  ([`TrainingArguments`](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer#transformers.TrainingArguments)),
  and another `save_strategy` would write checkpoints that nothing reads.
- Bad, because a run that peaks early keeps a later, worse model. The loss
  curve shows it, and an experiment can then try fewer epochs.

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
- a config that sets `load_best_model_at_end` or `resume_from_checkpoint` is
  refused;
- a `save_strategy` other than `no` is refused, in the config or as a flag,
  and `no` is read.

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

- Good, because an experiment can tune on a ranking, as the benchmarks score one,
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
where each text starts. CUDA reads `CUDA_VISIBLE_DEVICES` when it starts. The
command line imports torch only when a run reads its config.

**Decision.** Before it reads its config, a run sets `CUDA_VISIBLE_DEVICES`
for its own process. It keeps the first GPU that the variable lists, or GPU 0
when the variable is unset. When the variable lists several, the run names
them, and the one that it keeps.

**Consequences.**

- Good, because a run's batch, and so its loss's in-batch negatives, does not
  depend on the number of GPUs.
- Good, because a user changes nothing. Other processes keep their own
  `CUDA_VISIBLE_DEVICES`.
- Good, because the setting comes before CUDA starts, so it always takes
  effect, and nothing needs to check it.
- Bad, because the first GPU may be busy with another process. The run does
  not look for a free one; `CUDA_VISIBLE_DEVICES` chooses another.
- Bad, because a notebook that has used CUDA before it calls `train` keeps
  every GPU that CUDA saw. Such a notebook sets `CUDA_VISIBLE_DEVICES`
  itself, before it imports torch.

**Alternatives considered.**

- DataParallel, the trainer's default: the batch would grow with the number
  of GPUs, and the flat ids and their offsets would be split apart.
- Stopping, with a message to set `CUDA_VISIBLE_DEVICES`: on a machine with
  several GPUs, every run would need it, as
  `CUDA_VISIBLE_DEVICES=0 python -m embedding_model train ...`. GPU 0 is a
  sensible default.
- Looking for a free GPU: "free" has no clear definition, and the run would
  have to ask NVIDIA's management library before CUDA starts.
- Setting the trainer's `_n_gpu` to 1: `transformers` does not document that
  attribute, so a release could change it.
- DistributedDataParallel (DDP), launched with `torchrun` or `accelerate`:
  each GPU would rank its pairs only against its own batch. The loss's
  `gather_across_devices` joins the batches, and it is off by default. Its
  docstring warns that it may slow training and run out of memory. One GPU
  already trains the models here at a batch of 2,048 pairs.
- Counting the GPUs first, and narrowing the list when there are several.
  Whether the list can still change after a count depends on how torch
  counts. NVIDIA's management library reads the list again, and the CUDA
  driver keeps the first one it read. So the count needed two checks of its
  own, for cases that the command line does not reach.

**Confirmation.** The tests in
[test_embedding_model_training.py](../../tests/test_embedding_model_training.py)
check that:

- with `CUDA_VISIBLE_DEVICES=2,3`, the run keeps GPU 2, and names both;
- without a list, it keeps GPU 0;
- a list of one GPU, or an empty one, stays as it is;
- `train` keeps to one GPU before it reads its config;
- reading a config builds no training settings.

### A run's seed is given on the command line, not in the config

**Described in:** [training.md](training.md).

**Context.** Runs of one config at several seeds differ only by their seed.
A run is named after its config and its seed. An experiment trains each of
its arms at several seeds.

**Decision.** `train` takes the seed as `--seed`, which it needs. It overrides
a seed in the config. An experiment lists its seeds, and refuses
`training.seed` in its base and its arms.

**Consequences.**

- Good, because one config trains at many seeds, with no copy of it for each
  seed.
- Good, because a run's name says its seed, as in `run_example_s0`.
- Bad, because a config alone does not say how to train a run: the seed is
  needed too.

**Alternatives considered.**

- The seed in the config: a copy of the config for every seed.
- A default seed: two runs of one config could share a seed by accident.

**Confirmation.** The tests check that:

- the seed given reaches the run's settings, and a seed below 0 is refused
  ([test_embedding_model_training.py](../../tests/test_embedding_model_training.py));
- the run is named after its config and its seed
  ([test_embedding_model_training.py](../../tests/test_embedding_model_training.py));
- an experiment refuses `training.seed` in its base and in an arm
  ([test_embedding_model_experiments.py](../../tests/test_embedding_model_experiments.py)).

## Experiments

### One experiment asks one question, in one file

**Described in:** [experiments.md](experiments.md#the-experiments-file).

**Context.** Model development compares versions of a run's config, each at
several seeds. Which change to try next depends on what the last comparison
found. It also takes judgement, after a tie, a result near the margin, or a
change that costs more. A comparison is fair when its arms, its seeds and
its decision rule are fixed before its runs are read.

**Decision.** An experiment is one file: a base, its arms, its seeds and its
decision rule. It asks one question. The next experiment is another file,
whose base names an arm of the one before.

**Consequences.**

- Good, because each question keeps its record, its file, beside its runs.
- Good, because the user chooses the next question, from what the results
  show.
- Good, because the rule is written before the runs are read, and stays
  fixed for them: a [changed file is refused](#a-changed-experiment-file-is-refused-rather-than-trained-again).
- Bad, because a series of experiments is many small files.

**Alternatives considered.**

- One file for a whole series, which trains each next comparison by the
  rule: it would need a workflow engine. And the next question often needs a
  person, after a tie or a result near the margin.
- No file: each arm trained by hand, and compared when asked. The rule would
  be given again at each comparison, and could change between them.

**Confirmation.** The tests in
[test_embedding_model_experiments.py](../../tests/test_embedding_model_experiments.py)
check that:

- a file is read with the decision rule's defaults, or its own;
- the experiment is named after its file.

### An experiment's file is checked with pydantic

**Described in:** [experiments.md](experiments.md#the-experiments-file).

**Context.** The user writes an experiment's file by hand. It is nested
YAML: the decision rule's fields, and each arm's test and settings by
section. Most of its checks are of form: a field's type, its allowed values
and its default. A few are rules of an experiment, such as different seeds.
ai-harmonization depends on [pydantic](https://docs.pydantic.dev/), and its
`SimpleDataModel` is a pydantic model. A run's config is read by
`HfArgumentParser`.

**Decision.** pydantic models check the file. What pydantic can declare is
declared: a field's type, its values, its bounds, its length and its default.
A validator in the same model holds each rule that pydantic cannot declare,
such as different seeds. Each problem is reported with the file and the field.

**Consequences.**

- Good, because a field's type, values and default are written once, in the
  model, and pydantic checks them.
- Good, because every message has one form: the field, what is wrong, and
  what was given.
- Good, because pydantic is a dependency already, which ai-harmonization uses
  for its own data model.
- Good, because each validator holds one rule of an experiment, and nothing
  that pydantic checks itself.
- Bad, because pydantic's messages are in its own words, such as "Input should
  be a valid integer".
- Bad, because some rules of form still need a validator. pydantic 2 has no
  check that a list's items differ. YAML reads an empty field as None, and
  pydantic does not replace None with the field's default.

**Alternatives considered.**

- A check of our own for each field: a check and a message for each mistake,
  in as many forms.
- A JSON Schema, checked with jsonschema: a schema file beside the code, from
  a package that ai-harmonization does not depend on directly.
- `HfArgumentParser`, as for a run's config: it reads flat settings, while the
  file nests its arms and their sections.

**Confirmation.** The tests in
[test_embedding_model_experiments.py](../../tests/test_embedding_model_experiments.py)
check that:

- a file that breaks a rule of an experiment, or of its fields, is refused,
  with the file, the field and what is wrong;
- an empty section, question or decision rule is read as its default.

### The base can be an arm of an earlier experiment, whose runs are reused

**Described in:** [experiments.md](experiments.md#building-on-an-earlier-experiment).

**Context.** The next experiment most often starts from an arm of the last
one. That arm's runs, at the same seeds, are already trained, from the same
config.

**Decision.** An experiment's base can name an arm of an earlier experiment,
as `<experiment>/<arm>`. The base is then that arm's full config, and the
arm's runs at the same seeds are reused. The other seeds train in the new
experiment.

**Consequences.**

- Good, because no config is trained twice at the same seed.
- Good, because the selection is written in the new file, by name, where a
  user sees which arm was chosen.
- Bad, because the earlier experiment's folder must stay, since the new one
  reads its runs. Without it, a base can still give the arm's settings in
  full, from a copy of the arm's `configs/<arm>.yaml`. Its runs then train
  again.
- Bad, because a base run that the earlier experiment is still training
  stops the new one, until it ends.

**Alternatives considered.**

- Training the base again in each experiment: the same runs, twice.
- A path to a run config as the base: its runs could not be found, and a
  first experiment would need two files. The first experiment gives its
  base's settings in its own file instead.

**Confirmation.** The tests in
[test_embedding_model_experiments.py](../../tests/test_embedding_model_experiments.py)
check that:

- the earlier arm's runs are reused, and a seed that it lacks trains here,
  from the earlier arm's config;
- a base run that the earlier experiment is still training stops the new
  one;
- an earlier experiment or arm that does not exist is refused.

### An experiment queues its runs itself, rather than submitting each as a job

**Described in:** [experiments.md](experiments.md#gpus-and-cpus).

**Context.** An experiment trains many runs, and something must decide when
and where each one trains. The user trains on a laptop with a GPU, on CPUs
alone, or on a cluster with SLURM. `train` trains on the machine where it
runs, and on a cluster the user runs it inside a job. A SLURM job can hold
several GPUs of one node. A run trains on one GPU, as
[its entry](#a-run-trains-on-one-gpu-the-first-one-it-sees) decides.

**Decision.** `experiment train` queues its runs itself. Whenever one of its
GPUs is free, it starts the next run, as `train` with that GPU in
`CUDA_VISIBLE_DEVICES`. Without a GPU, one run trains at a time. On a
cluster, the experiment is submitted as
[one job](experiments.md#on-a-cluster-with-slurm), and its runs train inside
it, not as jobs of their own.

**Consequences.**

- Good, because one command trains on a laptop and on a cluster. On a
  cluster, it runs inside a job, as `train` does.
- Good, because each run is the public `train` command, which can be run
  again by hand.
- Good, because it works with any scheduler, or none: the code holds none.
- Good, because the experiment uses only the GPUs that its job shows it. A
  job that asks for its GPUs with `--gpus` cannot fill a shared cluster.
- Bad, because the experiment's job holds all its GPUs until the last run
  ends, although the GPUs of finished runs then wait.
- Bad, because the experiment's job holds the GPUs of one node. One job per
  run could train on many nodes at once, so on a large cluster it would
  finish sooner.

**Alternatives considered.**

- One SLURM job per run, through an option such as `--slurm`: faster on a
  large cluster. But it is code for SLURM alone, and its command has another
  shape than `train`'s. It can also take every free GPU of the cluster at
  once.
- [submitit](https://github.com/facebookincubator/submitit): it submits
  pickled Python functions, so a job would not be a public command, and its
  requeues are not needed.
- A SLURM job array: each task would look up its run from a number, so a job
  would not be a public command either.
- [Hydra](https://hydra.cc)'s multirun, with its submitit launcher: a second
  config system beside `train`'s, whose sweeps do not express arms against a
  base.

**Confirmation.** The tests in
[test_embedding_model_experiments.py](../../tests/test_embedding_model_experiments.py)
check that:

- with GPUs 2 and 3 visible, each run sees one of them;
- without a GPU, each run keeps this process's environment;
- each run is `train`'s public command, with its arm's config and its seed;
- a failed run does not stop the others, and the next training retries only
  it.

### A changed experiment file is refused, rather than trained again

**Described in:** [experiments.md](experiments.md#a-changed-file).

**Context.** An experiment's runs are trained from its file, and can take
hours of GPU time. `corpus build` and `compare` replace an output of the
same name. `train` refuses a run of the same name that other settings
trained. A file can be edited after its results are seen.

**Decision.** The first training keeps a copy of the file. Each command reads
the file and the copy, with the decision rule's defaults, and refuses a file
whose fields differ. The message names those fields, and says to delete the
experiment's folder to start again, or to give the file a new name.

**Consequences.**

- Good, because an experiment's folder holds the runs of one question.
- Good, because the rule cannot change after the results are read.
- Good, because a comment, another order of the fields, or a default
  written out changes nothing.
- Good, because `train` keeps a run the same way. It compares settings, not
  bytes, and
  [refuses a run of the same name](#a-run-of-the-same-name-with-other-settings-is-refused)
  that other settings trained.
- Bad, because the comparison is code of ours. Other checks of a file here
  compare SHA-256s.

**Alternatives considered.**

- Training the changed file again from the start, as `corpus build`
  replaces a corpus of the same name: one edited line would discard the
  experiment's runs. It could also change the question after its results
  were seen.
- Comparing the two files' SHA-256s, as other checks of a file here do: a
  changed comment would need a new name, or a deleted folder.
- Comparing the two files as YAML reads them, without the defaults: a default
  written out would count as a change.

**Confirmation.** The tests in
[test_embedding_model_experiments.py](../../tests/test_embedding_model_experiments.py)
check that:

- the first training keeps a copy of the file;
- a file of the same name whose seeds differ is refused, and the message
  names the seeds and both ways out;
- a comment, another order of the fields, or a default written out is no
  change.

### An experiment's status comes from its run folders

**Described in:** [experiments.md](experiments.md#following-it).

**Context.** An experiment's runs train over hours, inside a job or on a
laptop. A run that `train` finished has its manifest. A run that is training
has a staging folder beside its folder.

**Decision.** `experiment status` reads each run's folder: done with its
manifest, started with a staging folder, and not started otherwise.
`experiment train` trains the runs that are not done.

**Consequences.**

- Good, because there is no record of jobs to keep, and it works without a
  scheduler.
- Good, because training again uses the same states, so only the runs that
  are not done train.
- Bad, because a run that was killed, such as by `scancel` or a time limit,
  looks the same as one that is training. A later experiment whose base run
  has started stops, and says to wait for the run, or to delete its staging
  folder.

**Alternatives considered.**

- A stored table of the runs, a ledger: it can fall out of step with the
  folders, which hold the runs themselves.
- Asking the scheduler for the jobs, as with `squeue`: code for one
  scheduler, which knows nothing of runs on a laptop.

**Confirmation.** The tests in
[test_embedding_model_experiments.py](../../tests/test_embedding_model_experiments.py)
check that:

- a run is done with its manifest, started with a staging folder, and not
  started otherwise;
- `experiment status` counts each arm's runs by state.

### Arms are compared by their seeds' scores, with Welch's interval and t-test

**Described in:** [experiments.md](experiments.md#how-the-arms-are-compared).

**Context.** Each run gives one validation score. An experiment trains each
arm at the same few seeds. A seed sets a run's starting weights and the order
of its pairs. A change such as another table size also changes the starting
weights that a seed gives. Two arms can vary from seed to seed by different
amounts.

**Decision.** An arm's score is the mean of its seeds' scores, with its 95% t
interval. Each arm minus the base gets Welch's interval and Welch's t-test,
unpaired. scipy's `ttest_ind` computes both.

**Consequences.**

- Good, because the interval and the test come from one t distribution, at
  Welch's degrees of freedom. The interval excludes 0 exactly when the test's
  p is below the arm's alpha, so the two cannot disagree.
- Good, because Welch's test does not assume that the two arms vary alike,
  and loses little when they do. So it is the recommended default over
  Student's t-test ([Welch, 1947](https://doi.org/10.1093/biomet/34.1-2.28); [Delacre et al., 2017](https://doi.org/10.5334/irsp.82)).
- Good, because at few runs, a Welch interval over the runs is recommended
  over the bootstrap ([Knopov & Shvai, 2026](https://doi.org/10.48550/arXiv.2610.23700)).
- Bad, because the test assumes that the mean of the seeds' scores is about
  normally distributed. With few seeds, the shape of their spread cannot be
  checked.
- Bad, because a run that trains badly at one seed pulls its arm's mean
  down, and widens its interval.

**Alternatives considered.**

- Student's t-test: it pools the two arms' spreads. When the spreads differ
  and the seeds are few, it finds more false differences than its alpha says
  ([Delacre et al., 2017](https://doi.org/10.5334/irsp.82)).
- A paired t-test, by seed: it fits when a seed gives two arms the same
  start. A change such as another table size gives them other starting
  weights, so matching the runs by seed would carry little.
- A bootstrap interval: at few runs, it understates the spread. Its variance
  is short by the factor (n - 1) / n ([Knopov & Shvai, 2026](https://doi.org/10.48550/arXiv.2610.23700)).
- A permutation test: with 5 seeds per arm, the 10 runs split into two groups
  in only 252 ways. So its p cannot fall below 2 / 252, and it gives no
  interval directly.

**Confirmation.** The tests in
[test_embedding_model_experiment_compare.py](../../tests/test_embedding_model_experiment_compare.py)
check that:

- the interval is Welch's, at Welch-Satterthwaite's degrees of freedom, not
  at Student's pooled ones;
- when neither arm's scores vary, the difference is exact, with p 1 at no
  difference;
- `arms.csv` gives each arm's mean, and each seed's score.

### The superiority test passes on a significant gain of at least the minimum effect

**Described in:** [experiments.md](experiments.md#how-the-arms-are-compared).

**Context.** A `superiority` arm asks whether a change is better than the
base by at least `minimum_effect`. Validation scores can vary little from
seed to seed, so a gain smaller than `minimum_effect` can still be
significant. A test that the gain is larger than `minimum_effect` passes only
on a gain well beyond it.

**Decision.** The superiority test passes when Welch's p is below the arm's
alpha, and the arm's mean gain is at least `minimum_effect`.

**Consequences.**

- Good, because it is a dual-criterion design, made for go or no-go
  decisions. It asks for statistical significance, and for an estimate that
  reaches a decision value ([Roychoudhury et al., 2018](https://doi.org/10.1177/1740774518770661)).
- Good, because a gain too small to be worth a change does not pass the test,
  however sure it is.
- Bad, because an arm whose true gain is exactly `minimum_effect` passes at
  most half the time, and about half once the seeds suffice. Its mean falls
  on either side of it.
  The design expects this, and it does not make the study underpowered
  ([Roychoudhury et al., 2018](https://doi.org/10.1177/1740774518770661)).
- Bad, because the verdict rests on two criteria, not on one test.

**Alternatives considered.**

- Superiority by a margin, with the interval's low end above
  `minimum_effect`: one test. But it passes only a gain beyond
  `minimum_effect` by about two standard errors, so it needs many more seeds.
- A test against no difference alone, the usual superiority trial:
  `minimum_effect` would then only set the seeds needed. Any significant gain
  would pass, however small.

**Confirmation.** The tests in
[test_embedding_model_experiment_compare.py](../../tests/test_embedding_model_experiment_compare.py)
check that:

- a gain of 0.02 passes;
- a significant gain of 0.005 does not;
- a gain of 0.012 that is not significant at the arm's alpha does not;
- the decision's own metric and minimum effect decide, such as accuracy@1
  and 0.02.

### The non-inferiority test passes when the interval stays above minus the minimum effect

**Described in:** [experiments.md](experiments.md#how-the-arms-are-compared).

**Context.** A `non-inferiority` arm, such as a smaller table, is cheaper,
and may lose a little. `minimum_effect` is then the largest loss worth
taking: the margin. A non-inferiority trial sets its margin before the data,
and reads its result from a confidence interval of the difference.

**Decision.** The non-inferiority test passes when the low end of Welch's
interval, at the arm's confidence, is above minus `minimum_effect`.

**Consequences.**

- Good, because it is the rule of CONSORT's extension for non-inferiority
  trials: a two-sided interval against a margin set beforehand ([Piaggio et al., 2012](https://doi.org/10.1001/jama.2012.87802)).
  CONSORT, the Consolidated Standards of Reporting Trials, is the guideline
  that medical journals use for reporting randomized trials.
- Good, because the figure shows it: the interval stays right of the left
  edge of the margin's band.
- Good, because `superiority` reads the same interval, so one interval
  serves both tests.
- Bad, because an arm that passes may still be worse than the base. The test
  bounds the loss; it does not show that there is none.

**Alternatives considered.**

- A one-sided interval at 1 - alpha: it tests at alpha, where the two-sided
  interval's low end tests at alpha / 2. So it passes more arms, and an arm
  worse by exactly the margin passes twice as often.
- An equivalence test, with two one-sided tests at both margins. It also asks
  that the arm be no better by more than the margin, so a cheaper arm that is
  clearly better fails it. A cheaper arm need not show that.

**Confirmation.** The tests in
[test_embedding_model_experiment_compare.py](../../tests/test_embedding_model_experiment_compare.py)
check that:

- a loss of 0.003 passes;
- a loss of 0.009, whose interval reaches past -0.01, does not.

### Alpha is split over the arms, by Bonferroni

**Described in:** [experiments.md](experiments.md#how-the-arms-are-compared).

**Context.** The decision's `alpha`, 0.05 by default, is the chance of a
false finding that the experiment allows. A false finding is an arm that
passes its statistical test, superiority or non-inferiority, by chance. The
more arms an experiment tests against its base, the likelier that one passes
by chance. `experiment compare` reads these tests together, and names each arm
that passes. `compare` reports each test of two models alone.

**Decision.** Each arm is tested at the decision's alpha divided by the
number of arms. Its interval has the matching confidence, such as 97.5% with
2 arms.

**Consequences.**

- Good, because the chance that any arm passes by chance alone is at most
  alpha.
- Good, because each arm's interval and test still agree: both use the same
  alpha.
- Bad, because each added arm costs every arm power. An experiment with more
  arms needs more seeds.

**Alternatives considered.**

- Holm's correction: the same guarantee, and it never finds fewer
  differences. But it has no simple intervals that agree with its tests.
- Dunnett's procedure, made for several arms against one control
  ([Dunnett, 1955](https://doi.org/10.1080/01621459.1955.10501294)): in its standard form, it pools the arms' spreads, as
  Student's t-test does.
- No correction, as `compare` has for models: it suits claims about each
  test alone, not a choice among the arms.

**Confirmation.** The tests in
[test_embedding_model_experiment_compare.py](../../tests/test_embedding_model_experiment_compare.py)
check that:

- with 2 arms, each is tested at 0.025, with a 97.5% interval;
- a gain that passes alone, at p = 0.043, is not shown beside a second arm.

### An arm that does not pass its test is checked for enough seeds, at the minimum effect

**Described in:** [experiments.md](experiments.md#how-the-arms-are-compared).

**Context.** An arm that does not pass its statistical test, superiority or
non-inferiority, says one of two things. The change does not do what the test
asks, or the seeds were too few to tell. The seeds that a test needs follow
from the two arms' seed-to-seed spreads, the arm's alpha and
`minimum_effect`.

**Decision.** Each arm gets the seeds per arm that its test needs. For
superiority, they are the fewest at which a gain of exactly `minimum_effect`
is significant. For non-inferiority, they are the fewest for 80% power when
the arms do not differ. An arm that fails its test, whose interval still
reaches the margin, with fewer seeds, is marked underpowered.

**Consequences.**

- Good, because superiority's count is the sample size of a dual-criterion
  design ([Roychoudhury et al., 2018](https://doi.org/10.1177/1740774518770661)). Non-inferiority's aims at 80% power, the usual convention
  ([Cohen, 1988/2013](https://doi.org/10.4324/9780203771587)).
- Good, because "underpowered: k seeds needed" says how to settle the
  question: a new experiment, at k seeds.
- Bad, because the spreads are estimated from the same few seeds, so the
  count is an estimate too.
- Bad, because scipy has no function for a number of seeds. The count is the
  root of the test's need, found by scipy's `optimize.elementwise`:
  `bracket_root`, then `find_root`.

**Alternatives considered.**

- Power at the observed difference: it only restates the p, so it adds
  nothing ([Hoenig & Heisey, 2001](https://doi.org/10.1198/000313001300339897)).
- 90% power: fewer missed changes, at more seeds. 80% is the usual choice,
  though it has no strong justification ([Lakens, 2022](https://doi.org/10.1525/collabra.33267)).
- 80% power for superiority too: under its two criteria, a gain of exactly
  `minimum_effect` passes at most half the time, at any number of seeds. So
  80% cannot be reached there.
- statsmodels' power for two samples, `TTestIndPower`: it uses only the
  pooled variance, as Student's t-test does, and statsmodels is not a
  dependency.

**Confirmation.** The tests in
[test_embedding_model_experiment_compare.py](../../tests/test_embedding_model_experiment_compare.py)
check that:

- at a seed sd of 0.0045 in both arms, superiority needs 4 seeds and
  non-inferiority 6;
- at a seed sd of 0.02, the count is the first that suffices;
- scores that do not vary need 2 seeds, at any score;
- an arm that does not pass its test, with too few seeds, is marked
  underpowered;
- an arm whose question stays open, with enough seeds, is not;
- an arm whose interval lies wholly past the margin is not, however few its
  seeds.

### The comparison recommends one arm, or none when several pass

**Described in:** [experiments.md](experiments.md#the-recommendation).

**Context.** The next experiment builds on one arm, which its file names as
its base. Several arms can pass their statistical tests, such as a cheaper arm
that is no worse and a better one. Which is worth its cost depends on what
each costs to train and to run. The code does not know these costs.

**Decision.** `experiment compare` recommends the one arm that passes, or the
base when none does. When several arms pass, it recommends none, and prints
each one's base line, for the user to choose. The user writes the next
experiment's base.

**Consequences.**

- Good, because the user, who knows the costs, makes the choice that weighs
  them.
- Bad, because each decision in a series of experiments needs the user.

**Alternatives considered.**

- The passing arm with the best mean: it would pick a larger model for a
  small gain over a cheaper arm that is no worse.
- A file for the next experiment, written by the code: it would hide the
  choice, and the user would still have to check it.

**Confirmation.** The tests in
[test_embedding_model_experiment_compare.py](../../tests/test_embedding_model_experiment_compare.py)
check that:

- one passing arm is recommended, and with none, the base;
- with two passing arms, none is, and both base lines are printed.

## References

> Aarsen, T. (2025, January 15). *Train 400x faster Static Embedding Models
> with Sentence Transformers*. Hugging Face.
> https://huggingface.co/blog/static-embeddings

> Berger, R. L. (1982). Multiparameter Hypothesis Testing and Acceptance
> Sampling. *Technometrics, 24*(4), 295–300.
> https://doi.org/10.1080/00401706.1982.10487790

> Berrendorf, M., Faerman, E., Vermue, L., & Tresp, V. (2020). Interpretable
> and Fair Comparison of Link Prediction or Entity Alignment Methods. *2020
> IEEE/WIC/ACM International Joint Conference on Web Intelligence and
> Intelligent Agent Technology (WI-IAT)*, 371–374.
> https://doi.org/10.1109/wiiat50758.2020.00053

> Bowyer, S., Aitchison, L., & Ivanova, D. R. (2025). Position: Don't Use the
> CLT in LLM Evals With Fewer Than a Few Hundred Datapoints. *Proceedings of
> the 42nd International Conference on Machine Learning*, 81143–81184.
> https://proceedings.mlr.press/v267/bowyer25a.html

> Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval Estimation for a
> Binomial Proportion. *Statistical Science, 16*(2), 101–133.
> https://doi.org/10.1214/ss/1009213286

> Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P.,
> Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S.,
> Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A.,
> Ziegler, D. M., Wu, J., Winter, C., … Amodei, D. (2020). *Language Models
> are Few-Shot Learners* (Version 4). arXiv.
> https://doi.org/10.48550/ARXIV.2005.14165

> Cohen, J. (2013). *Statistical Power Analysis for the Behavioral Sciences*.
> Routledge. https://doi.org/10.4324/9780203771587 (Original work published
> 1988)

> Delacre, M., Lakens, D., & Leys, C. (2017). Why Psychologists Should by
> Default Use Welch's t-test Instead of Student's t-test. *International
> Review of Social Psychology, 30*(1), 92–101. https://doi.org/10.5334/irsp.82

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

> Dunnett, C. W. (1955). A Multiple Comparison Procedure for Comparing
> Several Treatments with a Control. *Journal of the American Statistical
> Association, 50*(272), 1096–1121.
> https://doi.org/10.1080/01621459.1955.10501294

> Efron, B. (1987). Better Bootstrap Confidence Intervals. *Journal of the
> American Statistical Association, 82*(397), 171–185.
> https://doi.org/10.1080/01621459.1987.10478410

> Fagerland, M. W., Lydersen, S., & Laake, P. (2013). The McNemar test for
> binary matched-pairs data: mid-p and asymptotic are better than exact
> conditional. *BMC Medical Research Methodology, 13*(1).
> https://doi.org/10.1186/1471-2288-13-91

> Fagerland, M. W., Lydersen, S., & Laake, P. (2014). Recommended tests and
> confidence intervals for paired binomial proportions. *Statistics in
> Medicine, 33*(16), 2850–2875. https://doi.org/10.1002/sim.6148

> Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and
> TensorFlow* (3rd ed.). O'Reilly Media.

> Ho, J., Tumkaya, T., Aryal, S., Choi, H., & Claridge-Chang, A. (2019).
> Moving beyond P values: data analysis with estimation graphics. *Nature
> Methods, 16*(7), 565–566. https://doi.org/10.1038/s41592-019-0470-3

> Hoenig, J. M., & Heisey, D. M. (2001). The Abuse of Power. *The American
> Statistician, 55*(1), 19–24. https://doi.org/10.1198/000313001300339897

> Ihemelandu, N., & Ekstrand, M. D. (2023). Inference at Scale: Significance
> Testing for Large Search and Recommendation Experiments. *Proceedings of the
> 46th International ACM SIGIR Conference on Research and Development in
> Information Retrieval*, 2087–2091. https://doi.org/10.1145/3539618.3592004

> Knopov, D., & Shvai, N. (2026). *Equivalence testing and run-count design
> for stochastic machine-learning experiments* (Version 1). arXiv.
> https://doi.org/10.48550/ARXIV.2610.23700

> Kopp, O., Armbruster, A., & Zimmermann, O. (2018). Markdown Architectural
> Decision Records: Format and Tool Support. In N. Herzberg, C. Hochreiner, O.
> Kopp, & J. Lenhard (Eds.), *Proceedings of the 10th Central European
> Workshop on Services and their Composition* (pp. 55–62). CEUR-WS.org.
> https://ceur-ws.org/Vol-2072/paper9.pdf

> Lakens, D. (2022). Sample Size Justification. *Collabra: Psychology,
> 8*(1). https://doi.org/10.1525/collabra.33267

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

> Machado, G. M., Oliveira, M. M., & Fernandes, L. (2009). A
> Physiologically-based Model for Simulation of Color Vision Deficiency.
> *IEEE Transactions on Visualization and Computer Graphics, 15*(6),
> 1291–1298. https://doi.org/10.1109/tvcg.2009.113

> McNemar, Q. (1947). Note on the Sampling Error of the Difference Between
> Correlated Proportions or Percentages. *Psychometrika, 12*(2), 153–157.
> https://doi.org/10.1007/bf02295996

> Melnik, S., Gubarev, A., Long, J. J., Romer, G., Shivakumar, S., Tolton,
> M., & Vassilakis, T. (2010). Dremel: Interactive Analysis of Web-Scale
> Datasets. *Proceedings of the VLDB Endowment, 3*(1-2), 330–339.
> https://doi.org/10.14778/1920841.1920886

> Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive
> Text Embedding Benchmark. *Proceedings of the 17th Conference of the
> European Chapter of the Association for Computational Linguistics*,
> 2014–2037. https://doi.org/10.18653/v1/2023.eacl-main.148

> Newcombe, R. G. (1998). Improved confidence intervals for the difference
> between binomial proportions based on paired data. *Statistics in Medicine,
> 17*(22), 2635–2650.
> https://doi.org/10.1002/(SICI)1097-0258(19981130)17:22<2635::AID-SIM954>3.0.CO;2-C

> Nygard, M. (2011, November 15). *Documenting architecture decisions*.
> Cognitect.
> https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions

> OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman,
> F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., Avila, R.,
> Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., Bao, H., Bavarian, M.,
> Belgum, J., … Zoph, B. (2023). *GPT-4 Technical Report* (Version 6). arXiv.
> https://doi.org/10.48550/ARXIV.2303.08774

> Petroff, M. A. (2021). *Accessible Color Sequences for Data Visualization*
> (Version 3). arXiv. https://doi.org/10.48550/ARXIV.2107.02270

> Piaggio, G., Elbourne, D. R., Pocock, S. J., Evans, S. J. W., Altman, D.
> G., & CONSORT Group. (2012). Reporting of Noninferiority and Equivalence
> Randomized Trials. *JAMA, 308*(24), 2594.
> https://doi.org/10.1001/jama.2012.87802

> Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
> using Siamese BERT-Networks. *Proceedings of the 2019 Conference on
> Empirical Methods in Natural Language Processing and the 9th International
> Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3980–3990.
> https://doi.org/10.18653/v1/d19-1410

> Rothman, K. J. (1990). No Adjustments Are Needed for Multiple Comparisons.
> *Epidemiology, 1*(1), 43–46. https://doi.org/10.1097/00001648-199001000-00010

> Roychoudhury, S., Scheuer, N., & Neuenschwander, B. (2018). Beyond
> *p*-values: A phase II dual-criterion design with statistical significance
> and clinical relevance. *Clinical Trials, 15*(5), 452–461.
> https://doi.org/10.1177/1740774518770661

> Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How
> Good is Your Tokenizer? On the Monolingual Performance of Multilingual
> Language Models. *Proceedings of the 59th Annual Meeting of the Association
> for Computational Linguistics and the 11th International Joint Conference on
> Natural Language Processing (Volume 1: Long Papers)*, 3118–3135.
> https://doi.org/10.18653/v1/2021.acl-long.243

> Saltzer, J. H., & Schroeder, M. D. (1975). The protection of information in
> computer systems. *Proceedings of the IEEE, 63*(9), 1278–1308.
> https://doi.org/10.1109/proc.1975.9939

> Santos, A., Wu, E., Lopez, R., Keegan, S., Pena, E., Liu, W., Liu, Y.,
> Fenyo, D., & Freire, J. (2025). *GDC-SM: The GDC Schema Matching Benchmark*
> (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.14963588

> Speer, R. (2019). *ftfy* (Version 5.5.1) [Computer software]. Zenodo.
> https://doi.org/10.5281/ZENODO.2591652

> Student. (1908). The Probable Error of a Mean. *Biometrika, 6*(1), 1–25.
> https://doi.org/10.2307/2331554

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

> Urbano, J. (2026). Stop Using the Wilcoxon Test: Myth, Misconception and
> Misuse in IR Research. *Proceedings of the 49th International ACM SIGIR
> Conference on Research and Development in Information Retrieval*,
> 2758–2768. https://doi.org/10.1145/3805712.3808540

> Urbano, J., Lima, H., & Hanjalic, A. (2019). Statistical Significance
> Testing in Information Retrieval. *Proceedings of the 42nd International
> ACM SIGIR Conference on Research and Development in Information
> Retrieval*, 505–514. https://doi.org/10.1145/3331184.3331259

> Welch, B. L. (1947). The Generalization of 'Student's' Problem when Several
> Different Population Variances are Involved. *Biometrika, 34*(1–2), 28–35.
> https://doi.org/10.1093/biomet/34.1-2.28

> Wilson, E. B. (1927). Probable Inference, the Law of Succession, and
> Statistical Inference. *Journal of the American Statistical Association,
> 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953

> Yang, S., Chiang, W.-L., Zheng, L., Gonzalez, J. E., & Stoica, I. (2023).
> *Rethinking Benchmark and Contamination for Language Models with Rephrased
> Samples* (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2311.04850

> Zeng, X., Hui, Y., Shen, J., Pavlo, A., McKinney, W., & Zhang, H. (2023). An
> Empirical Evaluation of Columnar Storage Formats. *Proceedings of the VLDB
> Endowment, 17*(2), 148–161. https://doi.org/10.14778/3626292.3626298
