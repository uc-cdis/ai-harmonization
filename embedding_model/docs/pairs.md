# Pairs

`pairs prepare` makes the training pairs from the tables that
[`sources fetch`](sources.md) wrote. It reads each source into items: a
concept, a term, a code, a data element or a variable. An item has one
**canonical** text, the name that its source prefers. An item that is not
[open](sources.md#sourcetableparquet) gives no pairs.

A pair holds two texts that mean the same thing. A pair has two sides, and
their names come from
[sentence-transformers](https://sbert.net/docs/sentence_transformer/loss_overview.html#loss-table),
the library that training uses:

- the **anchor**: a text as a study might word a variable, for example a
  short name or a question
- the **positive**: the same meaning, written as a data model writes a
  property, with its name, type, description and values

`pairs prepare` writes both texts in `compact`, the
[text format](scoring.md#text-formats) that scoring uses by default:

```
anchor:   SVG (string)
positive: Saphenous Vein Graft (string): A vessel graft comprised of the saphenous vein.
```

## Running `pairs prepare`

```
python -m embedding_model pairs prepare                        # every source
python -m embedding_model pairs prepare --source nhanes phenx  # some of them
```

- **`--source`** takes the ids in [sources.md](sources.md)'s table.
- **If a source is not fetched**, or its tables changed since its fetch,
  `pairs prepare` prints an error. The error gives the `sources fetch`
  command to run. `pairs prepare` still prepares the other sources. Then it
  exits with status 1.

## What `pairs prepare` writes

`pairs prepare` writes to `embedding_model/inputs/pairs/<source>/`.

```
pairs/<source>/
    pairs.parquet    one row per pair
    manifest.json    the pairs' manifest
```

### `pairs.parquet`

| column | what it holds |
|---|---|
| `anchor`, `positive` | the pair's two texts |
| `source` | the source, for example `ncit` |
| `kind` | the [kind of pair](#the-shape-a-star-around-each-item) |
| `item` | the source's own id of the item on the positive side |
| `term_type` | what the anchor's text is in its source, for example a synonym type or a field such as `short name`. It is `canonical` when the anchor is a canonical, and `description` when the anchor is a description of the item other than its first. |
| `term_group` | `expansion` when the anchor is a short form or a code, for example `BMI`, and `paraphrase` when it is words or a question |
| `status` | `current`, `retired` or `draft`: the status of the item that the anchor's text comes from |
| `study` | the [dbGaP](https://www.ncbi.nlm.nih.gov/gap/) studies that the anchor's text comes from, joined by `\|`. Only PhenX's pairs fill it. |
| `value_list` | the name of the positive's value list, where a source names its lists |
| `anchor_name`, `anchor_type`, `anchor_description`, `anchor_values` | the fields that `pairs prepare` writes the anchor's text from |
| `positive_name`, `positive_type`, `positive_description`, `positive_values` | the fields that `pairs prepare` writes the positive's text from |

### `manifest.json`

The pairs' manifest, with the fields that
[manifests.md](manifests.md#its-fields) lists:

- `files`: the pairs' row count, their file's `sha256` and their table's
  `content_sha256`
- `sources`: the fetch that the pairs come from, with its version and its
  tables' checksums
- `settings`: `format`, the [text format](scoring.md#text-formats) of the
  pairs' texts
- `counts`:
  - `items` and `items_not_open`: the items, and the items that are not open
  - `kind`, `term_group` and `status`: the pairs by each of these columns
  - `duplicates_removed`: the pairs removed as copies of another pair of the
    source, after the [tokenizer](tokenizer.md)'s normalisation
  - `ambiguous_texts`: the anchor names that point to two or more different
    canonicals

## Details

### The shape: a star around each item

Each item is the centre of a star: each pair joins the item's canonical to
one other text of the item. For example, `BMI` and "Quetelet Index" are both
anchors of the canonical "Body Mass Index". They are never paired with each
other. A source can name a replacement for a retired item. The retired
item's canonical and texts then become anchors of the replacement's
canonical.

| kind | anchor | positive |
|---|---|---|
| `name` | another text of the item | the canonical, its type and values |
| `name_description` | another text of the item | the canonical, its type, description and values |
| `description_description` | another description of the item, for example a second definition | the canonical, its type, its first description and values |
| `name_to_description` | the canonical, for an item with no other text | the description alone, with the type and values |

### Each source

| source | canonical (the positive's name) | anchors | description | values |
|---|---|---|---|---|
| [NCIt](sources.md#ncit) | the preferred name | synonyms of the same meaning | the first definition: a `DEFINITION` before an `ALT_DEFINITION`, and NCI's before another source's | each registry's list |
| [Ontologies](sources.md#ontologies) | the label | EXACT synonyms | the definition | none |
| [LOINC](sources.md#loinc) | the long common name | short and display names, question texts | the definition | the default answer list |
| [CDISC](sources.md#cdisc) | a term's NCI Preferred Term, and a codelist's Codelist Name | submission values, synonyms, question texts | the first definition | a codelist's terms |
| [NIH CDE](sources.md#nih-cde) | the first designation that has only name or question tags, or no tag | the other designations of that kind | the first definition | the value meanings, or else the permissible values |
| [caDSR](sources.md#cadsr) | the long name | the short name, alternate names, question texts | the definition | the value meanings, or else the values |
| [NHANES](sources.md#nhanes) | the codebook's English Text, or else the SAS label, or else the variable list's description | the variable's name, its SAS label and its variable list description | none | the value labels |
| [PhenX](sources.md#phenx) | the variable's description | the variable's name, and mapped dbGaP variables | none | the value labels |
