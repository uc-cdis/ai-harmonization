"""
NCI Thesaurus → REDCap-style noisy training pairs
for bi-encoder fine-tuning with HuggingFace sentence-transformers.

Quick wins:
  1. Load NCI thesaurus concepts (owl/csv)
  2. Generate noisy REDCap field paragraph (query)
  3. Generate clean CDE paragraph (target positive)
  4. Mine hard negatives from same semantic domain
  5. Output HuggingFace-ready triplet dataset

Install:
    pip install sentence-transformers datasets owlready2 pandas tqdm
"""

import re
import random
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD NCI THESAURUS
#     Download flat CSV from:
#     https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Thesaurus.FLAT.zip
#     Columns we care about:
#       code, label (preferred name), synonyms (pipe-sep), definition,
#       parents, children, semantic_type
# ─────────────────────────────────────────────────────────────────────────────

def load_nci_flat(path: str) -> pd.DataFrame:
    """
    Load NCI Thesaurus FLAT file.
    Expects tab-separated with header row.
    Minimal required columns: code, label, synonyms, definition
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Normalize column names — the flat file uses mixed case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Robust column mapping for current NCI Thesaurus.txt.
    # Current file columns include:
    # code, conceptIRI, parents, synonyms, definition, displayname,
    # conceptStatus, semantic_type, concept_in_subset

    if "definition" not in df.columns and "def" in df.columns:
        df["definition"] = df["def"]

    if "synonyms" not in df.columns:
        df["synonyms"] = ""

    # Keep only concepts that have a useful definition.
    df["def"] = df["definition"].fillna("").astype(str)
    df = df[df["def"].str.len() > 20].copy()

    # synonyms column is pipe-separated in the flat file.
    df["synonym_list"] = df["synonyms"].fillna("").astype(str).apply(
        lambda s: [
            x.strip()
            for x in s.split("|")
            if x.strip() and x.strip().lower() != "nan"
        ]
    )

    # Current Thesaurus.txt usually does not have a label column.
    # Use displayname when available; otherwise first synonym; otherwise code.
    def _pick_label(row):
        for col in ["label", "displayname", "preferred_name", "concept_name", "name"]:
            if col in row.index:
                v = row.get(col)
                if pd.notna(v) and str(v).strip() and str(v).strip().lower() != "nan":
                    return str(v).strip()
        syns = row.get("synonym_list", [])
        if isinstance(syns, list) and len(syns) > 0:
            return syns[0]
        return str(row.get("code", "unknown"))

    df["label"] = df.apply(_pick_label, axis=1)

    print(f"Loaded {len(df):,} NCI concepts with definitions")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  REDCAP NOISE TRANSFORMS
#     Each function takes a clean NCI concept and returns a realistic
#     REDCap field row as a dict, then we stringify it to a paragraph.
# ─────────────────────────────────────────────────────────────────────────────

INSTRUMENT_NAMES = [
    "Demographics", "Contact Information", "Medical History",
    "Baseline Assessment", "Follow-up Visit", "Screening",
    "Eligibility Criteria", "Adverse Events", "Vital Signs",
    "Laboratory Results", "Physical Examination", "Enrollment",
]

INSTRUMENT_ABBREVS = [
    "dem", "med", "lab", "sx", "hx", "bl", "fu", "ae", "vs", "scr",
]

REDCAP_BOOL_TYPES = ["boolean", "checkbox"]
REDCAP_ENUM_TYPES = ["radio", "dropdown", "checkbox"]
REDCAP_FREE_TYPES = ["text", "notes"]


# ── variable name generation ──────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 20) -> str:
    """Convert a concept label to a REDCap-style variable name."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")[:max_len]
    return s.rstrip("_")


def generate_variable_name(label: str, field_type: str,
                            choice_index: Optional[int] = None) -> str:
    slug = _slugify(label)
    style = random.random()

    if field_type in REDCAP_BOOL_TYPES and choice_index is not None:
        # checkbox pattern: concept___N  (the key failure mode in your eval set)
        return f"{slug}___{choice_index}"

    if style < 0.3:
        # instrument prefix: dem_race, med_diagnosis
        prefix = random.choice(INSTRUMENT_ABBREVS)
        return f"{prefix}_{slug}"[:26]
    elif style < 0.6:
        # abbreviated slug only
        words = slug.split("_")
        abbrev = "_".join(w[:4] for w in words)[:26]
        return abbrev
    else:
        return slug[:26]


# ── enum format perturbation ──────────────────────────────────────────────────

def _format_enum(pairs: list[tuple[str, str]]) -> str:
    """Serialize (code, label) pairs in one of several REDCap-observed formats."""
    fmt = random.choice(["python_dict", "json", "pipe", "semicolon", "bare"])
    sep = random.choice([" | ", "; ", "\n", ", "])

    if fmt == "python_dict":
        return "{" + ", ".join(f"'{k}': '{v}'" for k, v in pairs) + "}"
    elif fmt == "json":
        return "{" + ", ".join(f'"{k}": "{v}"' for k, v in pairs) + "}"
    elif fmt == "pipe":
        return " | ".join(f"{k}, {v}" for k, v in pairs)
    elif fmt == "semicolon":
        return "; ".join(f'"{k}": "{v}"' for k, v in pairs)
    else:  # bare
        return sep.join(f"{k}={v}" for k, v in pairs)


def make_boolean_enums() -> tuple[list[tuple], str]:
    """Generate checkbox-style 0/1 enum in random format."""
    unchecked = random.choice(["Unchecked", "No", "False", "0", "not checked"])
    checked   = random.choice(["Checked",   "Yes", "True", "1", "checked"])
    pairs = [("0", unchecked), ("1", checked)]
    return pairs, _format_enum(pairs)


def make_ordinal_enums(n: int = 5) -> tuple[list[tuple], str]:
    """Generate a simple ordinal scale."""
    labels = random.choice([
        ["Never", "Rarely", "Sometimes", "Often", "Always"],
        ["None", "Mild", "Moderate", "Severe", "Extreme"],
        ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
    ])[:n]
    pairs = [(str(i), l) for i, l in enumerate(labels)]
    return pairs, _format_enum(pairs)


# ── description perturbation ──────────────────────────────────────────────────

def perturb_description(definition: str, instrument: str, label: str) -> str:
    """Return a noisy version of the CDE definition as a REDCap field description."""
    strategy = random.random()

    if strategy < 0.15:
        return ""  # blank — very common in real instruments

    if strategy < 0.30:
        return definition.split(".")[0]  # first sentence only

    if strategy < 0.45:
        words = definition.split()
        if len(words) <= 10:
            return definition  # too short to truncate, return as-is
        trunc = random.randint(10, min(20, len(words)))
        return " ".join(words[:trunc])  # truncated

    if strategy < 0.60:
        # instrument path prefix  e.g. "Contact Information: Race[choice=American Indian]"
        choice_hint = f"[choice={label}]" if random.random() < 0.4 else ""
        return f"{instrument}: {label}{choice_hint}"

    if strategy < 0.75:
        # full definition with instrument prefix
        return f"{instrument} > {definition}"

    return definition  # full definition (less common in practice)


# ── field type assignment ─────────────────────────────────────────────────────

def assign_field_type(nci_type: str, has_enums: bool) -> str:
    """
    Map NCI semantic type to a plausible REDCap field type.
    NCI semantic types relevant to CDEs: Conceptual Entity, Finding,
    Laboratory Procedure, Qualitative Concept, Quantitative Concept, etc.
    """
    if has_enums:
        return random.choice(["radio", "dropdown", "checkbox"])

    quant_types = {"Quantitative Concept", "Laboratory Procedure",
                   "Diagnostic Procedure"}
    if any(t in nci_type for t in quant_types):
        return random.choice(["text", "integer", "float"])

    return random.choice(["text", "notes", "radio"])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PARAGRAPH CONSTRUCTORS
# ─────────────────────────────────────────────────────────────────────────────

def build_redcap_paragraph(row: dict) -> str:
    """
    Construct a noisy REDCap-style query paragraph from a field dict.
    This mirrors exactly what you're embedding at inference time.
    """
    parts = []

    if row.get("field_name"):
        parts.append(f"field_name: {row['field_name']}")
    if row.get("field_type"):
        parts.append(f"field_type: {row['field_type']}")
    if row.get("field_title"):
        parts.append(f"field_title: {row['field_title']}")
    if row.get("field_enumLabels"):
        parts.append(f"field_enumLabels: {row['field_enumLabels']}")
    if row.get("field_description"):
        parts.append(f"field_description: {row['field_description']}")
    if row.get("instrument"):
        parts.append(f"instrument: {row['instrument']}")

    return " | ".join(parts)


def build_cde_paragraph(concept: pd.Series) -> str:
    """
    Construct a clean CDE target paragraph from an NCI concept.
    This mirrors your existing target embedding format.
    """
    parts = []

    label = str(concept.get("label", ""))
    definition = str(concept.get("def", ""))
    synonyms = concept.get("synonym_list", [])
    code = str(concept.get("code", ""))
    sem_type = str(concept.get("semantic_type", ""))

    if label:
        parts.append(f"element_name: {label}")
    if sem_type:
        parts.append(f"element_type: {sem_type}")
    if label:
        parts.append(f"element_title: {label}")
    if definition:
        parts.append(f"element_description: {definition}")
    if synonyms:
        syn_str = "; ".join(synonyms[:5])  # cap at 5 synonyms
        parts.append(f"synonyms: {syn_str}")
    if code:
        parts.append(f"nci_code: {code}")

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GENERATE ONE TRAINING EXAMPLE FROM ONE NCI CONCEPT
# ─────────────────────────────────────────────────────────────────────────────

def concept_to_training_example(concept: pd.Series) -> list[dict]:
    """
    From one NCI concept, generate 1–3 noisy REDCap query variants
    all paired with the same clean CDE target paragraph.
    Returns a list of (query, positive) dicts.
    """
    label    = str(concept.get("label", "unknown"))
    defn     = str(concept.get("def", ""))
    sem_type = str(concept.get("semantic_type", ""))
    synonyms = concept.get("synonym_list", [])
    instrument = random.choice(INSTRUMENT_NAMES)

    examples = []
    n_variants = random.choice([1, 1, 2, 2, 3])  # weighted toward 1-2

    for variant_idx in range(n_variants):
        # Decide if this concept maps to an enumerated field
        is_enum     = random.random() < 0.35
        is_boolean  = random.random() < 0.25 and not is_enum
        choice_idx  = random.randint(1, 5) if is_boolean else None

        if is_boolean:
            field_type = random.choice(["checkbox", "checkbox", "boolean"])
        else:
            field_type = assign_field_type(sem_type, is_enum)

        # Variable name — use label or a synonym as the base
        name_base = random.choice([label] + synonyms[:2]) if synonyms else label
        var_name  = generate_variable_name(name_base, field_type, choice_idx)

        # Enum labels
        if is_boolean:
            _, enum_str = make_boolean_enums()
        elif is_enum:
            _, enum_str = make_ordinal_enums(random.randint(3, 6))
        else:
            enum_str = ""

        # Title — sometimes use a synonym, sometimes truncate, sometimes add path
        title_options = [label] + synonyms[:3]
        raw_title = random.choice(title_options)
        if random.random() < 0.3:
            raw_title = f"{instrument}: {raw_title}"
        if is_boolean and choice_idx is not None:
            raw_title = f"{instrument}: {label}[choice={label}]"

        # Description
        description = perturb_description(defn, instrument, label)

        field_row = {
            "field_name":        var_name,
            "field_type":        field_type,
            "field_title":       raw_title,
            "field_enumLabels":  enum_str,
            "field_description": description,
            "instrument":        instrument if random.random() < 0.6 else "",
        }

        query_paragraph    = build_redcap_paragraph(field_row)
        positive_paragraph = build_cde_paragraph(concept)

        examples.append({
            "query":    query_paragraph,
            "positive": positive_paragraph,
            "nci_code": str(concept.get("code", "")),
            "label":    label,
        })

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HARD NEGATIVE MINING
#     Quick win: use concepts from the SAME semantic_type as negatives.
#     They're topically related but conceptually distinct — exactly the
#     hard negative profile you need.
# ─────────────────────────────────────────────────────────────────────────────

def add_hard_negatives(examples: list[dict], df: pd.DataFrame,
                       n_negatives: int = 2) -> list[dict]:
    """
    For each example, attach hard negatives sampled from the same
    semantic_type bucket. This is fast (no model needed) and produces
    plausible distractors.

    For embedding-based hard negatives (better but slower), swap in
    a sentence-transformers model after you have a first trained checkpoint.
    """
    # Build index: semantic_type -> list of CDE paragraphs
    type_index: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        st = str(row.get("semantic_type", "unknown"))
        para = build_cde_paragraph(row)
        type_index.setdefault(st, []).append(para)

    result = []
    for ex in examples:
        # Find semantic type for this example's concept
        matching = df[df["code"] == ex["nci_code"]]
        if matching.empty:
            result.append(ex)
            continue

        sem_type = str(matching.iloc[0].get("semantic_type", "unknown"))
        pool = [p for p in type_index.get(sem_type, []) if p != ex["positive"]]

        if len(pool) < n_negatives:
            # Fall back to random negatives
            all_positives = [ex["positive"] for ex in examples]
            pool = [p for p in all_positives if p != ex["positive"]]

        negatives = random.sample(pool, min(n_negatives, len(pool)))
        ex["negatives"] = negatives
        result.append(ex)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BUILD HUGGINGFACE DATASET
#     sentence-transformers v3 expects InputExample or datasets.Dataset
#     with columns: anchor, positive, negative  (for TripletLoss)
#     or: anchor, positive  (for MultipleNegativesRankingLoss — preferred)
# ─────────────────────────────────────────────────────────────────────────────

def build_hf_dataset(examples: list[dict],
                     loss_type: str = "mnrl") -> Dataset:
    """
    loss_type:
        'mnrl'    → MultipleNegativesRankingLoss (recommended)
                    columns: anchor, positive
                    in-batch negatives handle the rest — no explicit negatives needed

        'triplet' → TripletLoss
                    columns: anchor, positive, negative
                    requires explicit negatives per example
    """
    if loss_type == "mnrl":
        records = [{"anchor": ex["query"], "positive": ex["positive"]}
                   for ex in examples]

    elif loss_type == "triplet":
        records = []
        for ex in examples:
            for neg in ex.get("negatives", []):
                records.append({
                    "anchor":   ex["query"],
                    "positive": ex["positive"],
                    "negative": neg,
                })
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING SNIPPET
#     Shows how to plug the dataset straight into sentence-transformers v3
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_SNIPPET = '''
# ── pip install sentence-transformers>=3.0 datasets ──────────────────────────

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# 1. Load your existing model (the one you already fine-tuned on NCI thesaurus)
model = SentenceTransformer("your-existing-model-checkpoint")

# 2. Load the generated dataset
from datasets import load_from_disk
dataset = load_from_disk("nci_redcap_pairs_mnrl")

# 3. Loss — MNRL is the best default for retrieval fine-tuning
#    It treats all other (anchor, positive) pairs in the batch as negatives.
#    Effective batch size matters: use 64-128 for good in-batch negatives.
loss = MultipleNegativesRankingLoss(model)

# 4. Training args
args = SentenceTransformerTrainingArguments(
    output_dir="nci-redcap-biencoder-v2",
    num_train_epochs=3,
    per_device_train_batch_size=64,   # larger = more in-batch negatives
    warmup_ratio=0.1,
    learning_rate=2e-5,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
)

# 5. Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=loss,
)
trainer.train()

# 6. Save
model.save_pretrained("nci-redcap-biencoder-v2/final")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN — put it all together
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    nci_flat_path: str,
    output_dir: str = "nci_redcap_pairs",
    max_concepts: Optional[int] = None,
    loss_type: str = "mnrl",
    add_negatives: bool = True,
) -> Dataset:
    """
    Full pipeline: NCI flat file → HuggingFace dataset ready for training.

    Args:
        nci_flat_path:  Path to Thesaurus.FLAT.txt (tab-separated)
        output_dir:     Where to save the HuggingFace dataset
        max_concepts:   Cap for quick testing (None = use all)
        loss_type:      'mnrl' or 'triplet'
        add_negatives:  Whether to mine hard negatives (required for triplet)
    """
    print("Step 1: Loading NCI Thesaurus...")
    df = load_nci_flat(nci_flat_path)

    if max_concepts:
        df = df.sample(min(max_concepts, len(df)), random_state=42)
        print(f"  Sampled {len(df):,} concepts for generation")

    print("Step 2: Generating noisy REDCap query paragraphs...")
    all_examples = []
    for _, concept in tqdm(df.iterrows(), total=len(df)):
        all_examples.extend(concept_to_training_example(concept))

    print(f"  Generated {len(all_examples):,} (query, positive) pairs")

    if add_negatives:
        print("Step 3: Mining hard negatives...")
        all_examples = add_hard_negatives(all_examples, df, n_negatives=2)

    print("Step 4: Building HuggingFace dataset...")
    dataset = build_hf_dataset(all_examples, loss_type=loss_type)
    print(f"  Dataset size: {len(dataset):,} rows")
    print(f"  Columns: {dataset.column_names}")

    out_path = f"{output_dir}_{loss_type}"
    dataset.save_to_disk(out_path)
    print(f"  Saved to: {out_path}/")

    # Print a few examples so you can sanity-check the noise patterns
    print("\n── Sample query paragraphs ──────────────────────────────────────")
    for i in range(min(3, len(dataset))):
        print(f"\n[{i}] QUERY:\n  {dataset[i]['anchor'][:200]}...")
        print(f"    POSITIVE:\n  {dataset[i]['positive'][:200]}...")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO — runs without the real NCI file, using toy data
# ─────────────────────────────────────────────────────────────────────────────

def demo_without_nci_file():
    """
    Demonstrates the paragraph generation on a few hardcoded NCI-style
    concepts so you can see the noise patterns without downloading the
    full thesaurus.
    """
    toy_concepts = [
        {
            "code": "C17998",
            "label": "Race",
            "def": "A social construct that divides people into groups based on physical traits, ancestry, genetics, or social relations.",
            "synonyms": "Racial Group|Ethnic Race",
            "semantic_type": "Conceptual Entity",
        },
        {
            "code": "C43390",
            "label": "American Indian or Alaska Native",
            "def": "A person having origins in any of the original peoples of North, Central, and South America, including Navajo Nation, Blackfeet Tribe, and Nome Eskimo Community.",
            "synonyms": "AI/AN|Native American|Indigenous American",
            "semantic_type": "Population Group",
        },
        {
            "code": "C25301",
            "label": "Age",
            "def": "The length of time that a person has lived or a thing has existed.",
            "synonyms": "Age in Years|Subject Age",
            "semantic_type": "Quantitative Concept",
        },
    ]

    df = pd.DataFrame(toy_concepts)
    df["synonym_list"] = df["synonyms"].apply(
        lambda s: [x.strip() for x in s.split("|")]
    )

    print("=" * 70)
    print("DEMO: REDCap noise generation from NCI concepts")
    print("=" * 70)

    for _, concept in df.iterrows():
        examples = concept_to_training_example(concept)
        cde_para = build_cde_paragraph(concept)
        print(f"\n── NCI Concept: {concept['label']} ({concept['code']}) ──")
        print(f"  CDE target paragraph:\n    {cde_para[:180]}")
        for i, ex in enumerate(examples):
            print(f"\n  Noisy REDCap variant {i+1}:\n    {ex['query']}")

    print("\n── Training snippet ─────────────────────────────────────────────")
    print(TRAINING_SNIPPET)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Real run: python nci_to_redcap_pairs.py /path/to/Thesaurus.FLAT.txt
        nci_path = sys.argv[1]
        max_c = int(sys.argv[2]) if len(sys.argv) > 2 else None
        generate_dataset(
            nci_flat_path=nci_path,
            max_concepts=max_c,
            loss_type="mnrl",
        )
    else:
        # Demo run without NCI file
        demo_without_nci_file()
