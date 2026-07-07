"""
data_prep.py
==============================
training-data preparation for the HEAL CDE->SDE bi-encoder.

Two inputs:
  1. Thesaurus.txt                        -> synthetic REDCap pairs 
  2. thesaurus_finetuning_data_nodup.csv  -> clean NCIt pairs 

"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
BASE_DIR = Path("/opt/gpudata/fan1/heal_cde/ncit")
SCRIPT_DIR = BASE_DIR / "scripts"

# ---- the two inputs ----
THESAURUS_TXT = SCRIPT_DIR / "Thesaurus.txt"                       # input 1 (synthetic source)
FIXED_SCRIPT = SCRIPT_DIR / "nci_to_redcap_pairs_fixed.py"         # generator for input-1
RAW_NCIT_CSV = BASE_DIR / "thesaurus_finetuning_data_nodup.csv"    # input 2 (clean, frozen)

# ---- output (fresh path; does not overwrite originals) ----
OUT_ROOT = BASE_DIR / "experiments" / "data_prep_v2"
SYN_DIR = OUT_ROOT / "synthetic"
OUT_DIR = OUT_ROOT / "aug_data"

MAX_CONCEPTS = 50000
REDCAP_SAMPLE_SIZES = [5000, 20000, 50000]   # Stage B uses the 5000 file
MIN_SYNTHETIC_TARGET_CHARS = 20
SEED = 42
REGENERATE_SYNTHETIC = True   # first run: True. Later runs: False -> reuse existing synthetic CSV.

BAD_TARGETS = {"", "nan", "none", "null", "yes", "no", "unknown", "other",
               "1", "2", "3", "4", "5", "not applicable", "not reported",
               "missing", "na", "n/a"}
MAX_QUERY_CHARS = 5000
MAX_TARGET_CHARS = 5000
MIN_TARGET_CHARS = 2


def normalize_text(text) -> str:
    s = "" if pd.isna(text) else str(text)
    return re.sub(r"\s+", " ", s.strip().lower())


def safe_text(text) -> str:
    return "" if pd.isna(text) else str(text).strip()


# ======================================================================================
# A. Regenerate synthetic REDCap pairs from Thesaurus.txt
# ======================================================================================
def generate_synthetic(thesaurus_txt: Path, fixed_script: Path, syn_dir: Path,
                       max_concepts: int = MAX_CONCEPTS) -> Path:
    """
    exec the _fixed script as a module (its top-level random.seed(42) gives a clean,
    deterministic RNG state) and call generate_dataset exactly as the original notebook did,
    then convert the HF dataset -> the *_variable_para_{N}.csv the augment step expects.
    """
    from datasets import load_from_disk

    syn_dir.mkdir(parents=True, exist_ok=True)
    out_csv = syn_dir / f"nci_redcap_pairs_mnrl_variable_para_{max_concepts}.csv"
    if not REGENERATE_SYNTHETIC and out_csv.exists():
        print("Reusing existing synthetic:", out_csv)
        return out_csv

    assert thesaurus_txt.exists(), f"missing input: {thesaurus_txt}"
    assert fixed_script.exists(), f"missing generator: {fixed_script}"

    spec = importlib.util.spec_from_file_location("nci_to_redcap_pairs_fixed", str(fixed_script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # runs random.seed(42) at module top level

    mod.generate_dataset(
        nci_flat_path=str(thesaurus_txt),
        output_dir=str(syn_dir / "nci_redcap_pairs"),
        max_concepts=max_concepts,
        loss_type="mnrl",
        add_negatives=False,           # MNRL: anchor/positive only
    )
    ds = load_from_disk(str(syn_dir / "nci_redcap_pairs_mnrl"))
    df = ds.to_pandas().rename(columns={"anchor": "variable_para",
                                        "positive": "alternate_variable_para"})
    df.to_csv(out_csv, index=False)
    print("Saved synthetic:", out_csv, df.shape)
    return out_csv


# ======================================================================================
# B. Clean the raw NCIt pairs
# ======================================================================================
def load_and_clean_ncit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"variable_para", "alternate_variable_para", "semantic_type"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    out = df[["variable_para", "alternate_variable_para", "semantic_type"]].copy()
    for c in ("variable_para", "alternate_variable_para", "semantic_type"):
        out[c] = out[c].map(safe_text)
    out["query_norm"] = out["variable_para"].map(normalize_text)
    out["target_norm"] = out["alternate_variable_para"].map(normalize_text)
    out["semantic_norm"] = out["semantic_type"].map(normalize_text)
    out["query_len"] = out["variable_para"].str.len()
    out["target_len"] = out["alternate_variable_para"].str.len()

    bad = (out["query_norm"].eq("") | out["target_norm"].isin(BAD_TARGETS)
           | out["semantic_norm"].eq("") | (out["target_len"] < MIN_TARGET_CHARS)
           | (out["query_len"] > MAX_QUERY_CHARS) | (out["target_len"] > MAX_TARGET_CHARS))
    cleaned = out.loc[~bad].reset_index(drop=True)
    cleaned["row_id"] = np.arange(len(cleaned))
    print(f"cleaned NCIt: {len(cleaned)} rows (unique query_norm={cleaned['query_norm'].nunique()})")
    return cleaned


# ======================================================================================
# C. Internal seed-42 split 
# ======================================================================================
def make_split(cleaned: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    sklearn train_test_split over SORTED unique query_norm, two cuts (80/20 -> 10/10),
    random_state=seed, shuffle=True. Verified to reproduce splits_...seed42.csv 100%.
    """
    unique_queries = pd.DataFrame({"query_norm": sorted(cleaned["query_norm"].unique())})
    train_q, temp_q = train_test_split(unique_queries, test_size=0.20, random_state=seed, shuffle=True)
    val_q, test_q = train_test_split(temp_q, test_size=0.50, random_state=seed, shuffle=True)
    split_df = pd.concat([
        train_q.assign(split="train"),
        val_q.assign(split="val"),
        test_q.assign(split="test"),
    ], ignore_index=True)
    print("split:", split_df["split"].value_counts().to_dict())
    return split_df


def reconstruct_clean_split(cleaned: pd.DataFrame, split_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    full = cleaned.merge(split_df[["query_norm", "split"]].drop_duplicates(), on="query_norm", how="left")
    if full["split"].isna().any():
        raise ValueError("split does not cover all query_norm values.")
    out_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_dir / "cleaned_multipos_full_split_reconstructed_exact_notebook.csv", index=False)
    train_only = full[full["split"].eq("train")][["variable_para", "alternate_variable_para"]]
    train_only.to_csv(out_dir / "cleaned_multipos_train_only_reconstructed_exact_notebook.csv", index=False)
    print("clean full/train:", full.shape, train_only.shape)
    return full


# ======================================================================================
# D. Merge clean + synthetic -> augmented (with source_type tag)
# ======================================================================================
def build_augmented_redcap(full_split: pd.DataFrame, synthetic_csv: Path, out_dir: Path,
                           sizes=REDCAP_SAMPLE_SIZES, seed: int = SEED) -> None:
    needed = ["variable_para", "alternate_variable_para"]
    syn = pd.read_csv(synthetic_csv)[needed].dropna().drop_duplicates()
    syn = syn[~syn["variable_para"].astype(str).str.lower().str.contains("unknown", na=False)]
    syn = syn[syn["alternate_variable_para"].astype(str).str.len() > MIN_SYNTHETIC_TARGET_CHARS].copy()
    print("synthetic usable:", syn.shape)

    for col in full_split.columns:
        if col not in syn.columns:
            syn[col] = None
    syn["split"] = "train"
    syn["query_norm"] = syn["variable_para"].map(normalize_text)
    syn["target_norm"] = syn["alternate_variable_para"].map(normalize_text)
    syn["semantic_type"] = "Synthetic REDCap"
    syn["semantic_norm"] = "synthetic redcap"
    syn["query_len"] = syn["variable_para"].astype(str).str.len()
    syn["target_len"] = syn["alternate_variable_para"].astype(str).str.len()
    syn["row_id"] = None
    syn = syn[full_split.columns]

    out_dir.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        sample = syn.sample(min(n, len(syn)), random_state=seed)
        merged = (pd.concat([full_split, sample], ignore_index=True)
                    .drop_duplicates(needed).reset_index(drop=True))
        merged["source_type"] = np.where(merged["semantic_norm"].eq("synthetic redcap"),
                                         "redcap_synthetic", "clean")
        merged.to_csv(out_dir / f"cleaned_multipos_full_split_plus_nci_redcap_{n}_fixed.csv", index=False)
        train_only = (merged[merged["split"].eq("train")][needed + ["source_type"]]
                      .dropna(subset=needed).drop_duplicates(needed))
        train_only.to_csv(out_dir / f"cleaned_multipos_train_only_plus_nci_redcap_{n}_fixed.csv", index=False)
        print(f"n={n}: full={merged.shape} train={train_only.shape}")


def main() -> None:
    syn_csv = generate_synthetic(THESAURUS_TXT, FIXED_SCRIPT, SYN_DIR)
    cleaned = load_and_clean_ncit(RAW_NCIT_CSV)
    split_df = make_split(cleaned)
    full_split = reconstruct_clean_split(cleaned, split_df, OUT_DIR)
    build_augmented_redcap(full_split, syn_csv, OUT_DIR)
    print("\nDATA PREP DONE ->", OUT_ROOT)


if __name__ == "__main__":
    main()
