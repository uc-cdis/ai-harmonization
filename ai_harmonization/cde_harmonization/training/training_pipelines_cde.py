"""
training_pipelines.py
=====================
Two-stage bi-encoder training for HEAL CDE->SDE mapping.

  Stage A : ground CTDS-base with LoRA on clean NCIt pairs (in-batch InfoNCE).
  Stage B : continue from the Stage A checkpoint on clean NCIt + filtered REDCap
            at a 15:1 oversampled ratio, with anti-drift MSE distillation to the
            frozen Stage-A reference. Saves LoRA checkpoints at epoch fractions.

Training DATA is produced by `data_prep_cde.py` and read as:
    CLEAN_FULL_SPLIT_PATH             = cleaned_multipos_full_split_reconstructed_exact_notebook.csv
    AUGMENTED_TRAIN_PLUS_REDCAP_PATH  = cleaned_multipos_train_only_plus_nci_redcap_5000_fixed.csv
"""

from __future__ import annotations
import gc
import hashlib
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

# ======================================================================================
# CONFIG
# ======================================================================================
BASE_MODEL_NAME = "uc-ctds/bge-large-en-v1.5-bio-mapping"
MODEL_NAME = BASE_MODEL_NAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SANDBOX = Path("/opt/gpudata/fan1/heal_cde/ncit/experiments/sandbox_myrun")
# Original run root — READ-ONLY, only used to reuse the existing Stage A checkpoint.
ORIG_REPRO_ROOT = Path("/opt/gpudata/fan1/heal_cde/ncit/experiments/redcap_repro_from_scratch")
# All NEW writes go under the sandbox.
REPRO_ROOT = SANDBOX / "repro"

# ---- data produced by data_prep_cde.py -------------------------------------------------
DATA_ROOT = Path("/opt/gpudata/fan1/heal_cde/ncit/experiments/data_prep_v2")   # == data_prep.OUT_ROOT
AUG_ROOT = DATA_ROOT / "aug_data"   # == data_prep.OUT_DIR
CLEAN_FULL_SPLIT_PATH = AUG_ROOT / "cleaned_multipos_full_split_reconstructed_exact_notebook.csv"
AUGMENTED_TRAIN_PLUS_REDCAP_PATH = AUG_ROOT / "cleaned_multipos_train_only_plus_nci_redcap_5000_fixed.csv"
RAW_NCIT_CSV = Path("/opt/gpudata/fan1/heal_cde/ncit/thesaurus_finetuning_data_nodup.csv")

# ------------------------------------------------------------------------------------
# Stage A / Stage B hyperparameters — paired for at-a-glance comparison
#   stage               A (grounding)      B (REDCap adapt)
#   learning rate       5e-6               5e-7        (B is 10x lower: gentle adapt)
#   batch size          64                 64
#   epochs              3 (full)           0.5 (fraction, via CHECKPOINT_FRACTIONS)
#   temperature         0.02               0.05
#   weight decay        0.01               0.01
#   grad accum steps    1                  1
#   max grad norm       None (no clip)     1.0 (clip)
#   seed                42                 42
#   loss                InfoNCE            symmetric InfoNCE + anti-drift MSE
# ------------------------------------------------------------------------------------
# ---- Stage A hyperparameters -------------------------------------------------------
BATCH_SIZE_A = 64         
LEARNING_RATE_A = 5e-6     
EPOCHS_A = 3              
TEMPERATURE_A = 0.02
WEIGHT_DECAY_A = 0.01    
GRAD_ACCUM_STEPS_A = 1    
MAX_GRAD_NORM_A = None    
MAX_LEN = 512
NUM_WORKERS = 0
VAL_EVERY_STEPS = 1000
CKPT_EVERY_STEPS = 3000
SEED_A = 42              
STAGE_A_DIR = REPRO_ROOT / "stageA_2stage_cleaned_multipos"

# ---- Stage B hyperparameters -------------------------------------------------------
# Reuse the ORIGINAL Stage A checkpoint (read-only).
INIT_CHECKPOINT_PATH = ORIG_REPRO_ROOT / "stageA_2stage_cleaned_multipos" / "biencoder" / "biencoder_lora_final.pt"
CLEAN_TO_REDCAP_RATIO = 15
REDCAP_OVERSAMPLE_WITH_REPLACEMENT = True
TARGETED_REDCAP_MIN_FRACTION = 0.60
CHECKPOINT_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]
EPOCHS_B = max(CHECKPOINT_FRACTIONS)   # was MAX_EPOCH_FRACTION; 0.5 = Stage B trains <=1/2 epoch
ANTI_DRIFT_WEIGHTS = [0.10]
LEARNING_RATE_B = 5e-7    
TEMPERATURE_B = 0.05
BATCH_SIZE_B = 64          
GRAD_ACCUM_STEPS_B = 1    
MAX_GRAD_NORM_B = 1.0     
WEIGHT_DECAY_B = 0.01     
SEED_B = 42            
STAGE_B_DIR = REPRO_ROOT / "stageB_redcap_15to1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNG. torch.manual_seed also seeds CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Module-level baseline seed (Stage A seed). Each stage re-seeds with its own SEED_A/SEED_B
# at the start of its training function, so the two stages are independently reproducible.
_seed_everything(SEED_A)


# ======================================================================================
# Shared text builders (fast-path: the CSV already stores compact query/target text)
# ======================================================================================
def clean_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def build_query_text(row: pd.Series) -> str:
    return clean_str(row["variable_para"])


def build_target_text(row: pd.Series) -> str:
    return clean_str(row["alternate_variable_para"])


def pair_fingerprint(anchor: str, positive: str) -> str:
    s = (anchor.strip().lower() + "\n---\n" + positive.strip().lower()).encode("utf-8")
    return hashlib.md5(s).hexdigest()


# ======================================================================================
# STAGE A — clean NCIt grounding 
# ======================================================================================
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    emb = last_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return F.normalize(emb, dim=-1)


def contrastive_scores_and_labels(q_emb, p_emb):
    scores = torch.matmul(q_emb, p_emb.T)
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return scores, labels


class BiEncoder(nn.Module):
    """Frozen CTDS-base + trainable LoRA adapter (r=8, alpha=16, target query/value)."""

    def __init__(self):
        super().__init__()
        base = AutoModel.from_pretrained(MODEL_NAME)
        cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
                         lora_dropout=0.1, target_modules=["query", "value"])
        self.encoder = get_peft_model(base, cfg)
        self.l2_normalize = True

    def encode(self, inputs):
        out = self.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return average_pool(out.last_hidden_state, inputs["attention_mask"])

    def forward(self, query, passage):
        return self.encode(query), self.encode(passage)


class BiEncoderDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.query = data["variable_para"].tolist()
        self.positive = data["alternate_variable_para"].tolist()

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        return {"query_text": self.query[idx], "passage_text": self.positive[idx]}


def biencoder_collate(batch):
    q = tokenizer([b["query_text"] for b in batch], max_length=MAX_LEN, padding=True,
                  truncation=True, return_tensors="pt")
    p = tokenizer([b["passage_text"] for b in batch], max_length=MAX_LEN, padding=True,
                  truncation=True, return_tensors="pt")
    return {"query": q, "passage": p}


def build_biencoder_loader(data: pd.DataFrame, shuffle: bool) -> DataLoader:
    return DataLoader(BiEncoderDataset(data), batch_size=BATCH_SIZE_A, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                      collate_fn=biencoder_collate, drop_last=shuffle)


@torch.no_grad()
def run_biencoder_validation(model: BiEncoder, loader: DataLoader, max_steps: int = 50) -> float:
    model.eval()
    losses: List[float] = []
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        q = {k: v.to(DEVICE) for k, v in batch["query"].items()}
        p = {k: v.to(DEVICE) for k, v in batch["passage"].items()}
        q_emb, p_emb = model(q, p)
        scores, labels = contrastive_scores_and_labels(q_emb, p_emb)
        losses.append(F.cross_entropy(scores / TEMPERATURE_A, labels).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_biencoder(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path) -> BiEncoder:
    _seed_everything(SEED_A)   # Stage A reproducibility
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = build_biencoder_loader(train_df, shuffle=True)
    val_loader = build_biencoder_loader(val_df, shuffle=False)
    model = BiEncoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_A, weight_decay=WEIGHT_DECAY_A)

    step = 0
    for epoch in range(EPOCHS_A):
        for batch in tqdm(train_loader, desc=f"stageA epoch {epoch}"):
            q = {k: v.to(DEVICE) for k, v in batch["query"].items()}
            p = {k: v.to(DEVICE) for k, v in batch["passage"].items()}
            q_emb, p_emb = model(q, p)
            scores, labels = contrastive_scores_and_labels(q_emb, p_emb)
            loss = F.cross_entropy(scores / TEMPERATURE_A, labels) / GRAD_ACCUM_STEPS_A
            loss.backward()
            step += 1
            if step % GRAD_ACCUM_STEPS_A == 0:
                if MAX_GRAD_NORM_A is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM_A)
                optimizer.step()
                optimizer.zero_grad()
            if step % 100 == 0:
                print(f"[train] epoch={epoch} step={step} loss={loss.item():.4f}")
            if step % VAL_EVERY_STEPS == 0:
                print(f"[val] step={step} loss={run_biencoder_validation(model, val_loader):.4f}")
            if step % CKPT_EVERY_STEPS == 0:
                torch.save({"model": model.state_dict(), "step": step, "epoch": epoch},
                           output_dir / f"biencoder_lora_step_{step}.pt")

    final = output_dir / "biencoder_lora_final.pt"
    torch.save({"model": model.state_dict(), "step": step, "epochs": EPOCHS_A}, final)
    print("[stageA] saved", final)
    return model


def run_stage_a() -> Path:
    """Train Stage A from the clean split (idempotent: skip if final ckpt exists)."""
    out = STAGE_A_DIR / "biencoder"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "biencoder_lora_final.pt"
    if final.exists():
        print("Stage A checkpoint exists, skipping:", final)
        return final

    full = pd.read_csv(CLEAN_FULL_SPLIT_PATH, low_memory=False)
    need = ["variable_para", "alternate_variable_para"]
    train_df = full[full["split"].eq("train")][need].dropna().reset_index(drop=True)
    val_df = full[full["split"].eq("val")][need].dropna().reset_index(drop=True)
    print("Stage A train/val:", train_df.shape, val_df.shape)
    train_biencoder(train_df, val_df, out)
    return final


# ======================================================================================
# STAGE B step 1 — load clean + augmented, derive REDCap rows
# ======================================================================================
def load_training_pairs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the augmented train file into clean vs REDCap pairs.
    """
    aug = pd.read_csv(AUGMENTED_TRAIN_PLUS_REDCAP_PATH, low_memory=False)
    aug["anchor"] = aug.apply(build_query_text, axis=1)
    aug["positive"] = aug.apply(build_target_text, axis=1)
    aug = aug[(aug["anchor"].str.len() > 0) & (aug["positive"].str.len() > 0)].copy()
    aug["pair_fp"] = [pair_fingerprint(a, p) for a, p in zip(aug["anchor"], aug["positive"])]

    clean_train = aug[aug["source_type"].eq("clean")].copy()
    redcap_raw = aug[aug["source_type"].eq("redcap_synthetic")].copy()
    redcap_raw["source_type"] = "redcap_synthetic_raw"
    print(f"[tagged] clean_train={len(clean_train)}  redcap_raw={len(redcap_raw)}")
    return clean_train, redcap_raw


# ======================================================================================
# STAGE B step 2 — SIMPLIFIED REDCap clean-positive filter
# ======================================================================================
GENERIC_TARGET_TERMS = {"status", "type", "other", "total", "score", "value", "result", "number",
                        "amount", "name", "date", "time", "day", "month", "year", "unknown",
                        "none", "id", "code"}
CRITICAL_QUALIFIERS = {"parent", "child", "children", "pediatric", "adult", "proxy", "self",
                       "caregiver", "least", "worst", "average", "current", "past", "last",
                       "previous", "baseline", "short", "form", "version", "item", "scale",
                       "subscale", "total", "left", "right", "upper", "lower", "bilateral"}
RACE_OPTION_TERMS = {"american indian", "alaska native", "asian", "black", "african american",
                     "native hawaiian", "pacific islander", "white", "middle eastern",
                     "north african", "unknown", "not reported", "decline", "prefer not", "other race"}
ETHNICITY_OPTION_TERMS = {"hispanic", "latino", "latina", "latinx", "spanish origin",
                          "not hispanic", "non hispanic", "unknown", "not reported", "decline", "prefer not"}
OPTION_CODE_HINTS = {"ai_an", "asian", "bl_aa", "black", "white", "nh_pi", "pi", "hi_la", "latino",
                     "hispanic", "mena", "unkn", "unknown", "not_rep", "notreported", "decline",
                     "prefer", "other"}
GENERIC_PARENT_TARGET_PHRASES = {"race and or ethnicity", "race and ethnicity", "parent race",
                                 "race parent", "race what is your race", "what is your race",
                                 "ethnicity", "hispanic or latino ethnicity", "race"}
OPTION_TYPE_HINTS = ["checkbox", "radio", "dropdown", "enum", "choice", "unchecked", "checked"]
ENUM_CONSTRAINT_HINTS = ["___", "checkbox", "radio", "dropdown", "unchecked", "checked", "enum", "constraint"]

# Normalizes text: lowercases, replaces every non-alphanumeric char with a space, and collapses whitespace
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", clean_str(s).lower())).strip()

# Returns the set of unique words in the normalized text
def _toks(s: str) -> set:
    return set(_norm(s).split())

# Returns True if any phrase from the list appears
def _has_any(text: str, phrases) -> bool:
    t = _norm(text)
    return any(_norm(p) in t for p in phrases if _norm(p))

# pulls the option value out of a raw REDCap query
def extract_option_label(anchor: str) -> str:
    """pull option label from raw query text: 'Race: Asian', ___suffix, or a known phrase."""
    raw = clean_str(anchor)
    m = re.search(r"\b(race|ethnicity)\s*[:=\-]\s*([^|;\n]+)", raw, flags=re.I)
    if m:
        opt = re.split(r"\b(field|type|description|enum|constraint)\b", m.group(2), flags=re.I)[0].strip()
        if opt:
            return opt[:120]
    if "___" in raw:
        m = re.search(r"___([a-zA-Z0-9_]+)", raw)
        if m:
            return m.group(1).strip()
    rn = _norm(raw)
    for phrase in sorted(RACE_OPTION_TERMS | ETHNICITY_OPTION_TERMS, key=len, reverse=True):
        if _norm(phrase) in rn:
            return phrase
    return ""

# Returns True if the query looks like a single choice/option field
def _is_option_level(anchor: str) -> bool:
    a = clean_str(anchor).lower()
    return "___" in a or any(h in a for h in OPTION_TYPE_HINTS) or bool(extract_option_label(anchor))

# Returns True if the query text carries enum
def _has_enum_or_constraints(anchor: str) -> bool:
    a = clean_str(anchor).lower()
    return any(h in a for h in ENUM_CONSTRAINT_HINTS)

# Returns True if the query has enough words (≥12) to count as having a real description 
def _has_description(anchor: str) -> bool:
    return len(_norm(anchor).split()) >= 12


def classify_and_decide(row) -> pd.Series:
    """Tag targeted-option flags AND make the keep/drop decision."""
    anchor, positive = clean_str(row["anchor"]), clean_str(row["positive"])
    option_label = extract_option_label(anchor)
    is_option = _is_option_level(anchor)

    q_for_match = f"{anchor} {option_label}"
    is_race = is_option and _has_any(q_for_match, {"race"} | RACE_OPTION_TERMS)
    is_ethnicity = is_option and _has_any(q_for_match, {"ethnicity"} | ETHNICITY_OPTION_TERMS)
    is_checkbox_enum = is_option and _has_enum_or_constraints(anchor)

    option_terms_for_target = RACE_OPTION_TERMS | ETHNICITY_OPTION_TERMS | (OPTION_CODE_HINTS - {"race", "other"})
    overlap = (_toks(option_label) & _toks(positive)) - {"race", "ethnicity", "parent", "what", "is", "your"}
    target_is_option_like = _has_any(positive, option_terms_for_target) or bool(overlap)
    target_is_generic_parent = _has_any(positive, GENERIC_PARENT_TARGET_PHRASES) and not target_is_option_like

    targeted = is_option and (is_race or is_ethnicity or is_checkbox_enum) and not target_is_generic_parent
    reasons_tag = []
    if targeted:
        if is_race:
            reasons_tag.append("race_option_level")
        if is_ethnicity:
            reasons_tag.append("ethnicity_option_level")
        if is_checkbox_enum:
            reasons_tag.append("checkbox_or_enum_option")
        if target_is_option_like:
            reasons_tag.append("target_option_like")

    protect = is_option and target_is_option_like and not target_is_generic_parent
    ttoks, qtoks = _toks(positive), _toks(anchor)
    drop = []
    if not protect:
        if len(ttoks) == 1 and next(iter(ttoks), "") in GENERIC_TARGET_TERMS:
            drop.append("target_too_generic_single_token")
        if len(ttoks) <= 3 and len(ttoks & GENERIC_TARGET_TERMS) >= max(1, len(ttoks) // 2):
            drop.append("target_too_generic")
    if len(qtoks) < 5 and not _has_enum_or_constraints(anchor) and not _has_description(anchor):
        drop.append("query_too_short_no_enum_no_description")
    if (not protect and _norm(anchor) == _norm(positive)
            and not _has_enum_or_constraints(anchor) and not _has_description(anchor)):
        drop.append("field_target_exact_copy_without_context")
    if not protect:
        missing = (ttoks & CRITICAL_QUALIFIERS) - (qtoks & CRITICAL_QUALIFIERS)
        if missing:
            drop.append("missing_critical_qualifier_in_query:" + ";".join(sorted(missing)))
    if target_is_generic_parent:
        drop.append("option_query_to_generic_parent_positive_risk")

    return pd.Series({
        "option_label": option_label,
        "is_race_option_query": bool(is_race),
        "is_ethnicity_option_query": bool(is_ethnicity),
        "is_checkbox_or_enum_option_query": bool(is_checkbox_enum),
        "is_targeted_option_row": bool(targeted),
        "targeted_reason": ";".join(reasons_tag),
        "keep_redcap": len(drop) == 0,
        "filter_reason": " | ".join(drop) if drop else "keep",
    })


def filter_redcap(redcap_raw: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    if len(redcap_raw) == 0:
        raise ValueError("No REDCap rows detected. Check augmented / clean split inputs.")
    tags = redcap_raw.apply(classify_and_decide, axis=1)
    tagged = pd.concat([redcap_raw.reset_index(drop=True), tags.reset_index(drop=True)], axis=1)
    keep = tagged[tagged["keep_redcap"]].copy()
    data_dir.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(data_dir / "redcap_synthetic_filter_audit.csv", index=False)
    keep.to_csv(data_dir / "redcap_synthetic_filtered_keep.csv", index=False)
    print(f"redcap_raw={len(redcap_raw)} keep={len(keep)} "
          f"keep_rate={len(keep) / max(len(redcap_raw), 1):.3f} "
          f"targeted_kept={int(keep['is_targeted_option_row'].sum())}")
    return keep


# ======================================================================================
# STAGE B step 3 — build 15:1 oversampled training pairs
# ======================================================================================
def build_mixed_training_pairs(clean_train: pd.DataFrame, redcap_keep: pd.DataFrame,
                               data_dir: Path) -> pd.DataFrame:
    assert len(redcap_keep) > 0, "No REDCap rows kept after filtering."

    clean_pairs = clean_train[["anchor", "positive", "source_type", "pair_fp"]].drop_duplicates("pair_fp").copy()
    clean_pairs["is_targeted_option_row"] = False

    redcap_pairs = redcap_keep[[
        "anchor", "positive", "source_type", "pair_fp", "is_targeted_option_row",
        "is_race_option_query", "is_ethnicity_option_query",
        "is_checkbox_or_enum_option_query", "option_label",
    ]].drop_duplicates("pair_fp").copy()

    target_redcap_n = int(math.ceil(len(clean_pairs) / CLEAN_TO_REDCAP_RATIO))
    targeted_pool = redcap_pairs[redcap_pairs["is_targeted_option_row"]].copy()
    other_pool = redcap_pairs[~redcap_pairs["is_targeted_option_row"]].copy()

    def sample_pool(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        if n <= 0 or len(pool) == 0:
            return pool.head(0).copy()
        replace = REDCAP_OVERSAMPLE_WITH_REPLACEMENT or (n > len(pool))
        return pool.sample(n=n, replace=replace, random_state=seed).copy()

    targeted_n = 0 if len(targeted_pool) == 0 else int(math.ceil(target_redcap_n * TARGETED_REDCAP_MIN_FRACTION))
    targeted_sample = sample_pool(targeted_pool, targeted_n, SEED_B)
    other_sample = sample_pool(other_pool, target_redcap_n - targeted_n, SEED_B + 1)
    remaining = target_redcap_n - len(targeted_sample) - len(other_sample)
    fill_sample = sample_pool(redcap_pairs, remaining, SEED_B + 2) if remaining > 0 else redcap_pairs.head(0).copy()

    redcap_sample = pd.concat([targeted_sample, other_sample, fill_sample], ignore_index=True)
    redcap_sample = redcap_sample.sample(frac=1.0, random_state=SEED_B).reset_index(drop=True)
    # Make oversampled fingerprints unique so downstream dedup does not collapse them.
    redcap_sample["pair_fp"] = (redcap_sample["pair_fp"].astype(str)
                                + "__oversample_" + np.arange(len(redcap_sample)).astype(str))

    train_pairs = pd.concat([clean_pairs, redcap_sample], ignore_index=True, sort=False)
    train_pairs = train_pairs.sample(frac=1.0, random_state=SEED_B).reset_index(drop=True)

    actual_ratio = len(clean_pairs) / max(len(redcap_sample), 1)
    print(f"clean={len(clean_pairs)} redcap_sample={len(redcap_sample)} "
          f"actual_ratio={actual_ratio:.2f} target={CLEAN_TO_REDCAP_RATIO}")
    assert abs(actual_ratio - CLEAN_TO_REDCAP_RATIO) < 0.15, f"ratio off: {actual_ratio:.3f}"

    data_dir.mkdir(parents=True, exist_ok=True)
    train_pairs.to_csv(data_dir / "training_pairs_clean_redcap_15to1.csv", index=False)
    return train_pairs


# ======================================================================================
# STAGE B step 4 — checkpoint loader 
# ======================================================================================
def _extract_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model", "module"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def _key_candidates(k: str) -> List[str]:
    cands = [k]
    for pref in ("encoder.", "model.", "module.", "0.", "0.auto_model.",
                 "sentence_transformer.", "sentence_transformer.0."):
        if k.startswith(pref):
            cands.append(k[len(pref):])
    for a, b in (("encoder.base_model.model.", "base_model.model."),
                 ("model.base_model.model.", "base_model.model."),
                 ("auto_model.base_model.model.", "base_model.model.")):
        if k.startswith(a):
            cands.append(b + k[len(a):])
    if k.startswith("encoder.layer."):
        cands.append("base_model.model." + k)
    return list(dict.fromkeys(cands))


def load_lora_sentence_transformer(base_model_name: str, ckpt_path: Path, device, trainable: bool):
    """Load a LoRA checkpoint into a SentenceTransformer. Handles the notebook key layouts."""
    state = _extract_state_dict(ckpt_path)
    lora_a = [k for k in state if "lora_A" in k and hasattr(state[k], "shape")]
    if not lora_a:
        raise ValueError("No lora_A keys found; not a PEFT LoRA checkpoint?")
    r = int(state[lora_a[0]].shape[0])
    targets = [tm for tm in ("query", "value", "key", "dense")
               if any(f".{tm}.lora_A" in k for k in lora_a)] or ["query", "value"]

    st = SentenceTransformer(base_model_name, device=str(device))
    transformer = st._first_module()
    cfg = LoraConfig(r=r, lora_alpha=32 if r == 16 else max(2 * r, 16), target_modules=targets,
                     lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION")
    peft_model = get_peft_model(transformer.auto_model, cfg)

    model_sd = peft_model.state_dict()
    mapped = {}
    for k, v in state.items():
        if not hasattr(v, "shape"):
            continue
        for cand in _key_candidates(k):
            if cand in model_sd and tuple(model_sd[cand].shape) == tuple(v.shape):
                mapped[cand] = v
                break
    peft_model.load_state_dict(mapped, strict=False)
    transformer.auto_model = peft_model
    st.to(device)

    matched = [k for k in mapped if "lora_" in k]
    if not matched:
        raise RuntimeError("No LoRA keys matched; checkpoint not loaded correctly.")
    if trainable:
        st.train()
        for name, p in st.named_parameters():
            p.requires_grad = "lora_" in name
    else:
        st.eval()
        for p in st.parameters():
            p.requires_grad = False
    print(f"Loaded {ckpt_path.name}: matched {len(matched)} LoRA keys (trainable={trainable}).")
    return st


def encode_with_grad(st, texts, device):
    feats = batch_to_device(st.tokenize(list(texts)), device)
    return F.normalize(st(feats)["sentence_embedding"], p=2, dim=1)


@torch.no_grad()
def encode_no_grad(st, texts, device):
    st.eval()
    return encode_with_grad(st, texts, device)


# ======================================================================================
# STAGE B step 5 — anti-drift training loop
# ======================================================================================
class PairTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.anchor = df["anchor"].astype(str).tolist()
        self.positive = df["positive"].astype(str).tolist()
        self.source_type = df.get("source_type", pd.Series(["unknown"] * len(df))).astype(str).tolist()
        self.targeted = df.get("is_targeted_option_row", pd.Series([False] * len(df))).fillna(False).astype(bool).tolist()

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        return {"anchor": self.anchor[idx], "positive": self.positive[idx],
                "source_type": self.source_type[idx], "is_targeted_option_row": self.targeted[idx]}


def collate_pairs(batch):
    return {k: [x[k] for x in batch] for k in ("anchor", "positive", "source_type", "is_targeted_option_row")}


def train_stage_b(train_pairs: pd.DataFrame, ckpt_dir: Path) -> List[Path]:
    _seed_everything(SEED_B)   # Stage B reproducibility (independent of Stage A)
    loader = DataLoader(PairTextDataset(train_pairs), batch_size=BATCH_SIZE_B,
                        shuffle=True, num_workers=0, collate_fn=collate_pairs, drop_last=True)
    steps_per_epoch = len(loader)
    ckpt_steps = {max(1, math.ceil(steps_per_epoch * f)): f for f in CHECKPOINT_FRACTIONS}
    max_steps = max(ckpt_steps)
    print(f"steps_per_epoch={steps_per_epoch} max_steps={max_steps} ckpt_steps={ckpt_steps}")

    saved: List[Path] = []
    for adw in ANTI_DRIFT_WEIGHTS:
        label = str(adw).replace(".", "p")
        run_dir = ckpt_dir / f"antidrift_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        reference = load_lora_sentence_transformer(BASE_MODEL_NAME, INIT_CHECKPOINT_PATH, DEVICE, trainable=False)
        student = load_lora_sentence_transformer(BASE_MODEL_NAME, INIT_CHECKPOINT_PATH, DEVICE, trainable=True)

        # student must start ~identical to reference
        with torch.no_grad():
            s = encode_no_grad(student, train_pairs["anchor"].head(4).tolist(), DEVICE)
            t = encode_no_grad(reference, train_pairs["anchor"].head(4).tolist(), DEVICE)
            assert float((s * t).sum(1).mean()) > 0.999, "student != reference at init"

        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                      lr=LEARNING_RATE_B, weight_decay=WEIGHT_DECAY_B)
        student.train()
        reference.eval()
        step = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, total=max_steps, desc=f"stageB anti_drift={adw:g}")
        for batch_idx, batch in enumerate(pbar):
            if step >= max_steps:
                break
            s_anchor = encode_with_grad(student, batch["anchor"], DEVICE)
            s_pos = encode_with_grad(student, batch["positive"], DEVICE)
            logits = torch.matmul(s_anchor, s_pos.T) / TEMPERATURE_B
            labels = torch.arange(logits.size(0), device=DEVICE)
            contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

            drift = torch.tensor(0.0, device=DEVICE)
            if adw > 0:
                with torch.no_grad():
                    t_anchor = encode_no_grad(reference, batch["anchor"], DEVICE)
                    t_pos = encode_no_grad(reference, batch["positive"], DEVICE)
                drift = 0.5 * (F.mse_loss(s_anchor, t_anchor) + F.mse_loss(s_pos, t_pos))

            loss = (contrastive + adw * drift) / GRAD_ACCUM_STEPS_B
            loss.backward()
            if (batch_idx + 1) % GRAD_ACCUM_STEPS_B == 0:
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], MAX_GRAD_NORM_B)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.set_postfix(loss=f"{contrastive.item():.4f}", drift=f"{float(drift):.6f}")

            if step in ckpt_steps:
                frac = str(ckpt_steps[step]).replace(".", "p")
                path = run_dir / f"biencoder_lora_step{step}_epochfrac{frac}_antidrift{label}.pt"
                _save_lora(student, path, {"anti_drift_distill_weight": adw,
                                           "checkpoint_step": step, "temperature": TEMPERATURE_B})
                saved.append(path)
                print("Saved", path)

        del student, reference, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return saved


def _save_lora(student, ckpt_path: Path, metadata: dict) -> None:
    student.eval()
    full_sd = student._first_module().auto_model.state_dict()
    lora_sd = {k: v.detach().cpu() for k, v in full_sd.items() if "lora_" in k}
    if not lora_sd:
        raise RuntimeError("No LoRA weights to save.")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": lora_sd, "base_model_name": BASE_MODEL_NAME,
                "init_checkpoint_path": str(INIT_CHECKPOINT_PATH), **metadata}, ckpt_path)


def run_stage_b() -> List[Path]:
    data_dir = STAGE_B_DIR / "data"
    ckpt_dir = STAGE_B_DIR / "checkpoints"
    clean_train, redcap_raw = load_training_pairs()
    redcap_keep = filter_redcap(redcap_raw, data_dir)
    train_pairs = build_mixed_training_pairs(clean_train, redcap_keep, data_dir)
    return train_stage_b(train_pairs, ckpt_dir)


# ======================================================================================
def main() -> None:
    # Stage A is REUSED from the original checkpoint (read-only) to avoid a multi-hour retrain.
    # To train Stage A from scratch into the sandbox instead, point INIT_CHECKPOINT_PATH at
    # STAGE_A_DIR/biencoder/biencoder_lora_final.pt and uncomment the next line:
    run_stage_a()
    assert INIT_CHECKPOINT_PATH.exists(), f"Stage A checkpoint not found (read-only reuse): {INIT_CHECKPOINT_PATH}"
    print("Reusing Stage A checkpoint (read-only):", INIT_CHECKPOINT_PATH)

    ckpts = run_stage_b()  # produces the anti-drift LoRA checkpoints (the eval targets)
    print("\nTRAINING DONE. Stage B checkpoints:")
    for c in ckpts:
        print("  ", c)


if __name__ == "__main__":
    main()
