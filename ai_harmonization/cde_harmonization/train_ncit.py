#!/usr/bin/env python
# coding: utf-8

# # Train/fine-tune ctds embedding model for CDE use case
# - we will use some approaches from NVIDIA nemotron models, links below

# running on no dups from NCIt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os
import pandas as pd


# start by restarting kernel to clear GPU memory
# del model  # delete your model
# gc.collect()  # clear Python memory
# torch.cuda.empty_cache()  # release cached GPU memory

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ### relevant resources from NVIDIA

# relevant resources from NVIDIA nemotron model specs:
# embedding model configurations, including base model, average pooling and troubleshooting: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/embed/README.md
# recipe to train biencoder is here: https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/retrieval/train_bi_encoder.py
# above is called by https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/embed/stage2_finetune/train.py
# nemotron model definitions are here: https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/models/nemotron_v3/model.py
# biencoder base yml is here: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/embed/stage2_finetune/biencoder_base.yaml
# see above yml for code configurations
# HF doc is here: https://huggingface.co/blog/nvidia/domain-specific-embedding-finetune


# ### Data set up and training
# - test batch size = 64
# - in batch negatives


# =========================
# CONFIG
# see example yml https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/embed/stage2_finetune/biencoder_base.yaml
# optimizer adam lr 5e-6
# =========================
MODEL_NAME = "uc-ctds/bge-large-en-v1.5-bio-mapping"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# on a V100 only batch size of 6 works which is very slow
# move to nvl nodes, and run there?
BATCH_SIZE = 64
LR = 5e-6
# test 2-3, start with 2
EPOCHS = 3
TEMPERATURE = 0.02
# max len for query and passage per nvidia code
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# with batch size of 64
# Steps per epoch = 915,789 / 64 ≈ 14,309 # this is with dups
# steps per epoch with no dups = 188541/64 ~ about 2945
# Total steps for 3 epochs = ~42k # with dups
# total steps for 3 epochs with no dups ~9000
# start with validate every 1500 steps
# and save a ckpt every 1000 steps
VAL_EVERY_STEPS = 1000  # ~8-9 validations total
CKPT_EVERY_STEPS = 3000  # three chkpoints, one per epoch


# ### custom functions

# =========================
# define average token embedding POOLING
# see example for mean pooling here
# https://github.com/NVIDIA-NeMo/Nemotron/blob/8d4f39259afe0e01013ba1e7d8fa7d18bae01c57/use-case-examples/RAG%20Agent%20with%20Nemotron%20RAG%20Models/RAG%20Agent%20with%20Nemotron%20RAG%20Models.ipynb#L163


# =========================
def average_pool(last_hidden_states, attention_mask):
    """Avg pooling with attention mask"""
    last_hidden_states = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    emb = last_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    emb = F.normalize(emb, dim=-1)
    return emb


# =========================
# in-batch contrastive learning
# instead of 5 examples, we will use 64 in a batch
# for each query i, one example is postive, remaining 63 are negative

# What the following does:

# For each query i:

# Positive: p_emb[i]
# Negatives: all other passages in the batch (p_emb[j≠i])

# So each row of scores looks like:


# [q_i · p_0, q_i · p_1, ..., q_i · p_(B-1)]
# =========================
def contrastive_scores_and_labels(q_emb, p_emb):
    """
    q_emb: [B, H]
    p_emb: [B, H]

    scores: [B, B]
    labels: [0..B-1]
    """
    scores = torch.matmul(q_emb, p_emb.T)  # full similarity matrix
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return scores, labels


# =============================
# to load training data from ncit
def load_csv(path):
    df = pd.read_csv(path)
    return df.to_dict("records")


# =========================
# collate fn for a batch
# =========================
def collate_fn(batch):
    return {
        "query": {
            "input_ids": torch.stack([b["q_input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["q_attention_mask"] for b in batch]),
        },
        "passage": {
            "input_ids": torch.stack([b["p_input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["p_attention_mask"] for b in batch]),
        },
    }


# ### model


# MODEL
# we will follow the contrastive learning diagram shown in the link below
# https://huggingface.co/blog/nvidia/domain-specific-embedding-finetune
# except we will use a batch size of 64, where 1 passage is correct, 63 are in-batch negatives
# =========================
class BiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = AutoModel.from_pretrained(MODEL_NAME)

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,  # low-rank size
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],  # attention layers
        )
        self.encoder = get_peft_model(base_model, lora_config)
        self.l2_normalize = True

    def encode(self, inputs):
        outputs = self.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return average_pool(outputs.last_hidden_state, inputs["attention_mask"])

    def forward(self, query, passage):
        q_emb = self.encode(query)  # [B, H]
        p_emb = self.encode(passage)  # [B, H]
        return q_emb, p_emb


# ### data prep code


# =========================
# DATASET
# parse /opt/gpudata/aartiv/heal_cde/ncit/thesaurus_finetuning_data_nodup.csv
# =========================
class RetrievalDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        query = row["variable_para"]
        positive = row["alternate_variable_para"]

        q = self.tokenizer(
            query,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        p = self.tokenizer(
            positive,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "q_input_ids": q["input_ids"].squeeze(0),
            "q_attention_mask": q["attention_mask"].squeeze(0),
            "p_input_ids": p["input_ids"].squeeze(0),
            "p_attention_mask": p["attention_mask"].squeeze(0),
        }


# =============================================
# data loader builder


def build_dataloader(data_path, tokenizer, shuffle=True):
    data = load_csv(data_path)
    dataset = RetrievalDataset(data, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return loader


# ### validation loop
# note: no external validation set here
# size of training set quite limited, this is only training loss


# =========================
# VALIDATION LOOP
# =========================
@torch.no_grad()
def run_validation(model, dataloader):
    model.eval()
    losses = []

    for step, batch in enumerate(dataloader):
        if step >= 20:  # small validation subset
            break

        query = {k: v.to(DEVICE) for k, v in batch["query"].items()}
        passage = {k: v.to(DEVICE) for k, v in batch["passage"].items()}

        q_emb, p_emb = model(query, passage)

        scores, labels = contrastive_scores_and_labels(q_emb, p_emb)

        if model.l2_normalize:
            scores = scores / TEMPERATURE

        loss = F.cross_entropy(scores, labels)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


# ### training loop


def train(ft_data):
    print("starting training")

    dataloader = build_dataloader(ft_data, tokenizer)

    model = BiEncoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    global_step = 0

    for epoch in range(EPOCHS):
        print(f"epoch: {epoch}")
        for step, batch in enumerate(dataloader):
            query = {k: v.to(DEVICE) for k, v in batch["query"].items()}
            passage = {k: v.to(DEVICE) for k, v in batch["passage"].items()}

            q_emb, p_emb = model(query, passage)

            scores, labels = contrastive_scores_and_labels(q_emb, p_emb)

            if model.l2_normalize:
                scores = scores / TEMPERATURE

            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # -------------------------
            # training log
            # -------------------------
            if global_step % 100 == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} loss={loss.item():.4f}"
                )

            # -------------------------
            # validation
            # -------------------------
            if global_step % VAL_EVERY_STEPS == 0:
                val_loss = run_validation(model, dataloader)
                print(f"[val] step={global_step} loss={val_loss:.4f}")

            # -------------------------
            # checkpoint
            # -------------------------
            if global_step % CKPT_EVERY_STEPS == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                    },
                    f"/opt/gpudata/aartiv/heal_cde/ncit/checkpoints/peft_step_no_dups_{global_step}.pt",
                )
                print(f"[ckpt] saved step {global_step}")


def main():
    # path to finetuning data
    ft_data = "/opt/gpudata/aartiv/heal_cde/ncit/thesaurus_finetuning_data_nodup.csv"
    train(ft_data)


if __name__ == "__main__":
    main()
