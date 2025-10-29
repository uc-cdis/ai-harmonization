import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class BaseEmbeddings:
    def __init__(self, model_name, device=None, trust_remote_code=False):
        # Load tokenizer/model, set device, set eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            .to(self.device)
            .eval()
        )

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def mean_pool(self, last_hidden, mask):
        # Shape: (batch, seq_len, dim), mask: (batch, seq_len, 1)
        masked = last_hidden * mask
        summed = masked.sum(1)
        counts = mask.sum(1)
        mean_pooled = summed / counts.clamp(min=1e-9)
        return mean_pooled

    def embed_documents(self, texts):
        raise NotImplementedError("Implement in subclass.")


class MedGemmaEmbeddings(BaseEmbeddings):
    def __init__(self, model_name="google/medgemma-4b-it", device=None):
        super().__init__(model_name, device, trust_remote_code=True)

    def embed_documents(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(last_hidden.size())
                .float()
            )
            pooled = self.mean_pool(last_hidden, mask)
            return pooled.cpu().numpy().tolist()


class QwenEmbeddings(BaseEmbeddings):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device=None):
        super().__init__(model_name, device, trust_remote_code=True)

    def embed_documents(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # recommended by Qwen
            pooled = outputs.get("pooled_output", None)
            if pooled is not None:
                return pooled.cpu().numpy().tolist()
            # fallback to mean pool
            last_hidden = outputs.hidden_states[-1]
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(last_hidden.size())
                .float()
            )
            pooled = self.mean_pool(last_hidden, mask)
            return pooled.cpu().numpy().tolist()


class BGEEmbeddings(BaseEmbeddings):
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", device=None):
        super().__init__(model_name, device=device)

    def embed_documents(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = self.mean_pool(model_output.last_hidden_state, mask)
        pooled = pooled.cpu().numpy()
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled.tolist()
