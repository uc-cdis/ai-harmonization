import torch
from transformers import AutoTokenizer, AutoModel


class MedGemmaEmbeddings:
    def __init__(self, model_name="google/medgemma-4b-pt", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO: commented lines chash kernel when ran locally
        # device_map=auto offload model to disk, which is not recommended
        # Need to test commented lines on more powerful machine
        # self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        # model = AutoModel.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.device = model.device
        self.model = model.eval()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # shape: (batch, seq, dim)
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(last_hidden.size())
                .float()
            )
            masked = last_hidden * mask
            summed = masked.sum(1)
            counts = mask.sum(1)
            mean_pooled = summed / counts.clamp(min=1e-9)
            return mean_pooled.cpu().numpy().tolist()


class QwenEmbeddings:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = AutoModel.from_pretrained(model_name)
        self.model = model.eval()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            pooled = outputs.get("pooled_output", None)
            if pooled is not None:
                return pooled.cpu().numpy().tolist()
            last_hidden = outputs.hidden_states[-1]  # shape: (batch, seq, dim)
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(last_hidden.size())
                .float()
            )
            masked = last_hidden * mask
            summed = masked.sum(1)
            counts = mask.sum(1)
            mean_pooled = summed / counts.clamp(min=1e-9)
            return mean_pooled.cpu().numpy().tolist()
