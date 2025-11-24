import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from schema.payload import Payload


class BrandEmbeddingGenerator:
    def __init__(self, payload: Payload):
        self.payload = payload
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("🔥 Using device:", self.device)


    def _payload_to_text(self) -> str:
        return (
            f"{self.payload.brandName} is a {self.payload.industry} brand "
            f"running a campaign on {self.payload.platform} aimed at {self.payload.audience} "
            f"in {self.payload.region}. The campaign targets {self.payload.customerSegment} customers, "
            f"mainly {self.payload.gender} audience, with a budget of {self.payload.budget}. "
            f"Description: {self.payload.description}"
        )

    def create_embeddings(self, output_path: str = "embeddings/brand_query_embeddings.npy"):
        text = self._payload_to_text()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        np.save(output_path, emb)
        print(f"✅ Saved embedding with shape {emb.shape} → {output_path}")
        return emb
