# ---------------------------
# NLP Embedding Model (BERT Layer)
# Member 3 - Influencer Voice & Brand-Fit Matchmaker
# ---------------------------

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------
# 1️⃣ Load Pretrained Model
# ---------------------------
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embedding for a given text
def get_embedding(text):
    if not isinstance(text, str) or text.strip() == "":
        # return zero vector if text is empty
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean Pooling
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# ---------------------------
# 2️⃣ Process Influencer Data
# ---------------------------
influencer_df = pd.read_csv("influencer_data.csv")

# Check for column named "description" (you can change this if needed)
if "description" not in influencer_df.columns:
    raise ValueError("Column 'description' not found in influencer_data.csv. Please rename your text column.")

influencer_embeddings = []
print("🔹 Generating Influencer Embeddings...")
for text in tqdm(influencer_df["description"].fillna("")):
    influencer_embeddings.append(get_embedding(text))

# Convert to DataFrame
influencer_emb_df = pd.DataFrame(influencer_embeddings)
influencer_emb_df.to_csv("influencer_embeddings.csv", index=False)
np.save("influencer_embeddings.npy", influencer_embeddings)
print("✅ Influencer embeddings saved as influencer_embeddings.csv & influencer_embeddings.npy")

# ---------------------------
# 3️⃣ Process Brand Data
# ---------------------------
brand_df = pd.read_csv("brand_data.csv")

if "description" not in brand_df.columns:
    raise ValueError("Column 'description' not found in brand_data.csv. Please rename your text column.")

brand_embeddings = []
print("🔹 Generating Brand Embeddings...")
for text in tqdm(brand_df["description"].fillna("")):
    brand_embeddings.append(get_embedding(text))

# Convert to DataFrame
brand_emb_df = pd.DataFrame(brand_embeddings)
brand_emb_df.to_csv("brand_embeddings.csv", index=False)
np.save("brand_embeddings.npy", brand_embeddings)
print("✅ Brand embeddings saved as brand_embeddings.csv & brand_embeddings.npy")

# ---------------------------
# 4️⃣ Summary Output
# ---------------------------
print("\n🎯 All embeddings generated successfully!")
print(f"Influencer Data: {len(influencer_emb_df)} embeddings of size {influencer_emb_df.shape[1]}")
print(f"Brand Data: {len(brand_emb_df)} embeddings of size {brand_emb_df.shape[1]}")
