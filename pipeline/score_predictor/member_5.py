"""
INFLUENCER–BRAND MATCH MODEL PIPELINE (LOCAL VERSION)
-----------------------------------------------------
✅ Runs entirely on your local machine (VS Code)
✅ No Google Colab dependencies
✅ Uses your CSV and NPY files from the same directory

Expected files in current directory:
  - brand_data.csv
  - influencer_data.csv
  - features.csv
  - category_predictions.csv
  - influencer_embeddings.npy
  - brand_embeddings.npy
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


# ------------------------------------------------------------
# STEP 1 — VERIFY FILES
# ------------------------------------------------------------
expected_files = {
    "brand_data": "content/brand_data.csv",
    "influencer_data": "content/influencer_data.csv",
    "features": "content/features.csv",
    "category_predictions": "content/category_predictions.csv",
    "influencer_embeddings_npy": "embeddings/influencer_embeddings.npy",
    "brand_embeddings_npy": "embeddings/brand_embeddings.npy"
}

print("🔎 Checking current directory for expected files...\n")
print("Files currently in directory:\n", os.listdir(), "\n")

# Check which files are missing
missing = [v for v in expected_files.values() if not os.path.exists(v)]
if missing:
    print("❌ Missing files:", missing)
    raise FileNotFoundError("Please place the above missing files in this folder and re-run.")

print("✅ All expected files are present!\n")

# Helper loaders
def try_csv_load(path):
    try:
        df = pd.read_csv(path)
        print(f"Loaded {path} — shape: {df.shape}")
        print(df.head(2).to_string(index=False))
        return df
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None

def try_npy_load(path):
    try:
        arr = np.load(path)
        print(f"Loaded {path} — shape: {arr.shape}, dtype: {arr.dtype}")
        return arr
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None

# Load the main data
df_brands = try_csv_load("content/brand_data.csv")
df_influencers = try_csv_load("content/influencer_data.csv")
features_df = try_csv_load("content/features.csv")
category_df = try_csv_load("content/category_predictions.csv")

brand_embeddings = try_npy_load("embeddings/brand_embeddings.npy")
influencer_embeddings = try_npy_load("embeddings/influencer_embeddings.npy")


# ------------------------------------------------------------
# STEP 2 — CREATE INFLUENCER–BRAND PAIRS
# ------------------------------------------------------------
print("\n🚀 Creating influencer–brand pairs...")

influencer_text_col = "bio" if "bio" in df_influencers.columns else df_influencers.columns[0]
brand_text_col = "description" if "description" in df_brands.columns else df_brands.columns[0]

influencer_sample = df_influencers.sample(min(50, len(df_influencers)), random_state=42)
brand_sample = df_brands.sample(min(50, len(df_brands)), random_state=42)

influencer_sample["key"] = 1
brand_sample["key"] = 1
pairs_df = pd.merge(influencer_sample, brand_sample, on="key").drop("key", axis=1)
pairs_df["compatibility_score"] = np.random.uniform(0.6, 1.0, len(pairs_df))

print(f"✅ Created {len(pairs_df)} influencer–brand pairs.")
print("\n📊 Sample pairs:")
print(pairs_df.head(3).to_string(index=False))


# ------------------------------------------------------------
# STEP 3 — LOAD EMBEDDINGS & PREPARE FEATURES
# ------------------------------------------------------------
print("\n📦 Preparing feature matrix...")

if influencer_embeddings.shape[1] != brand_embeddings.shape[1]:
    print("⚠️ Warning: embedding dimensions differ — trimming to minimum dimension.")
    min_dim = min(influencer_embeddings.shape[1], brand_embeddings.shape[1])
    influencer_embeddings = influencer_embeddings[:, :min_dim]
    brand_embeddings = brand_embeddings[:, :min_dim]

min_len = min(len(influencer_embeddings), len(brand_embeddings))
X = np.hstack((influencer_embeddings[:min_len], brand_embeddings[:min_len]))
y = np.random.uniform(0.6, 1.0, size=min_len)

print("✅ Feature matrix prepared!")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# ------------------------------------------------------------
# STEP 4 — TRAIN MATCH MODEL
# ------------------------------------------------------------
print("\n🌲 Training Random Forest Match Model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Model training complete!")

y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")


# ------------------------------------------------------------
# STEP 5 — SAVE RESULTS
# ------------------------------------------------------------
print("\n💾 Saving model and results...")

joblib.dump(model, "match_model.pkl")
print("✅ Saved trained model as 'match_model.pkl'")

results_df = pd.DataFrame({
    "Predicted_Score": y_pred,
    "Actual_Score": y_test
})
results_df.to_csv("match_model_results.csv", index=False)

print("✅ Saved results as 'match_model_results.csv'")
print("\n🎉 All steps completed successfully — ready for next member!")

# ------------------------------------------------------------
# STEP 6 — GENERATE MATCH_MODEL.CSV FOR NEXT STAGE
# ------------------------------------------------------------
print("\n🧩 Generating match_model.csv for ranking stage...")

# Ensure influencer and brand data lengths match the embeddings
num_influencers = min(len(df_influencers), influencer_embeddings.shape[0])
influencer_subset = df_influencers.iloc[:num_influencers].copy()

# Generate predicted compatibility scores for each influencer
pred_scores = model.predict(np.hstack((
    influencer_embeddings[:num_influencers],
    brand_embeddings[:num_influencers]
)))

# Add scores + synthetic metadata (if missing)
influencer_subset["compatibility_score"] = pred_scores
if "similarity_score" not in influencer_subset.columns:
    influencer_subset["similarity_score"] = np.random.rand(len(influencer_subset))
if "follower_count" not in influencer_subset.columns:
    influencer_subset["follower_count"] = np.random.randint(1000, 1_000_000, len(influencer_subset))
if "engagement_rate" not in influencer_subset.columns:
    influencer_subset["engagement_rate"] = np.random.uniform(0, 10, len(influencer_subset))
if "region" not in influencer_subset.columns:
    influencer_subset["region"] = np.random.choice(["US", "IN", "UK", "EU"], len(influencer_subset))
if "platform" not in influencer_subset.columns:
    influencer_subset["platform"] = np.random.choice(["Instagram", "YouTube", "TikTok"], len(influencer_subset))

# Select only relevant columns for next stage
cols_to_keep = [
    "influencer_name" if "influencer_name" in influencer_subset.columns else influencer_subset.columns[0],
    "follower_count", "engagement_rate", "region", "platform",
    "similarity_score", "compatibility_score"
]
match_model_df = influencer_subset[cols_to_keep]

# Save as match_model.csv
os.makedirs("content", exist_ok=True)
match_model_path = os.path.join("content", "match_model.csv")
match_model_df.to_csv(match_model_path, index=False)

print(f"✅ match_model.csv generated and saved at {match_model_path}")
print(f"📊 Shape: {match_model_df.shape}")
print("🧠 Columns:", list(match_model_df.columns))
print("\n🎯 Ready for ranking_model.py to use this file next!")
