import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


DATA_DIR = "content"
EMB_DIR = 'embeddings'

# ------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------
def load_embeddings(path=os.path.join(EMB_DIR, "influencer_embeddings.npy")):
    if os.path.exists(path):
        emb = np.load(path)
        print(f"Loaded embeddings: {emb.shape}")
        return emb
    raise FileNotFoundError("influencer_embeddings.npy not found!")

def load_features(path=os.path.join(DATA_DIR, "match_model.csv")):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded features: {df.shape}")
        return df
    raise FileNotFoundError("match_model.csv not found!")

def load_brand_embeddings(path=os.path.join(EMB_DIR, "brand_query_embeddings.npy")):
    if os.path.exists(path):
        emb = np.load(path)
        print(f"Loaded brand embeddings: {emb.shape}")
        return emb
    print("⚠ brand_embeddings.npy not found, using first influencer embedding as dummy")
    return None


# ------------------------------------------------------------
# DATA INITIALIZATION
# ------------------------------------------------------------
influencer_embeddings = load_embeddings()
features_df = load_features()
features_df = features_df.reset_index(drop=True)
features_df["emb_index"] = features_df.index


brand_embeddings = load_brand_embeddings()
# if brand_embeddings is None:
#     brand_embeddings = influencer_embeddings[0].reshape(1, -1)

brand_embedding = brand_embeddings[0]


# ------------------------------------------------------------
# MODEL TRAINING HELPERS
# ------------------------------------------------------------
def train_knn(embeddings, n_neighbors=500):
    n_neighbors = min(n_neighbors, embeddings.shape[0])
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(embeddings)
    joblib.dump(knn, "models/knn_model.pkl")
    print("k-NN Model saved → knn_model.pkl")
    return knn


def get_candidates_for_brand(brand_embedding, knn_model, influencer_embeddings, features_df, top_k=500):
    brand_emb = np.array(brand_embedding).reshape(1, -1)
    distances, indices = knn_model.kneighbors(brand_emb, n_neighbors=min(top_k, influencer_embeddings.shape[0]))
    similarities = 1 - distances.flatten()
    idx = indices.flatten()
    candidates = features_df.loc[idx].copy().reset_index(drop=True)
    candidates["similarity_score"] = similarities
    candidates["emb_index"] = idx
    return candidates


def get_compatibility_scores_for_candidates(candidates_df):
    if "compatibility_score" in candidates_df.columns:
        return candidates_df["compatibility_score"].values
    print("⚠ No compatibility score found — defaulting to 0")
    return np.zeros(len(candidates_df))


def prepare_ranker_features(df_candidates):
    df = df_candidates.copy()
    numeric = ["follower_count", "engagement_rate", "similarity_score", "compatibility_score"]
    numeric = [c for c in numeric if c in df.columns]

    cat = ["region", "platform"]
    cat = [c for c in cat if c in df.columns]

    df_num = df[numeric]
    if cat:
        df_cat = pd.get_dummies(df[cat], drop_first=True)
        X = pd.concat([df_num, df_cat], axis=1)
    else:
        X = df_num
    return X


def train_ranking_model(train_df):
    if "final_rank_score" not in train_df.columns:
        print("⚠ No final score → using proxy = 0.6*compat + 0.4*similarity")
        train_df["final_rank_score"] = (
            0.6 * train_df["compatibility_score"] + 0.4 * train_df["similarity_score"]
        )
    X = prepare_ranker_features(train_df)
    y = train_df["final_rank_score"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"R2: {r2_score(y_test, preds):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.3f}")
    bundle = dict(model=model, scaler=scaler, features=X.columns.tolist())
    joblib.dump(bundle, "models/ranking_model.pkl")
    print("Ranking model saved → ranking_model.pkl")
    return bundle


def recommend_for_brand(
    brand_embedding,
    knn_model,
    influencer_embeddings,
    rf_model,
    features_df = features_df,
    top_k=500,
    top_n=20,
):
    candidates = get_candidates_for_brand(
        brand_embedding,
        knn_model,
        influencer_embeddings,
        features_df,
        top_k=top_k,
    )
    candidates["compatibility_score"] = get_compatibility_scores_for_candidates(candidates)

    model = rf_model["model"]
    scaler = rf_model["scaler"]
    feature_cols = rf_model["features"]
    X = prepare_ranker_features(candidates)
    X = X.reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    candidates["final_match_score"] = preds * 100


    candidates = candidates.sort_values("final_match_score", ascending=False)
    candidates = candidates.reset_index(drop=True)
    candidates["rank"] = candidates.index + 1
    cols = ["rank"] + [col for col in candidates.columns if col != "rank"]
    candidates = candidates[cols]

    print("\nTop recommendations:")
    print(candidates.head(top_n).to_string(index=False))
    return candidates.head(top_n)


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
if __name__ == "__main__":
    # rf_model = train_ranking_model(candidates)

    try:
        knn_model = joblib.load("models/knn_model.pkl")
    except FileNotFoundError:
        knn_model = train_knn(influencer_embeddings)
        

    try:
        ranking_bundle = joblib.load("models/ranking_model.pkl")
    except FileNotFoundError:
        print("⚠ ranking_model.pkl not found, training new one...")
        candidates_for_training = get_candidates_for_brand(
            brand_embedding,
            knn_model,
            influencer_embeddings,
            features_df,
            top_k=500,
        )
        candidates_for_training["compatibility_score"] = get_compatibility_scores_for_candidates(
            candidates_for_training
        )
        ranking_bundle = train_ranking_model(candidates_for_training)
        print("Ranking model trained and loaded.")

    top_results = recommend_for_brand(
        brand_embedding,
        knn_model,
        influencer_embeddings,
        features_df=features_df,
        rf_model=ranking_bundle,
    )

    # rf_model = train_ranking_model(candidates.copy()) #train if model dosent exist

    # Save results
    top_results.to_csv("top_influencer_recommendations.csv", index=False)
    print("\n✅ Saved top recommendations → top_influencer_recommendations.csv")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.bar(top_results['influencer_name'].astype(str), top_results['final_match_score'])
    plt.xlabel('Influencer Name')
    plt.ylabel('Final Match Score')
    plt.title('Top 20 Influencer Recommendations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()