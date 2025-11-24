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

class BrandInfluencerRecommender:
    def __init__(self, 
                  influencer_embeddings, features_df, brand_embedding, knn_model, rf_model,
                  data_dir="content", model_dir="models", emb_dir = 'embeddings'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.emb_dir = emb_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Load embeddings and features
        self.influencer_embeddings = influencer_embeddings

        self.features_df = features_df #self.load_features("match_model.csv")
        self.features_df = self.features_df.reset_index(drop=True)
        self.features_df["emb_index"] = self.features_df.index

        self.brand_embedding = brand_embedding

        self.knn_model = knn_model
        self.rf_model = rf_model

    # ----------------- Loaders -----------------
    def load_influencer_embeddings(self, filename="influencer_embeddings.npy"):
        path = os.path.join(self.emb_dir, filename)
        if os.path.exists(path):
            emb = np.load(path)
            print(f"Loaded influencer embeddings: {emb.shape}")
            return emb
        raise FileNotFoundError(f"{filename} not found!")

    def load_features(self, filename):
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded features: {df.shape}")
            return df
        raise FileNotFoundError(f"{filename} not found!")

    def load_brand_embeddings(self, filename="brand_embeddings.npy"):
        path = os.path.join(self.emb_dir, filename)
        if os.path.exists(path):
            emb = np.load(path)
            print(f"Loaded embeddings: {emb.shape}")
            return emb
        raise FileNotFoundError(f"{filename} not found!")

    # ----------------- k-NN -----------------
    def train_knn(self, n_neighbors=500):
        n_neighbors = min(n_neighbors, self.influencer_embeddings.shape[0])
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.knn_model.fit(self.influencer_embeddings)
        joblib.dump(self.knn_model, os.path.join(self.model_dir, "knn_model.pkl"))
        print("k-NN Model saved → knn_model.pkl")

    def load_knn(self):
        path = os.path.join(self.model_dir, "knn_model.pkl")
        if os.path.exists(path):
            self.knn_model = joblib.load(path)
            print("k-NN model loaded")
        else:
            self.train_knn()

    # ----------------- Ranking Model -----------------
    def prepare_ranker_features(self, df_candidates):
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

    def train_ranking_model(self, candidates_df):
        if "final_rank_score" not in candidates_df.columns:
            candidates_df["final_rank_score"] = 0.6 * candidates_df.get("compatibility_score", 0) + \
                                                0.4 * candidates_df.get("similarity_score", 0)
        X = self.prepare_ranker_features(candidates_df)
        y = candidates_df["final_rank_score"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"R2: {r2_score(y_test, preds):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f}, MAE: {mean_absolute_error(y_test, preds):.3f}")
        self.rf_model = {"model": model, "scaler": scaler, "features": X.columns.tolist()}
        joblib.dump(self.rf_model, os.path.join(self.model_dir, "ranking_model.pkl"))
        print("Ranking model saved → ranking_model.pkl")

    def load_ranking_model(self):
        path = os.path.join(self.model_dir, "ranking_model.pkl")
        if os.path.exists(path):
            self.rf_model = joblib.load(path)
            print("Ranking model loaded")
        else:
            # Generate candidates from first influencer if no model
            sample_candidates = self.get_candidates_for_brand(self.brand_embedding)
            sample_candidates["compatibility_score"] = self.get_compatibility_scores(sample_candidates)
            self.train_ranking_model(sample_candidates)

    # ----------------- Candidate selection -----------------
    def get_candidates_for_brand(self, brand_embedding=None, top_k=500):
        brand_emb = brand_embedding if brand_embedding is not None else self.brand_embedding
        distances, indices = self.knn_model.kneighbors(brand_emb, n_neighbors=min(top_k, self.influencer_embeddings.shape[0]))
        similarities = 1 - distances.flatten()
        idx = indices.flatten()
        candidates = self.features_df.loc[idx].copy().reset_index(drop=True)
        candidates["similarity_score"] = similarities
        candidates["emb_index"] = idx
        return candidates

    def get_compatibility_scores(self, candidates_df):
        if "compatibility_score" in candidates_df.columns:
            return candidates_df["compatibility_score"].values
        return np.zeros(len(candidates_df))

    # ----------------- Recommendation -----------------
    def recommend(self, top_k=500, top_n=20):
        candidates = self.get_candidates_for_brand(top_k=top_k)
        candidates["compatibility_score"] = self.get_compatibility_scores(candidates)

        # Predict
        X = self.prepare_ranker_features(candidates)
        X = X.reindex(columns=self.rf_model["features"], fill_value=0)
        X_scaled = self.rf_model["scaler"].transform(X)
        candidates["final_match_score"] = self.rf_model["model"].predict(X_scaled) * 100

        candidates = candidates.sort_values("final_match_score", ascending=False).reset_index(drop=True)
        candidates["rank"] = candidates.index + 1
        cols = ["rank"] + [c for c in candidates.columns if c != "rank"]
        candidates = candidates[cols]

        return candidates.head(top_n)

    # ----------------- Plot -----------------
    def plot_recommendations(self, recommendations):
        plt.figure(figsize=(12, 6))
        plt.bar(recommendations['influencer_name'].astype(str), recommendations['final_match_score'])
        plt.xlabel('Influencer Name')
        plt.ylabel('Final Match Score')
        plt.title('Top Influencer Recommendations')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
