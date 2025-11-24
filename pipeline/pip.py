import os
import joblib
import numpy as np
import pandas as pd
from pipeline.integrator.member_6_edited import BrandInfluencerRecommender

def check_for_models():
    if not os.path.exists("models/knn_model.pkl"):
        raise FileNotFoundError('KNN Model not found in path.')
    if not os.path.exists("models/ranking_model.pkl"):
        raise FileNotFoundError('Ranking Model not found in path.')
    if not os.path.exists("embeddings/brand_embeddings.npy"):
        raise FileNotFoundError('Brand embedding not found in path.')
    if not os.path.exists("embeddings/influencer_embeddings.npy"):
        raise FileNotFoundError('Influencer embedding not found in path.')
    if not os.path.exists("content/match_model.csv"):
        raise FileNotFoundError('Match model csv not found in path.')


def load_models():
    check_for_models()
    try:
        knn_model = joblib.load('models/knn_model.pkl')
        ranking_model = joblib.load('models/ranking_model.pkl')

        brand_ebeddings = np.load('embeddings/brand_embeddings.npy')
        influencer_ebeddings = np.load('embeddings/influencer_embeddings.npy')

        features_df = pd.read_csv('content/match_model.csv')
    except Exception as e:
        print("⚠ Loading failed:", e)


    recommender = BrandInfluencerRecommender(
        influencer_embeddings=influencer_ebeddings,
        brand_embedding=brand_ebeddings,

        features_df=features_df,

        knn_model=knn_model,
        rf_model=ranking_model
    )


    top_results = recommender.recommend()
    print('Recommendations done')
    return top_results.to_json(orient="records")
    
if __name__ == "__main__":
    check_for_models()
    load_models()