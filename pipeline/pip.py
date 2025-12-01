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
import os
import joblib
import numpy as np
import pandas as pd
from pipeline.integrator.member_6_edited import BrandInfluencerRecommender


"""
Small loader utility for the project's recommendation pipeline.

Responsibilities:
- Verify required model and artifact files exist on disk.
- Load persisted models, embeddings and feature CSVs.
- Instantiate the `BrandInfluencerRecommender` and return top recommendations.

This file is intended to be a lightweight glue script used by the API
to load precomputed artifacts and obtain recommendations.
"""


def check_for_models():
    """Ensure all required model/artifact files exist.

    Raises:
        FileNotFoundError: if any expected file is missing.
    """
    # Check KNN nearest-neighbors model used for retrieval
    if not os.path.exists("models/knn_model.pkl"):
        raise FileNotFoundError('KNN Model not found at "models/knn_model.pkl"')
    # Check ranking/regression model used to score candidates
    if not os.path.exists("models/ranking_model.pkl"):
        raise FileNotFoundError('Ranking model not found at "models/ranking_model.pkl"')
    # Check precomputed embeddings for brands and influencers
    if not os.path.exists("embeddings/brand_embeddings.npy"):
        raise FileNotFoundError('Brand embeddings not found at "embeddings/brand_embeddings.npy"')
    if not os.path.exists("embeddings/influencer_embeddings.npy"):
        raise FileNotFoundError('Influencer embeddings not found at "embeddings/influencer_embeddings.npy"')
    # Features CSV used to join / enrich candidate rows
    if not os.path.exists("content/match_model.csv"):
        raise FileNotFoundError('Match model CSV not found at "content/match_model.csv"')


def load_models():
    """Load models, embeddings and feature data, instantiate recommender.

    Returns:
        str: JSON string containing recommendation records.

    Notes:
    - Any exception during loading is printed and will leave variables
      undefined; this file expects the surrounding environment (API)
      to ensure artifacts exist via `check_for_models()`.
    """
    check_for_models()

    try:
        # Load persisted models
        knn_model = joblib.load('models/knn_model.pkl')
        ranking_model = joblib.load('models/ranking_model.pkl')

        # Load precomputed embeddings (NumPy arrays)
        brand_ebeddings = np.load('embeddings/brand_embeddings.npy')
        influencer_ebeddings = np.load('embeddings/influencer_embeddings.npy')

        # Load feature table used to enrich candidate rows
        features_df = pd.read_csv('content/match_model.csv')
    except Exception as e:
        # Surface a clear warning; callers may choose to handle this.
        print("⚠ Loading failed:", e)
        raise

    # Instantiate the pipeline recommender with the loaded artifacts
    recommender = BrandInfluencerRecommender(
        influencer_embeddings=influencer_ebeddings,
        brand_embedding=brand_ebeddings,
        features_df=features_df,
        knn_model=knn_model,
        rf_model=ranking_model,
    )

    # Run recommendation and return as JSON records
    top_results = recommender.recommend()
    print('Recommendations done')
    return top_results.to_json(orient="records")


if __name__ == "__main__":
    # When executed directly, validate artifacts and produce results
    check_for_models()
    load_models()