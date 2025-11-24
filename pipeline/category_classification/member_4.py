# ============================================================
# Tonelink - Member 4: Category Classification Model (Fixed)
# Trains a supervised model to classify influencer/brand categories.
# Outputs:
#   ✅ category_model.pkl
#   ✅ category_predictions.csv (true + predicted)
#   ✅ Confusion matrix plots saved as PNGs
# ============================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import itertools
import joblib

RANDOM_STATE = 42

# ------------------------- Helpers -------------------------
def smart_text_column(df: pd.DataFrame) -> pd.Series:
    candidates = [c for c in df.columns if c.lower() in
                  ['bio','description','brand_description','about','tagline','captions','caption','title','summary','campaign_summary']]
    if not candidates:
        candidates = [c for c in df.columns if df[c].dtype == 'object']
        candidates = candidates[:5]
    txt = df[candidates].astype(str).agg(' | '.join, axis=1) if candidates else pd.Series([""]*len(df))
    txt = txt.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    return txt

def try_load_embeddings(n_rows: int) -> Optional[np.ndarray]:
    if os.path.exists("embeddings/influencer_embeddings.npy"):
        arr = np.load("embeddings/influencer_embeddings.npy")
        if arr.shape[0] != n_rows:
            print("[WARN] embeddings.npy rows mismatch — ignoring.")
            return None
        print(f"[OK] Loaded embeddings.npy with shape {arr.shape}.")
        return arr.astype(np.float32)
    print("[INFO] No embeddings found, using TF-IDF.")
    return None

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', fname=None):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=250)
        plt.close()
    else:
        plt.show()

# ------------------------- Load Dataset -------------------------
if not os.path.exists("content/features.csv"):
    raise FileNotFoundError("features.csv not found inside /content/")

df = pd.read_csv("content/features.csv")
print(f"[OK] Loaded features.csv with {df.shape}")

if 'category' not in df.columns:
    raise ValueError("Expected 'category' column in features.csv")

df['__text__'] = smart_text_column(df)
df = df.dropna(subset=['category'])
y_raw = df['category'].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_.tolist()

emb = try_load_embeddings(len(df))

numeric_candidates = [c for c in df.columns if c.lower() in
                      ['engagement_rate','followers','follower_count']]
categorical_candidates = [c for c in df.columns if c.lower() in
                          ['region','platform','language','gender','age_group']]

for c in ['category','__text__']:
    numeric_candidates = [n for n in numeric_candidates if n != c]
    categorical_candidates = [n for n in categorical_candidates if n != c]

print(f"[INFO] Using numeric: {numeric_candidates}")
print(f"[INFO] Using categorical: {categorical_candidates}")

train_idx, val_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# ------------------------- Train -------------------------
if emb is not None:
    X_train, X_val = emb[train_idx], emb[val_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X_train_s, y_train)
    X_val_s = scaler.transform(X_val)
    y_pred = clf.predict(X_val_s)

    model_bundle = {
        "mode": "embeddings",
        "scaler": scaler,
        "clf": clf,
        "label_encoder": le,
        "expected_n_features": X_train.shape[1]
    }

else:
    text_col = "__text__"
    text_vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_features=100_000)
    pre = ColumnTransformer([("text", text_vec, text_col)], remainder="drop")
    clf = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(df_train, y_train)
    y_pred = pipe.predict(df_val)
    model_bundle = {
        "mode": "tfidf",
        "pipeline": pipe,
        "label_encoder": le,
        "text_col": text_col
    }

# ------------------------- Evaluate -------------------------
acc = accuracy_score(y_val, y_pred)
f1m = f1_score(y_val, y_pred, average='macro')
print(f"\n✅ Accuracy: {acc:.4f} | Macro F1: {f1m:.4f}\n")
print(classification_report(y_val, y_pred, digits=4, target_names=[classes[i] for i in np.unique(y_val)]))

cm = confusion_matrix(y_val, y_pred, labels=np.unique(y_val))
plot_confusion_matrix(cm, [classes[i] for i in np.unique(y_val)],
                      normalize=False, fname="confusion_counts.png")
plot_confusion_matrix(cm, [classes[i] for i in np.unique(y_val)],
                      normalize=True, fname="confusion_normalized.png")

# ------------------------- Save Model + CSV -------------------------
joblib.dump(model_bundle, "category_model.pkl")
print("[OK] Saved category_model.pkl")

df_out = df.copy()
df_out['true_category'] = y_raw
df_out['pred_category'] = np.nan
df_out.iloc[val_idx, df_out.columns.get_loc('pred_category')] = le.inverse_transform(y_pred)
df_out.to_csv("category_predictions.csv", index=False)
print("[OK] Saved predictions to category_predictions.csv")

print("\n🎯 Done — Model + CSV + confusion matrix images ready.")
