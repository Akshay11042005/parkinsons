# backend.py - Model and training logic
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

# ------------------- Load Dataset -------------------
def load_data(path="parkinsons.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found! Make sure it is in the repo root.")
    return pd.read_csv(path)

# ------------------- Split Data -------------------
def split_data(df, target_col="status"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------- Preprocessing -------------------
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ------------------- Apply PCA -------------------
def apply_pca(X_train, X_test, n_components=10):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# ------------------- Train Models -------------------
def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=200),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

# ------------------- Evaluate Models -------------------
def evaluate_model(models, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc, "F1-Score": f1})
    return pd.DataFrame(results)

# ------------------- Feature Importance -------------------
def feature_importance(model, feature_names, ax=None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(importances)), importances[indices], align="center")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(np.array(feature_names)[indices], rotation=90)
        ax.set_title("Feature Importance")
    else:
        if ax:
            ax.text(0.5, 0.5, "Model has no feature_importances_", ha="center")
