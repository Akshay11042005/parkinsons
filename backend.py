# backend.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data(file):
    """Load dataset from uploaded file or path"""
    df = pd.read_csv(file)
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    X = df.drop(columns=["status"])
    y = df["status"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def preprocess_data(X_train, X_test):
    """Scale features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train RF, SVM, MLP models and return results"""
    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    svm = SVC(kernel="rbf", random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

    models = {"Random Forest": rf, "SVM": svm, "MLP": mlp}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "y_pred": y_pred
        }
    return results

def apply_pca(X_train_scaled, X_test_scaled, n_components=10):
    """Apply PCA and return transformed data and explained variance"""
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    explained_var = np.sum(pca.explained_variance_ratio_)
    return X_train_pca, X_test_pca, pca, explained_var

def feature_importance(rf_model, feature_names):
    """Return feature importance dataframe"""
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    return importance
