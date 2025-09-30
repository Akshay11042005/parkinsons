# backend.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ====== Load Dataset ======
def load_data(path="parkinsons.csv"):
    """Load dataset from CSV in repo root"""
    return pd.read_csv(path)

# ====== Preprocessing ======
def split_data(df):
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    X = df.drop(columns=["status"])
    y = df["status"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ====== PCA ======
def apply_pca(X_train, X_test, n_components=10):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

# ====== Model Training ======
def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    svm = SVC(kernel="rbf", random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    return rf, svm, mlp

# ====== Model Evaluation ======
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm

# ====== Feature Importance ======
def feature_importance(model, feature_names, top_n=10):
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    return importance.head(top_n)
