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
def load_data(path="parkinsons.data"):
    df = pd.read_csv(path)
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    return df

# ====== Preprocess Data ======
def split_data(df):
    X = df.drop(columns=["status"])
    y = df["status"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# ====== PCA ======
def apply_pca(X_train_scaled, X_test_scaled, n_components=10):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, pca

# ====== Train Models ======
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier
