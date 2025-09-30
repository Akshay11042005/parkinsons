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
    """Load the dataset from CSV in repo root"""
    return pd.read_csv(path)

# ====== Preprocess Data ======
def preprocess_data(df):
    """Drop non-numeric columns, separate features and target"""
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    X = df.drop(columns=["status"])
    y = df["status"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Train-test split"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_data(X_train, X_test):
    """Standardize features"""
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

# ====== Models ======
def train_models(X_train, y_train):
    """Initialize and train models"""
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
        "SVM": SVC(kernel="rbf", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# ====== Evaluation ======
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm

# ====== Feature Importance ======
def feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        return fi_df
    else:
        return None
