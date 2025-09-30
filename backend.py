<<<<<<< HEAD
# backend.py
=======
<<<<<<< HEAD
>>>>>>> 8972463 (Ready-to-deploy Streamlit Parkinson's app with dataset)
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

<<<<<<< HEAD
# ====== Feature Importance ======
def feature_importance(model, feature_names, top_n=10):
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    return importance.head(top_n)
=======
def feature_importance(rf):
    feature_names = rf.feature_names_in_
    importance = pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    st.dataframe(importance.head(10))
    fig, ax = plt.subplots()
    ax.barh(importance["Feature"].head(10), importance["Importance"].head(10), color="skyblue")
    ax.invert_yaxis()
    ax.set_title("Top 10 Feature Importances")
=======
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from backend import (
    load_data,
    split_data,
    preprocess_data,
    apply_pca,
    train_models,
    evaluate_model,
    feature_importance
)

st.title("🧠 Parkinson's Disease Detection App")

# ------------------- Load Dataset -------------------
data = load_data()
st.write("### Dataset Preview")
st.dataframe(data.head())

# ------------------- Preprocessing -------------------
X_train, X_test, y_train, y_test = split_data(data)
X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

# PCA option
if st.sidebar.checkbox("Apply PCA"):
    X_train_scaled, X_test_scaled = apply_pca(X_train_scaled, X_test_scaled)

# ------------------- Train Models -------------------
models = train_models(X_train_scaled, y_train)

# ------------------- Evaluate Models -------------------
results = evaluate_model(models, X_test_scaled, y_test)
st.write("### Model Performance")
st.dataframe(results)

# ------------------- Download Results -------------------
csv = results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Results as CSV",
    data=csv,
    file_name="model_results.csv",
    mime="text/csv",
)

# ------------------- Download Trained Model -------------------
model_choice = st.selectbox("Select a model to download:", list(models.keys()))
if model_choice:
    # Save model as Pickle
    with open(f"{model_choice}_model.pkl", "wb") as f:
        pickle.dump(models[model_choice], f)
    # Download button
    with open(f"{model_choice}_model.pkl", "rb") as f:
        st.download_button(
            label=f"📥 Download {model_choice} Model (Pickle)",
            data=f,
            file_name=f"{model_choice}_model.pkl",
            mime="application/octet-stream",
        )

# ------------------- Feature Importance -------------------
if model_choice == "RandomForest" and st.sidebar.checkbox("Show Feature Importance"):
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_importance(models["RandomForest"], data.drop("status", axis=1).columns, ax)
>>>>>>> 9742231 (Initial commit: add app.py, backend.py, requirements.txt, parkinsons.csv)
    st.pyplot(fig)
>>>>>>> 8972463 (Ready-to-deploy Streamlit Parkinson's app with dataset)
