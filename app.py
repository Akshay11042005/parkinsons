# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Import backend functions
from backend import (
    load_data,
    split_data,
    preprocess_data,
    apply_pca,
    train_models,
    evaluate_model,
    feature_importance
)

# ------------------- Streamlit UI -------------------

st.title("🧠 Parkinson's Disease Detection")

# Load dataset
st.sidebar.header("Data")
data = load_data()
st.write("### Dataset Preview")
st.dataframe(data.head())

# Preprocessing
X_train, X_test, y_train, y_test = split_data(data)
X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

# PCA option
if st.sidebar.checkbox("Apply PCA"):
    X_train_scaled, X_test_scaled = apply_pca(X_train_scaled, X_test_scaled)

# Train models
models = train_models(X_train_scaled, y_train)

# Evaluate models
results = evaluate_model(models, X_test_scaled, y_test)

st.write("### Model Performance")
st.dataframe(results)

# Feature importance (for tree models)
if st.sidebar.checkbox("Show Feature Importance"):
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_importance(models["RandomForest"], data.drop("status", axis=1).columns, ax)
    st.pyplot(fig)
