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

st.title("🧠 Parkinson's Disease Detection App")

# Load data
df = load_data()
st.write("### Dataset", df.head())

# Split and preprocess
X_train, X_test, y_train, y_test = split_data(df)
X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

# Train models
models, accuracies = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

# Show accuracies
st.subheader("Model Accuracies")
st.write(pd.DataFrame(accuracies, columns=["Model", "Accuracy"]))

# PCA
X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
st.write(f"Explained variance (10 components): {pca.explained_variance_ratio_.sum():.3f}")

# Feature importance
feat_imp = feature_importance(X_train, y_train)
st.subheader("Top Features")
st.bar_chart(feat_imp.set_index("Feature").head(10))
