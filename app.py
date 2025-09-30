# app.py

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
    st.pyplot(fig)
