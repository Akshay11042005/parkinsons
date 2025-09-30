# app.py
import streamlit as st
from backend import load_data, split_data, preprocess_data, apply_pca, train_models, evaluate_model, feature_importance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")
st.title("Parkinson's Disease Classification App")

# Load dataset
data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Split and preprocess
X_train, X_test, y_train, y_test = split_data(data)
X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

# Train models
rf, svm, mlp = train_models(X_train_scaled, y_train)

# Evaluate Random Forest
acc_rf, report_rf, cm_rf = evaluate_model(rf, X_test_scaled, y_test)
st.subheader("Random Forest Results")
st.write(f"Accuracy: {acc_rf:.3f}")
st.text(report_rf)
st.write("Confusion Matrix:")
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
st.pyplot(plt)

# Feature Importance
fi = feature_importance(rf, X_train.columns)
st.subheader("Top 10 Features by Importance")
st.dataframe(fi.head(10))
