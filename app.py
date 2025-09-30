import streamlit as st
import pandas as pd
from backend import load_data, split_data, preprocess_data, apply_pca, train_models, evaluate_model, feature_importance

st.title("Parkinson's Disease Detection")

# Load dataset
data = load_data()
st.write("Dataset preview:")
st.dataframe(data.head())

# Split & preprocess
X_train, X_test, y_train, y_test = split_data(data)
X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

# Train models
rf, svm, mlp = train_models(X_train_scaled, y_train)

# Evaluate models
st.subheader("Model Evaluation")
acc_rf = evaluate_model(rf, X_test_scaled, y_test, "Random Forest")
acc_svm = evaluate_model(svm, X_test_scaled, y_test, "SVM")
acc_mlp = evaluate_model(mlp, X_test_scaled, y_test, "MLP Neural Network")

# PCA analysis
X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled)
rf_pca, svm_pca, mlp_pca = train_models(X_train_pca, y_train)
st.subheader("Model Evaluation with PCA")
evaluate_model(rf_pca, X_test_pca, y_test, "Random Forest + PCA")
evaluate_model(svm_pca, X_test_pca, y_test, "SVM + PCA")
evaluate_model(mlp_pca, X_test_pca, y_test, "MLP + PCA")

# Feature importance
st.subheader("Top Features")
feature_importance(rf)
