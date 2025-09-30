# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from backend import load_data, preprocess_data, split_data, scale_data, apply_pca, train_models, evaluate_model, feature_importance

st.set_page_config(page_title="Parkinson's Detection", layout="wide")
st.title("Parkinson's Disease Detection App")

# ====== Load Dataset ======
data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data.head())

# ====== Preprocess ======
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# ====== Train Models ======
models = train_models(X_train_scaled, y_train)

st.subheader("Model Accuracy (Without PCA)")
results = {}
for name, model in models.items():
    acc, report, cm = evaluate_model(model, X_test_scaled, y_test)
    results[name] = acc
    st.write(f"**{name}** Accuracy: {acc:.3f}")
    st.text(report)
    # Confusion matrix plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Healthy','Parkinson'],
                yticklabels=['Healthy','Parkinson'])
    plt.title(f"{name} Confusion Matrix")
    st.pyplot(plt)
    plt.clf()

# ====== PCA Section ======
st.subheader("Model Accuracy (With PCA)")
X_train_pca, X_test_pca, pca_model = apply_pca(X_train_scaled, X_test_scaled, n_components=10)
models_pca = train_models(X_train_pca, y_train)

results_pca = {}
for name, model in models_pca.items():
    acc, report, cm = evaluate_model(model, X_test_pca, y_test)
    results_pca[name+" + PCA"] = acc
    st.write(f"**{name} + PCA** Accuracy: {acc:.3f}")

# ====== Feature Importance ======
st.subheader("Top 10 Features Importance (Random Forest)")
fi_df = feature_importance(models["Random Forest"], X.columns)
if fi_df is not None:
    st.dataframe(fi_df.head(10))
    plt.figure(figsize=(8,6))
    sns.barplot(x="Importance", y="Feature", data=fi_df.head(10))
    plt.title("Top 10 Features for Parkinson's Detection")
    st.pyplot(plt)
