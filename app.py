# app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from backend import load_data, split_data, preprocess_data, apply_pca, train_models, evaluate_model, feature_importance

st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")
sns.set_palette("viridis")

st.title("Parkinson's Disease Classification")

# ====== Load Dataset ======
st.write("Loading dataset...")
data = load_data()
st.write("Dataset preview:")
st.dataframe(data.head())

# ====== Train-Test Split ======
X_train, X_test, y_train, y_test = split_data(data)
X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

# ====== Train Models ======
rf, svm, mlp = train_models(X_train_scaled, y_train)

# ====== Evaluate Models Without PCA ======
st.subheader("Model Evaluation without PCA")
for model, name in zip([rf, svm, mlp], ["Random Forest", "SVM", "MLP"]):
    acc, report, cm = evaluate_model(model, X_test_scaled, y_test)
    st.write(f"**{name} Accuracy:** {acc:.3f}")
    st.text(report)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Healthy', 'Parkinson'],
                yticklabels=['Healthy', 'Parkinson'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ====== Evaluate Models With PCA ======
X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
rf_pca, svm_pca, mlp_pca = train_models(X_train_pca, y_train)

st.subheader("Model Evaluation with PCA")
for model, name in zip([rf_pca, svm_pca, mlp_pca], ["Random Forest + PCA", "SVM + PCA", "MLP + PCA"]):
    acc, report, cm = evaluate_model(model, X_test_pca, y_test)
    st.write(f"**{name} Accuracy:** {acc:.3f}")
    st.text(report)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Healthy', 'Parkinson'],
                yticklabels=['Healthy', 'Parkinson'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ====== Feature Importance ======
st.subheader("Top Features (Random Forest)")
top_features = feature_importance(rf, X_train.columns)
st.dataframe(top_features)
