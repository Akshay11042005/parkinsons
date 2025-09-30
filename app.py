# app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from backend import load_data, split_data, preprocess_data, train_models, apply_pca, feature_importance

st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")
st.title("Parkinson's Disease Classification")

# Upload dataset
uploaded_file = st.file_uploader("Upload Parkinson's dataset (.csv)", type="csv")

if uploaded_file:
    X, y = load_data(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(X.head())

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    st.subheader("Model Training Without PCA")
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    for name, res in results.items():
        st.write(f"**{name} Accuracy:** {res['accuracy']:.3f}")
        cm = confusion_matrix(y_test, res['y_pred'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=['Healthy','Parkinson'], yticklabels=['Healthy','Parkinson'])
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.subheader("Model Training With PCA (10 components)")
    X_train_pca, X_test_pca, pca, var = apply_pca(X_train_scaled, X_test_scaled)
    st.write(f"Explained Variance by 10 components: {var:.3f}")

    results_pca = train_models(X_train_pca, X_test_pca, y_train, y_test)
    for name, res in results_pca.items():
        st.write(f"**{name} + PCA Accuracy:** {res['accuracy']:.3f}")

    st.subheader("Feature Importance (Random Forest)")
    rf_model = results['Random Forest']['model']
    importance_df = feature_importance(rf_model, X.columns)
    st.dataframe(importance_df.head(10))
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette="viridis", ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)
