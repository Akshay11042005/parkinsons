# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from backend import load_data, split_data, preprocess_data, apply_pca, train_models, evaluate_model, feature_importance

st.title("🧠 Parkinson's Disease Detection")
st.write("Upload dataset, train models, and compare results with/without PCA.")

# ====== Load Data ======
uploaded_file = st.file_uploader("Upload Parkinson's Dataset (.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset")
    df = load_data("parkinsons.data")

st.write("### Dataset Preview")
st.dataframe(df.head())

# ====== Split and Preprocess ======
X_train, X_test, y_train, y_test = split_data(df)
X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

# ====== Train Models Without PCA ======
st.subheader("🔹 Training Without PCA")
models = train_models(X_train_scaled, y_train)

results = {}
for name, model in models.items():
    acc, report, cm = evaluate_model(model, X_test_scaled, y_test)
    results[name] = acc

    st.write(f"**{name}** Accuracy: {acc:.3f}")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Healthy','Parkinson'], 
                yticklabels=['Healthy','Parkinson'], ax=ax)
    st.pyplot(fig)

# ====== Train Models With PCA ======
st.subheader("🔹 Training With PCA (10 components)")
X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
models_pca = train_models(X_train_pca, y_train)

for name, model in models_pca.items():
    acc, report, cm = evaluate_model(model, X_test_pca, y_test)
    results[f"{name}+PCA"] = acc

    st.write(f"**{name}+PCA** Accuracy: {acc:.3f}")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=['Healthy','Parkinson'], 
                yticklabels=['Healthy','Parkinson'], ax=ax)
    st.pyplot(fig)

# ====== Accuracy Comparison ======
st.subheader("📊 Accuracy Comparison")
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
st.bar_chart(results_df.set_index("Model"))

# ====== Feature Importance ======
st.subheader("⭐ Feature Importance (Random Forest)")
rf_model = models["Random Forest"]
feat_imp = feature_importance(rf_model, X_train)
if feat_imp is not None:
    st.dataframe(feat_imp.head(10))
    fig, ax = plt.subplots()
    top_10 = feat_imp.head(10)
    ax.barh(top_10["Feature"], top_10["Importance"], color="green")
    ax.set_title("Top 10 Important Features")
    plt.gca().invert_yaxis()
    st.pyplot(fig)
