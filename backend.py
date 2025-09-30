import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path="parkinsons.csv"):
    return pd.read_csv(path)

def split_data(df):
    X = df.drop(columns=["status", "name"])
    y = df["status"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def apply_pca(X_train_scaled, X_test_scaled, n_components=10):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X_train_scaled), pca.transform(X_test_scaled)

def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    svm = SVC(kernel="rbf", random_state=42).fit(X_train, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42).fit(X_train, y_train)
    return rf, svm, mlp

def evaluate_model(model, X_test, y_test, title="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"{title} Accuracy: {acc:.3f}")
    st.text(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title} - Confusion Matrix")
    st.pyplot(fig)
    return acc

def feature_importance(rf):
    feature_names = rf.feature_names_in_
    importance = pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    st.dataframe(importance.head(10))
    fig, ax = plt.subplots()
    ax.barh(importance["Feature"].head(10), importance["Importance"].head(10), color="skyblue")
    ax.invert_yaxis()
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)
