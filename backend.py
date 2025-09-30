# backend.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
def load_data(path="parkinsons.csv"):
    return pd.read_csv(path)

# Split dataset
def split_data(df, test_size=0.2):
    X = df.drop(columns=["status", "name"], errors="ignore")
    y = df["status"]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# Scale features
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

# Apply PCA
def apply_pca(X_train_scaled, X_test_scaled, n_components=10):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X_train_scaled), pca.transform(X_test_scaled)

# Train models
def train_models(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    svm = SVC(kernel="rbf", random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    return rf, svm, mlp

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm

# Feature importance
def feature_importance(model, feature_names):
    import pandas as pd
    importance = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    return importance.sort_values("Importance", ascending=False)
