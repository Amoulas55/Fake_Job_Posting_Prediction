import numpy as np
import torch
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("🚀 Starting sanity checks...")

# === Load data ===
X_bert = torch.load("distilbert_embeddings.pt").numpy()
X_meta = np.load("meta_features.npy")
y = np.load("labels.npy")

X = np.hstack((X_bert, X_meta))
print(f"✅ Feature shape: {X.shape}")
print(f"✅ Label shape: {y.shape}")

# === Label distribution ===
unique, counts = np.unique(y, return_counts=True)
print(f"✅ Label counts: {dict(zip(unique, counts))}")

# === Check for NaNs or infs ===
if np.isnan(X).any() or np.isinf(X).any():
    raise ValueError("❌ Found NaNs or infs in X!")
else:
    print("✅ No NaNs or infs in features")

# === Sanity check value ranges ===
print(f"ℹ️ Feature value range: min {X.min():.3f}, max {X.max():.3f}")

# === Load final model ===
model_path = "xgboost_final_model.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Final model not found. Make sure 'xgboost_final_model.joblib' exists.")
model = joblib.load(model_path)

# === Re-split data to verify metrics ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("✅ Model loaded. Making predictions...")
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Confusion matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("✅ Sanity Check: Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("sanity_confusion_matrix.png")
print("📁 Saved confusion matrix to sanity_confusion_matrix.png")

print("✅ All sanity checks complete.")
