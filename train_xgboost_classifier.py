import numpy as np
import torch
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split

# Load features
X_bert = torch.load("distilbert_embeddings.pt").numpy()
X_meta = np.load("meta_features.npy")
y = np.load("labels.npy")

X = np.hstack([X_bert, X_meta])
print(f"✅ Features loaded - X shape: {X.shape}, y shape: {y.shape}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Handle class imbalance
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"⚖️ scale_pos_weight: {scale_pos_weight:.2f}")

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

print("🏋️ Training XGBoost model...")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)

# Predict
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.5).astype(int)

# Evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("xgb_roc_curve.png")
plt.close()

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("xgb_precision_recall_curve.png")
plt.close()

# Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=20, importance_type="gain")
plt.title("XGBoost Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png")
plt.close()

print("✅ Training complete. Plots saved:")
print("- xgb_confusion_matrix.png")
print("- xgb_roc_curve.png")
print("- xgb_precision_recall_curve.png")
print("- xgb_feature_importance.png")
