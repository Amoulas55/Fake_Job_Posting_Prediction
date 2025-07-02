import numpy as np
import torch
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc
)
from sklearn.model_selection import train_test_split

# Load features
X_bert = torch.load("distilbert_embeddings.pt").numpy()
X_meta = np.load("meta_features.npy")
y = np.load("labels.npy")
X = np.hstack([X_bert, X_meta])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Best hyperparameters from Optuna
best_params = {
    'learning_rate': 0.2074360635720522,
    'max_depth': 6,
    'min_child_weight': 6,
    'subsample': 0.7466109830843792,
    'colsample_bytree': 0.7382599720416346,
    'lambda': 0.6757349336831314,
    'alpha': 0.5319278483675299,
    'n_estimators': 500,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train),
    'tree_method': 'gpu_hist',
    'random_state': 42,
    'verbosity': 0
}

print("🚀 Training final XGBoost model with best hyperparameters...")
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Predict
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.5).astype(int)

# Evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap='Blues')
plt.title("Final Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("final_confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Final Model - ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("final_roc_curve.png")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Final Model - Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("final_precision_recall_curve.png")
plt.close()

# Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=20, importance_type="gain")
plt.title("Final Model - Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("final_feature_importance.png")
plt.close()

print("✅ Final model trained and evaluated. Plots saved.")
