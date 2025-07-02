import numpy as np
import torch
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load features and labels
X_bert = torch.load("distilbert_embeddings.pt").numpy()
X_meta = np.load("meta_features.npy")
y = np.load("labels.npy")
X = np.hstack([X_bert, X_meta])

# Train/test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Imbalance ratio
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"⚖️ scale_pos_weight: {scale_pos_weight:.2f}")

# Objective function for Optuna
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "n_estimators": 500,
        "verbosity": 0,
        "tree_method": "gpu_hist"  # Enable GPU acceleration
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return f1_score(y_valid, preds)

# Run Optuna study
print("🚀 Starting Optuna hyperparameter search...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=3600)

print("✅ Best trial:")
print(study.best_trial)
