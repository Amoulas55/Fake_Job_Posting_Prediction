# 🧠 Fake Job Posting Prediction

This project tackles the challenge of identifying **fraudulent job postings** using **Natural Language Processing (NLP)** and **machine learning**. It walks through the **entire ML pipeline**, from raw data and exploratory analysis to advanced modeling with **DistilBERT** and **XGBoost**.

---

## 🔍 Objective

Detect whether a job listing is **real** or **fake**, based on its title, description, requirements, and metadata.

---

## 📁 Project Structure

```bash
.
├── eda.py                        # Exploratory Data Analysis
├── preprocess.py                # Preprocessing & feature creation
├── extract_bert_embeddings.py   # DistilBERT embeddings (batched + Focal Loss)
├── train_logistic_baseline.py   # Simple baseline: logistic regression
├── train_wide_deep_focal.py     # Wide & Deep model (PyTorch + Focal loss)
├── train_xgboost_classifier.py  # Initial XGBoost model
├── xgb_optuna_tuning.py         # Hyperparameter tuning with Optuna
├── train_xgboost_final.py       # Final XGBoost model with tuned params
├── sanity_check_final_model.py  # Sanity tests to verify everything
├── *.npy / *.npz                # Saved feature arrays
├── *.png                        # Evaluation plots
└── *.joblib                     # Saved models



## 🛠️ Techniques Used
📊 EDA: Class imbalance, text length, top words

🧼 Preprocessing: Cleaning text, TF-IDF, metadata

🔡 Embeddings: DistilBERT with Huggingface Transformers

⚖️ Imbalance Handling: Focal loss + XGBoost scale_pos_weight

🧠 Modeling:

Logistic Regression (baseline)

Wide & Deep (custom PyTorch)

XGBoost with GPU acceleration

📈 Evaluation:

Confusion matrix

Precision / Recall / F1

ROC & PR curves

🧪 Sanity Checks: Ensures data integrity, class balance, consistency

## 📊 Final Model Performance (XGBoost + Optuna)
Metric	Value
Accuracy	98.18%
F1-score (Fake)	78.83%
Recall (Fake)	69.94%
Precision (Fake)	90.30%

📁 All evaluation plots saved locally (confusion matrix, ROC, PR curve, etc.)

## 💡 Why This Matters
Fake job listings are a rising threat to job seekers. This project uses advanced ML and NLP tools to help automate fake job detection, and demonstrates a real-world data science workflow with interpretability and sanity checks.

## 👤 Author
Angelos Moulas
MSc in Data Science & Society
