🧠 Fake Job Posting Prediction
This project tackles the challenge of identifying fraudulent job postings using Natural Language Processing (NLP) and machine learning. It walks through the entire ML pipeline, from raw data and exploratory analysis to advanced modeling with DistilBERT and XGBoost.

🔍 Objective
Detect whether a job listing is real or fake, based on its title, description, requirements, and metadata.

📁 Project Structure
.
├── fake\_job\_postings.csv        # Original dataset (from Kaggle)
├── eda.py                       # Exploratory Data Analysis
├── preprocess.py                # Preprocessing & feature creation
├── extract\_bert\_embeddings.py   # DistilBERT embeddings (batched + Focal Loss)
├── train\_logistic\_baseline.py   # Simple baseline: logistic regression
├── train\_wide\_deep\_focal.py     # Wide & Deep model (PyTorch + Focal loss)
├── train\_xgboost\_classifier.py  # Initial XGBoost model
├── xgb\_optuna\_tuning.py         # Hyperparameter tuning with Optuna
├── train\_xgboost\_final.py       # Final XGBoost model with tuned params
├── sanity\_check\_final\_model.py  # Sanity tests to verify everything
├── \*.npy / \*.npz                # Saved feature arrays
├── \*.png                        # Evaluation plots
└── \*.joblib                     # Saved models

## 🛠️ Techniques Used

📊 EDA: Class imbalance, text length, top words

🧼 Preprocessing: Cleaning text, TF-IDF, metadata

🛁 Embeddings: DistilBERT with Huggingface Transformers

⚖️ Imbalance Handling: Focal loss + XGBoost scale\_pos\_weight

🧠 Modeling:

* Logistic Regression (baseline)
* Wide & Deep (custom PyTorch)
* XGBoost with GPU acceleration

📈 Evaluation:

* Confusion matrix
* Precision / Recall / F1
* ROC & PR curves

🧪 Sanity Checks: Ensures data integrity, class balance, consistency

## 📊 Final Model Performance (XGBoost + Optuna)

| Metric           | Value  |
| ---------------- | ------ |
| Accuracy         | 98.18% |
| F1-score (Fake)  | 78.83% |
| Recall (Fake)    | 69.94% |
| Precision (Fake) | 90.30% |

All evaluation plots saved locally (confusion matrix, ROC, PR curve, etc.)

## 💡 Why This Matters

Fake job listings are a rising threat to job seekers. This project uses advanced ML and NLP tools to help automate fake job detection, and demonstrates a real-world data science workflow with interpretability and sanity checks.

## 👤 Author

Angelos Moulas
MSc in Data Science & Society
