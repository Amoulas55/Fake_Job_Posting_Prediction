🧠 Fake Job Posting Prediction
This project tackles the challenge of identifying fraudulent job postings using Natural Language Processing (NLP) and machine learning. It walks through the entire ML pipeline, from raw data and exploratory analysis to advanced modeling with DistilBERT and XGBoost. The solution is designed with scalability and transparency in mind, aligning with consulting best practices to ensure adaptability across multiple client contexts.

🔍 Objective
Detect whether a job listing is real or fake, based on its title, description, requirements, and metadata. The ultimate goal is to reduce exposure to employment scams and enable trust in digital job marketplaces.

## 🛠️ Techniques Used

📊 **EDA**: Investigated class imbalance, text lengths, common keywords, and linguistic patterns associated with fake listings. Insights from this stage guided preprocessing and feature selection.

🧼 **Preprocessing**: Applied robust text normalization, tokenization, TF-IDF vectorization, and structured metadata integration to create informative, model-ready features.

🛁 **Embeddings**: Integrated semantic representations using DistilBERT from Huggingface Transformers, capturing the contextual richness of job descriptions and requirements.

⚖️ **Imbalance Handling**: Applied Focal Loss in deep learning models and calibrated `scale_pos_weight` in XGBoost to address skewed class distribution, crucial for high precision in detecting fraud.

🧠 **Modeling Approaches**:

* **Logistic Regression**: Interpretable baseline to establish performance lower-bound.
* **Wide & Deep Model**: Combines deep feature interactions and raw semantic understanding; implemented in PyTorch with Focal Loss.
* **XGBoost (GPU)**: High-performance gradient boosting with Optuna-based hyperparameter optimization for robust, production-ready deployment.

📈 **Evaluation**:

* Quantitative: Accuracy, Precision, Recall, F1 Score, ROC-AUC, and PR-AUC.
* Qualitative: Confusion matrix analysis and business-case validation.

🧪 **Sanity Checks**: Comprehensive consistency validations were implemented, including checks on data integrity, class balance after processing, and prediction drift across folds.

## 📊 Final Model Performance (XGBoost + Optuna)

| Metric           | Value  |
| ---------------- | ------ |
| Accuracy         | 98.18% |
| F1-score (Fake)  | 78.83% |
| Recall (Fake)    | 69.94% |
| Precision (Fake) | 90.30% |

All evaluation artifacts, including ROC and PR curves, are stored to support client-facing reporting and model documentation.

## 💡 Why This Matters

Fake job listings are a growing cyber risk. They exploit trust and personal data, creating real-world harm. By applying enterprise-grade machine learning techniques and explainable NLP workflows, this project provides a scalable and auditable solution. It is ready to be integrated into fraud detection pipelines or job platforms as part of broader digital trust strategies.

## 👤 Author

Angelos Moulas
MSc in Data Science & Society
