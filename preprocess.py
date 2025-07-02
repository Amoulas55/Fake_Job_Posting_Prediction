# preprocess.py
import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from scipy.sparse import hstack, save_npz
import joblib

nltk.download('stopwords')

# Load data
df = pd.read_csv('fake_job_postings.csv')
print("✅ Loaded dataset with shape:", df.shape)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop irrelevant or sparse columns
df.drop(columns=['job_id', 'logo', 'telecommuting', 'has_questions', 'salary_range'], errors='ignore', inplace=True)

# Fill missing values
text_cols = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    df[col] = df[col].fillna('')
cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

# Label encode categorical variables
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Binary encode target
df['fraudulent'] = df['fraudulent'].astype(int)

# Combine text fields
df['text_combined'] = (
    df['title'] + ' ' + df['company_profile'] + ' ' +
    df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']
)

# Clean and preprocess text
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['text_cleaned'] = df['text_combined'].apply(clean_text)

# Add custom features
df['desc_length'] = df['description'].apply(lambda x: len(x.split()))
df['req_length'] = df['requirements'].apply(lambda x: len(x.split()))
df['has_clickbait'] = df['text_combined'].str.contains(
    r'(?i)work from home|no experience|quick money|urgent'
).astype(int)

# Remove rare categories
for col in cat_cols:
    freq = df[col].value_counts()
    rare = freq[freq < 10].index
    df[col] = df[col].replace(rare, -1)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_text = vectorizer.fit_transform(df['text_cleaned'])

# Prepare final feature matrix
X_meta = df[cat_cols + ['desc_length', 'req_length', 'has_clickbait']]
y = df['fraudulent']

# Train-test split
X_train_meta, X_test_meta, X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_meta, X_text, y, test_size=0.2, random_state=42, stratify=y
)

# Combine sparse matrix with dense features
X_train = hstack([X_train_text, X_train_meta])
X_test = hstack([X_test_text, X_test_meta])

# Save to disk
save_npz("X_train.npz", X_train)
save_npz("X_test.npz", X_test)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le_dict, "label_encoders.pkl")

print("✅ Preprocessing complete. Files saved:")
print("- X_train.npz, X_test.npz")
print("- y_train.csv, y_test.csv")
print("- tfidf_vectorizer.pkl, label_encoders.pkl")
