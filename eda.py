import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('fake_job_postings.csv')

# Print basic info
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())

# Target variable distribution
print("\nClass distribution (0 = real, 1 = fake):")
print(df['fraudulent'].value_counts())

# Visualize class imbalance
sns.set(style="whitegrid")
sns.countplot(x='fraudulent', data=df)
plt.title("Fraudulent vs Real Job Postings")
plt.xlabel("Fraudulent")
plt.ylabel("Count")
plt.savefig("class_distribution.png")  # Save plot to file
plt.show()

# Preview key fields
print("\nSample job postings:")
print(df[['title', 'company_profile', 'description', 'fraudulent']].head())
