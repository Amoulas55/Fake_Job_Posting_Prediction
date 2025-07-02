import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Set seed
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()
print("🚀 Seed set and starting...")

# Load dataset
df = pd.read_csv("fake_job_postings.csv")
df.fillna('', inplace=True)

# Create features
df['desc_length'] = df['description'].apply(lambda x: len(x.split()))
df['req_length'] = df['requirements'].apply(lambda x: len(x.split()))
df['has_clickbait'] = df['title'].str.contains(r'(?i)work from home|no experience|quick money|urgent').astype(int)
df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']

texts = df['text'].tolist()
meta_features = df[['desc_length', 'req_length', 'has_clickbait']].values
labels = df['fraudulent'].astype(int).values

print("✅ Data loaded and features created.")

# Tokenizer and BERT model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
print(f"🧠 DistilBERT loaded on {device}")

# Define Dataset
class JobTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# DataLoader
dataset = JobTextDataset(texts, tokenizer)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Run in batches
all_embeddings = []
print("📦 Starting batch-wise embedding...")

with torch.no_grad():
    for i, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0].cpu()
        all_embeddings.append(cls_embeddings)
        if (i + 1) % 10 == 0:
            print(f"✅ Processed batch {i+1}/{len(loader)}")

bert_embeddings = torch.cat(all_embeddings, dim=0)

# Save outputs
torch.save(bert_embeddings, "distilbert_embeddings.pt")
np.save("meta_features.npy", meta_features)
np.save("labels.npy", labels)

print("✅ Embeddings saved:")
print("- distilbert_embeddings.pt")
print("- meta_features.npy")
print("- labels.npy")

# Focal loss class (for your next script)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

print("🎯 FocalLoss class is defined and ready.")
print("🏁 Done!")
