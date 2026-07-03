import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define the AutoEncoder (AE) for dimensionality reduction
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),  # input_dim will be dynamically determined based on TF-IDF size
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the BERT-based model for feature extraction and classification
class RobertaTextClassifier(nn.Module):
    def __init__(self, roberta_model, tfidf_dim, encoding_dim, num_classes):
        super(RobertaTextClassifier, self).__init__()
        self.roberta = roberta_model
        self.autoencoder = AutoEncoder(input_dim=tfidf_dim, encoding_dim=encoding_dim)  # tfidf_dim is dynamically set
        self.fc = nn.Linear(768 + encoding_dim // 2, num_classes)  # Concatenate BERT and AE features
        
    def forward(self, input_ids, attention_mask, tfidf_features):
        # BERT-based semantic feature extraction
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        semantic_features = outputs.pooler_output  # [CLS] token output
        
        # Apply AutoEncoder to TF-IDF features
        encoded_tfidf, _ = self.autoencoder(tfidf_features)
        
        # Concatenate BERT and AE features
        combined_features = torch.cat((semantic_features, encoded_tfidf), dim=1)
        
        # Classification layer
        output = self.fc(combined_features)
        return output

# Load and shuffle data
csv_file_path = os.path.join('..', 'AML_Dataset', 'movie_0_1_ac.csv') 
    
# Read data from CSV file
df = pd.read_csv(csv_file_path)
df = df.sample(frac=1).reset_index(drop=True)

# Preprocess Text Data
texts = df['text'].tolist()
labels = df['label'].values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Tokenize Text using BERT
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

tokenized_texts = tokenize(texts)
input_ids = tokenized_texts['input_ids']
attention_mask = tokenized_texts['attention_mask']

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer()
tfidf_features = vectorizer.fit_transform(texts).toarray()

# Dynamically set tfidf_dim to the number of features in the TF-IDF vectors
tfidf_dim = tfidf_features.shape[1]  # This will give the number of features (columns) in the TF-IDF matrix

# Convert TF-IDF features to torch tensor
tfidf_features = torch.tensor(tfidf_features, dtype=torch.float32)

# Split the dataset for training and evaluation
X_train_ids, X_test_ids, X_train_attention, X_test_attention, X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    input_ids, attention_mask, tfidf_features, encoded_labels, 
    test_size=0.2, random_state=42
)

# Move all data to GPU during training and testing
X_train_ids, X_train_attention, X_train_tfidf, y_train = X_train_ids.to(device), X_train_attention.to(device), X_train_tfidf.to(device), torch.tensor(y_train).to(device)
X_test_ids, X_test_attention, X_test_tfidf, y_test = X_test_ids.to(device), X_test_attention.to(device), X_test_tfidf.to(device), torch.tensor(y_test).to(device)

# Create DataLoader for batching
batch_size = 16  

train_data = TensorDataset(X_train_ids, X_train_attention, X_train_tfidf, y_train)
test_data = TensorDataset(X_test_ids, X_test_attention, X_test_tfidf, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the model, loss function, and optimizer
encoding_dim = 128  

roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
model = RobertaTextClassifier(roberta_model=roberta_model, tfidf_dim=tfidf_dim, encoding_dim=encoding_dim, num_classes=2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-5)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, tfidf_features, labels = batch

        # Move batch to GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        tfidf_features = tfidf_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(input_ids=input_ids, attention_mask=attention_mask, tfidf_features=tfidf_features)

        # Compute loss
        loss = criterion(output, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Free up GPU memory after each epoch
    torch.cuda.empty_cache()

# Evaluation
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, tfidf_features, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        tfidf_features = tfidf_features.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(input_ids=input_ids, attention_mask=attention_mask, tfidf_features=tfidf_features)
        _, predicted = torch.max(output, 1)

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.4f}")
