import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define the BERT-based model for only BERT features (Ablation 1)
class ModernBERTTextClassifierOnlyBERT(nn.Module):
    def __init__(self, bert_model, num_classes, dropout_rate=0.2):
        super(ModernBERTTextClassifierOnlyBERT, self).__init__()
        self.bert = bert_model
        self.fc = nn.Sequential( 
            nn.Linear(768, num_classes),  # Only using BERT's [CLS] token output (768-dimension)
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT-based semantic feature extraction
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # Using the last hidden state
        cls_token_state = last_hidden_state[:, 0, :]  # [CLS] token output
        
        # Classification layer
        output = self.fc(cls_token_state)
        return output


# ModernBERT + TF-IDF Model (Ablation 2)
class ModernBERTTextClassifierWithTFIDF(nn.Module):
    def __init__(self, bert_model, tfidf_dim, num_classes, dropout_rate=0.2):
        super(ModernBERTTextClassifierWithTFIDF, self).__init__()
        self.bert = bert_model
        self.fc = nn.Sequential( 
            nn.Linear(768 + tfidf_dim, num_classes),  # Combine BERT and TF-IDF features
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, input_ids, attention_mask, tfidf_features):
        # BERT-based semantic feature extraction
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  
        cls_token_state = last_hidden_state[:, 0, :]  # [CLS] token output
        
        # Concatenate BERT and TF-IDF features
        combined_features = torch.cat((cls_token_state, tfidf_features), dim=1)
        
        # Classification layer
        output = self.fc(combined_features)
        return output


# ModernBERT + Chi-Square Model (Ablation 3)
class ModernBERTTextClassifierWithChiSquare(nn.Module):
    def __init__(self, bert_model, tfidf_dim, num_classes, dropout_rate=0.2):
        super(ModernBERTTextClassifierWithChiSquare, self).__init__()
        self.bert = bert_model
        self.fc = nn.Sequential( 
            nn.Linear(768 + tfidf_dim, num_classes),  # Combine BERT and Chi-Square features
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, input_ids, attention_mask, chi_square_features):
        # BERT-based semantic feature extraction
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  
        cls_token_state = last_hidden_state[:, 0, :]  # [CLS] token output
        
        # Concatenate BERT and Chi-Square features
        combined_features = torch.cat((cls_token_state, chi_square_features), dim=1)
        
        # Classification layer
        output = self.fc(combined_features)
        return output


# ModernBERT + Document Frequency Model (Ablation 4)
class ModernBERTTextClassifierWithDF(nn.Module):
    def __init__(self, bert_model, tfidf_dim, num_classes, dropout_rate=0.2):
        super(ModernBERTTextClassifierWithDF, self).__init__()
        self.bert = bert_model
        self.fc = nn.Sequential( 
            nn.Linear(768 + tfidf_dim, num_classes),  # Combine BERT and Document Frequency features
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, input_ids, attention_mask, df_features):
        # BERT-based semantic feature extraction
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  
        cls_token_state = last_hidden_state[:, 0, :]  # [CLS] token output
        
        # Concatenate BERT and Document Frequency features
        combined_features = torch.cat((cls_token_state, df_features), dim=1)
        
        # Classification layer
        output = self.fc(combined_features)
        return output


# Chi-Square Feature Extraction
def compute_chi_square_features(texts, labels):
    vectorizer = TfidfVectorizer()
    tfidf_features = vectorizer.fit_transform(texts).toarray()

    # Using Chi-Square to select top features
    chi_square_selector = SelectKBest(chi2, k=100)  # Select top 100 features
    chi_square_features = chi_square_selector.fit_transform(tfidf_features, labels)
    
    return torch.tensor(chi_square_features, dtype=torch.float32)


# Document Frequency Feature Extraction (IDF)
def compute_document_frequency_features(texts):
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_features = vectorizer.fit_transform(texts).toarray()
    
    # Extract only the IDF values (Document Frequency)
    idf_features = vectorizer.idf_
    idf_features = torch.tensor(idf_features, dtype=torch.float32).unsqueeze(0).repeat(len(texts), 1)
    
    return idf_features


# Load and shuffle data
csv_file_path = os.path.join('..', 'AML_Dataset', 'movie_vae_0_1_hc.csv') 
df = pd.read_csv(csv_file_path)
df = df.sample(frac=1).reset_index(drop=True)

# Preprocess Text Data
texts = df['text'].tolist()
labels = df['label'].values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels) 

# Tokenize Text using ModernBERT
tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

tokenized_texts = tokenize(texts)
input_ids = tokenized_texts['input_ids']
attention_mask = tokenized_texts['attention_mask']

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer()
tfidf_features = vectorizer.fit_transform(texts).toarray()

# Dynamically set tfidf_dim
tfidf_dim = tfidf_features.shape[1]

# Convert TF-IDF features to torch tensor
tfidf_features = torch.tensor(tfidf_features, dtype=torch.float32)

# Chi-Square Feature Extraction
chi_square_features = compute_chi_square_features(texts, encoded_labels)

# Document Frequency Feature Extraction
df_features = compute_document_frequency_features(texts)

# Split the dataset for training, validation, and evaluation
X_train_ids, X_temp_ids, X_train_attention, X_temp_attention, X_train_tfidf, X_temp_tfidf, y_train, y_temp = train_test_split(input_ids, attention_mask, tfidf_features, encoded_labels, test_size=0.3, random_state=42)

X_val_ids, X_test_ids, X_val_attention, X_test_attention, X_val_tfidf, X_test_tfidf, y_val, y_test = train_test_split(X_temp_ids, X_temp_attention, X_temp_tfidf, y_temp, test_size=0.5, random_state=42)

# Move data to GPU
X_train_ids, X_train_attention, X_train_tfidf, y_train = X_train_ids.to(device), X_train_attention.to(device), X_train_tfidf.to(device), torch.tensor(y_train).to(device)
X_val_ids, X_val_attention, X_val_tfidf, y_val = X_val_ids.to(device), X_val_attention.to(device), X_val_tfidf.to(device), torch.tensor(y_val).to(device)
X_test_ids, X_test_attention, X_test_tfidf, y_test = X_test_ids.to(device), X_test_attention.to(device), X_test_tfidf.to(device), torch.tensor(y_test).to(device)

# Create DataLoader for batching
batch_size = 16  

train_data = TensorDataset(X_train_ids, X_train_attention, X_train_tfidf, y_train)
val_data = TensorDataset(X_val_ids, X_val_attention, X_val_tfidf, y_val)
test_data = TensorDataset(X_test_ids, X_test_attention, X_test_tfidf, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the model, loss function, and optimizer
encoding_dim = 128  
modernbert_model = ModernBertForSequenceClassification.from_pretrained('answerdotai/ModernBERT-base').to(device)

# For Only ModernBERT
model_only_bert = ModernBERTTextClassifierOnlyBERT(bert_model=modernbert_model, num_classes=2).to(device)

# For ModernBERT + TF-IDF
model_tfidf = ModernBERTTextClassifierWithTFIDF(bert_model=modernbert_model, tfidf_dim=tfidf_dim, num_classes=2).to(device)

# For ModernBERT + Chi-Square
model_chi_square = ModernBERTTextClassifierWithChiSquare(bert_model=modernbert_model, tfidf_dim=chi_square_features.shape[1], num_classes=2).to(device)

# For ModernBERT + Document Frequency
model_df = ModernBERTTextClassifierWithDF(bert_model=modernbert_model, tfidf_dim=df_features.shape[1], num_classes=2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_only_bert.parameters(), lr=1e-5)  

# Early Stopping Parameters
patience = 3
best_val_loss = float('inf')
epochs_without_improvement = 0

# Evaluation Metrics Function
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, tfidf_features, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # tfidf_features = tfidf_features.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(input_ids=input_ids, attention_mask=attention_mask)

            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return accuracy, precision, recall, f1

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model_only_bert.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        input_ids, attention_mask, tfidf_features, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        tfidf_features = tfidf_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model_only_bert(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples

    # Print progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    # Early stopping
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping due to no improvement.")
            break

# After Training, Evaluate on the Test Set
test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model_only_bert, test_loader)

# Report metrics for the test set
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
