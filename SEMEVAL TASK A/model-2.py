


import os
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ------------------------- Helper Functions ----------------------------
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    word_to_idx = {word: idx + 2 for idx, word in enumerate(sorted(vocab))}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    return word_to_idx

def load_glove_embeddings(glove_path, word_to_idx, embedding_dim=300):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(word_to_idx), embedding_dim))
    for word, i in word_to_idx.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix

# ------------------------- Dataset ----------------------------
class StanceDataset(Dataset):
    def __init__(self, texts, targets, labels, word_to_idx):
        self.texts = texts
        self.targets = targets
        self.labels = labels
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx], self.labels[idx]

def collate_fn(batch):
    texts, targets, labels = zip(*batch)
    return list(texts), list(targets), torch.tensor(labels, dtype=torch.long)

# ------------------------- Model ----------------------------
class StanceDetectionModel13(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, max_seq_len, dropout=0.3):
        super(StanceDetectionModel13, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        gru_out, _ = self.gru(embeds)
        attn_weights = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        output = self.dropout(context)
        return self.fc(output)

# ------------------------- Training ----------------------------
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=15, patience=5):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for texts, targets, labels in train_loader:
            optimizer.zero_grad()
            token_ids = encode_batch(texts, model.embedding.num_embeddings, model.embedding.padding_idx)
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            outputs = model(token_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_stance_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    return model, train_losses

# Dummy encoding for batch
def encode_batch(texts, vocab_size, pad_idx, max_len=128):
    batch = []
    for text in texts:
        tokens = text.lower().split()
        indices = [word_to_idx.get(word, 1) for word in tokens][:max_len]
        if len(indices) < max_len:
            indices += [pad_idx] * (max_len - len(indices))
        batch.append(indices)
    return torch.tensor(batch, dtype=torch.long)

# ------------------------- Evaluation ----------------------------
def evaluate_and_visualize(model, test_loader, device, word_to_idx, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, targets, labels in test_loader:
            token_ids = encode_batch(texts, model.embedding.num_embeddings, model.embedding.padding_idx)
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            outputs = model(token_ids)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Visualize confusion matrix (optional)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

# ------------------------- Main ----------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_excel("train_data_semeval2016.xlsx")
    test_df = pd.read_excel("SemEval2016-Task6-subtaskA-testdata-gold.xlsx")

    all_texts = list(train_df["Tweet"]) + list(train_df["Target"])
    global word_to_idx
    word_to_idx = build_vocab(all_texts, min_freq=2)
    vocab_size = len(word_to_idx)

    glove_path = r"C:/Users/CSE RGUKT/Desktop/Major_Project/glove.42B.300d/glove.42B.300d.txt"
    embedding_matrix_np = load_glove_embeddings(glove_path, word_to_idx, embedding_dim=300)
    embedding_tensor = torch.tensor(embedding_matrix_np)

    label_encoder = LabelEncoder()
    train_df["label_encoded"] = label_encoder.fit_transform(train_df["Stance"])
    test_df["label_encoded"] = label_encoder.transform(test_df["Stance"])
    class_names = list(label_encoder.classes_)

    train_dataset = StanceDataset(train_df["Tweet"].tolist(), train_df["Target"].tolist(), train_df["label_encoded"].tolist(), word_to_idx)
    test_dataset = StanceDataset(test_df["Tweet"].tolist(), test_df["Target"].tolist(), test_df["label_encoded"].tolist(), word_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_df["label_encoded"]), y=train_df["label_encoded"])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = StanceDetectionModel13(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_size=256,
        num_classes=len(class_names),
        max_seq_len=128,
        dropout=0.3
    ).to(device)

    model.embedding.weight.data.copy_(embedding_tensor)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training the model
    train_start = time.time()
    model, train_losses = train_model(model, train_loader, criterion, optimizer, device, num_epochs=15, patience=5)
    train_end = time.time()
    training_time = train_end - train_start

    best_model = StanceDetectionModel13(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_size=256,
        num_classes=len(class_names),
        max_seq_len=128
    ).to(device)
    best_model.load_state_dict(torch.load("best_stance_model.pth"))

    # Inference
    inference_start = time.time()
    test_metrics = evaluate_and_visualize(best_model, test_loader, device, word_to_idx, class_names)
    inference_end = time.time()
    inference_time = inference_end - inference_start

    results = {
        "model": "GRU + Attention ",
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
        "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "metrics": {
            "accuracy": test_metrics['accuracy'],
            "precision": test_metrics['precision'],
            "recall": test_metrics['recall'],
            "f1_score": test_metrics['f1_score']
        },
        "confusion_matrix": test_metrics['confusion_matrix']
    }

    Path("experiment_results").mkdir(parents=True, exist_ok=True)
    with open("experiment_results/GRU_ATTENTION.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Results saved: experiment_results/GRU ATTENTION .json")

if __name__ == "__main__":
    main()
