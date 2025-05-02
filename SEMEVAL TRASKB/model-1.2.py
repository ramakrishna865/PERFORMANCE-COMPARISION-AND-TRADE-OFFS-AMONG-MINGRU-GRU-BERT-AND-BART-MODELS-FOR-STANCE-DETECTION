# [Previous imports remain the same]
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

import time
import json


import numpy as np

def load_glove_embeddings(glove_file_path, word_to_idx, embedding_dim=200):
    """
    Load GloVe vectors from a text file and align them with your word_to_idx vocab.
    Returns a NumPy matrix of shape (vocab_size, embedding_dim).
    """
    vocab_size = len(word_to_idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim)).astype(np.float32)

    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word_to_idx:
                vector = np.array(parts[1:], dtype=np.float32)
                embedding_matrix[word_to_idx[word]] = vector

    return embedding_matrix

# Custom tokenizer
def simple_tokenizer(text):
    """Simple tokenizer that converts text to lowercase and splits on non-alphanumeric characters"""
    text = str(text).lower()
    # Replace non-alphanumeric with space, then split
    return re.sub(r'[^a-zA-Z0-9]', ' ', text).split()

# Build vocabulary from texts
def build_vocab(texts, min_freq=2):
    """Build vocabulary from list of texts"""
    word_counts = Counter()
    for text in texts:
        word_counts.update(simple_tokenizer(text))

    # Filter by minimum frequency
    valid_words = [word for word, count in word_counts.items() if count >= min_freq]

    # Create word-to-index mapping
    word_to_idx = {'<pad>': 0, '<unk>': 1}
    for word in valid_words:
        word_to_idx[word] = len(word_to_idx)

    return word_to_idx

# Text encoding function
def encode_text(text, word_to_idx):
    """Encode text using vocabulary"""
    tokens = simple_tokenizer(text)
    return [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]

# Custom dataset class that handles both text and topic
class StanceDataset(Dataset):
    def __init__(self, texts, topics, labels, word_to_idx):
        self.texts = texts
        self.topics = topics
        self.labels = labels
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        topic = self.topics[idx]

        # Encode text and topic
        text_encoded = encode_text(text, self.word_to_idx)
        topic_encoded = encode_text(topic, self.word_to_idx)

        return {
            'text': torch.tensor(text_encoded, dtype=torch.long),
            'topic': torch.tensor(topic_encoded, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Collate function for DataLoader to handle variable length sequences
def collate_fn(batch):
    texts = [item['text'] for item in batch]
    topics = [item['topic'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    # Pad sequences
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    padded_topics = pad_sequence(topics, batch_first=True, padding_value=0)

    return {
        'text': padded_texts,
        'topic': padded_topics,
        'label': labels
    }

# Minimal GRU Cell (simplified version of GRU)
class MinGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinGRUCell, self).__init__()
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        z = torch.sigmoid(self.update_gate(combined))
        h_tilde = torch.tanh(self.output_gate(combined))
        h = (1 - z) * hidden + z * h_tilde
        return h

# Minimal GRU Layer
class MinGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(MinGRULayer, self).__init__()
        self.cell = MinGRUCell(input_size, hidden_size)
        self.batch_first = batch_first
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            hidden = self.cell(x[:, t, :], hidden)
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden


# ✅ Updated StanceDetectionModel13 with Safe Positional Embeddings
class StanceDetectionModel13(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, max_seq_len=512, dropout=0.3):
        super(StanceDetectionModel13, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.text_gru = MinGRULayer(embedding_dim, hidden_size, batch_first=True)
        self.topic_gru = MinGRULayer(embedding_dim, hidden_size, batch_first=True)

        self.attention = SelfAttentionOverGRU(hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, topic):
        device = text.device
        batch_size, seq_len = text.size()

        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        positions = torch.clamp(positions, max=self.position_embedding.num_embeddings - 1)

        text_embed = self.word_embedding(text) + self.position_embedding(positions)

        topic_seq_len = topic.size(1)
        topic_positions = torch.arange(0, topic_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        topic_positions = torch.clamp(topic_positions, max=self.position_embedding.num_embeddings - 1)

        topic_embed = self.word_embedding(topic) + self.position_embedding(topic_positions)

        text_output, _ = self.text_gru(text_embed)
        topic_output, topic_hidden = self.topic_gru(topic_embed)

        context_vector, attention_weights = self.attention(text_output, topic_hidden)

        combined = torch.cat((context_vector, topic_hidden), dim=1)
        x = self.dropout(torch.relu(self.fc1(combined)))
        logits = self.fc2(self.dropout(x))

        return logits, attention_weights


# Topic-aware Attention mechanism
class SelfAttentionOverGRU(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttentionOverGRU, self).__init__()
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, text_hidden, topic_hidden):
        batch_size, seq_len, _ = text_hidden.size()
        topic_expanded = topic_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat((text_hidden, topic_expanded), dim=2)
        scores = self.linear2(self.tanh(self.linear1(combined)))
        attn_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attn_weights * text_hidden, dim=1)
        return context_vector, attn_weights
    



#         return logits, attention_weights

def analyze_attention_weights(model, data_loader, device, word_to_idx, class_names, idx_to_word=None, num_examples=5):
    if idx_to_word is None:
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    model.eval()
    examples = []
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            if count >= num_examples:
                break

            texts = batch['text'].to(device)
            topics = batch['topic'].to(device)
            labels = batch['label'].to(device)

            logits, attention_weights = model(texts, topics)
            probabilities = torch.softmax(logits, dim=1)
            _, predictions = torch.max(logits, 1)

            for i in range(min(len(texts), num_examples - count)):
                text_indices = texts[i].cpu().tolist()
                text_words = [idx_to_word.get(idx, '<unk>') for idx in text_indices if idx != 0]

                topic_indices = topics[i].cpu().tolist()
                topic_words = [idx_to_word.get(idx, '<unk>') for idx in topic_indices if idx != 0]

                attn = attention_weights[i, :len(text_words)].squeeze(-1).cpu().tolist()

                pred = predictions[i].item()
                true_label = labels[i].item()
                probs = probabilities[i].cpu().tolist()

                examples.append({
                    'text_words': text_words,
                    'topic_words': topic_words,
                    'attention': attn,
                    'prediction': pred,
                    'true_label': true_label,
                    'probabilities': probs
                })
                count += 1

    try:
        for i, example in enumerate(examples):
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(example['text_words'])), example['attention'])
            plt.yticks(range(len(example['text_words'])), example['text_words'])

            topic_str = ' '.join(example['topic_words'])
            pred_label = class_names[example['prediction']]
            true_label = class_names[example['true_label']]

            plt.title(f"Attention Weights for Example {i+1}\nTopic: {topic_str}\nPrediction: {pred_label} (True: {true_label})")
            plt.xlabel('Attention Weight')
            plt.tight_layout()
            plt.savefig(f'attention_example_{i+1}.png')
            plt.close()
        print("Attention weight visualizations saved as 'attention_example_X.png'")
    except Exception as e:
        print(f"Skipping attention visualization due to error: {e}")

    return examples
def train_model(model, train_loader,  criterion, optimizer, device, num_epochs=10, patience=3):
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []  # Store train losses
    val_losses = [] 
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            texts = batch['text'].to(device)
            topics = batch['topic'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(texts, topics)
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate statistics
            train_loss += loss.item() * texts.size(0)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)  # Append train loss
        

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

       

    
        
    torch.save(model.state_dict(), 'best_stance_model.pth')
    return model, train_losses, val_losses

# Prediction function
def predict_stance(model, text, topic, word_to_idx, device):
    # Encode text and topic
    text_encoded = encode_text(text, word_to_idx)
    topic_encoded = encode_text(topic, word_to_idx)

    # Convert to tensors and add batch dimension
    text_tensor = torch.tensor([text_encoded], dtype=torch.long).to(device)
    topic_tensor = torch.tensor([topic_encoded], dtype=torch.long).to(device)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        logits, attention_weights = model(text_tensor, topic_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Map class index to label
    stance_labels = {0: "AGAINST", 1: "FAVOR", 2: "NEUTRAL"}

    return {
        'stance': stance_labels[predicted_class],
        'probabilities': probabilities.cpu().detach()[0],
        'attention_weights': attention_weights.cpu()
    }

def evaluate_model_performance(model, data_loader, device, class_names):
    """
    Evaluate model performance on a dataset.

    Args:
        model: The trained model
        data_loader: DataLoader containing evaluation data
        device: Device to run inference on (cpu or cuda)
        class_names: List of class names (e.g. ["AGAINST", "FAVOR", "NEUTRAL"])

    Returns:
        Dictionary containing performance metrics
    """
    model.eval()

    # Initialize lists to store predictions and actual labels
    all_predictions = []
    all_labels = []
    all_probabilities = []

    # Disable gradient calculations for inference
    with torch.no_grad():
        # Use tqdm to show progress bar
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            texts = batch['text'].to(device)
            topics = batch['topic'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits, _ = model(texts, topics)
            probabilities = torch.softmax(logits, dim=1)

            # Get predicted class
            _, predictions = torch.max(logits, dim=1)

            # Store results
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    # Calculate basic accuracy (can be done without numpy)
    correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0

    # Create a simple confusion matrix using Python lists
    num_classes = len(class_names)
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, pred_label in zip(all_labels, all_predictions):
        cm[true_label][pred_label] += 1

    # Calculate precision, recall, and f1 for each class
    class_metrics = []
    for class_idx in range(num_classes):
        # True positives: predicted class_idx and actually class_idx
        tp = cm[class_idx][class_idx]

        # False positives: predicted class_idx but actually not class_idx
        fp = sum(cm[i][class_idx] for i in range(num_classes) if i != class_idx)

        # False negatives: predicted not class_idx but actually class_idx
        fn = sum(cm[class_idx][i] for i in range(num_classes) if i != class_idx)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(cm[class_idx])

        class_metrics.append({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })

    # Calculate macro averages
    macro_precision = sum(m['precision'] for m in class_metrics) / num_classes
    macro_recall = sum(m['recall'] for m in class_metrics) / num_classes
    macro_f1 = sum(m['f1'] for m in class_metrics) / num_classes

    # Calculate weighted averages
    total_support = sum(m['support'] for m in class_metrics)
    weighted_precision = sum(m['precision'] * m['support'] for m in class_metrics) / total_support if total_support > 0 else 0
    weighted_recall = sum(m['recall'] * m['support'] for m in class_metrics) / total_support if total_support > 0 else 0
    weighted_f1 = sum(m['f1'] * m['support'] for m in class_metrics) / total_support if total_support > 0 else 0

    # Store all metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'class_precision': {class_names[i]: class_metrics[i]['precision'] for i in range(num_classes)},
        'class_recall': {class_names[i]: class_metrics[i]['recall'] for i in range(num_classes)},
        'class_f1': {class_names[i]: class_metrics[i]['f1'] for i in range(num_classes)},
        'class_support': {class_names[i]: class_metrics[i]['support'] for i in range(num_classes)},
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'true_labels': all_labels,
        'probabilities': all_probabilities
    }

    return metrics
def print_metrics_summary(metrics, class_names):
    """
    Print a summary of model performance metrics.

    Args:
        metrics: Dictionary containing performance metrics
        class_names: List of class names
    """
    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")

    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'-'*50}")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*50}")

    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10} {metrics['class_precision'][class_name]:.4f}{'':<7} "
              f"{metrics['class_recall'][class_name]:.4f}{'':<7} "
              f"{metrics['class_f1'][class_name]:.4f}{'':<7} "
              f"{metrics['class_support'][class_name]:<10}")

    print(f"{'-'*50}")
    print(f"{'Macro Avg':<10} {metrics['macro_precision']:.4f}{'':<7} "
          f"{metrics['macro_recall']:.4f}{'':<7} "
          f"{metrics['macro_f1']:.4f}{'':<7}")

    print(f"{'Weighted Avg':<10} {metrics['weighted_precision']:.4f}{'':<7} "
          f"{metrics['weighted_recall']:.4f}{'':<7} "
          f"{metrics['weighted_f1']:.4f}{'':<7}")
    print(f"{'='*50}")

def plot_confusion_matrix(metrics, class_names, figsize=(10, 8)):
    """
    Plot confusion matrix as a heatmap.

    Args:
        metrics: Dictionary containing performance metrics
        class_names: List of class names
        figsize: Figure size (width, height) in inches
    """
    try:
        plt.figure(figsize=figsize)

        # Get confusion matrix
        cm = metrics['confusion_matrix']

        # Normalize confusion matrix to show percentages
        cm_norm = []
        for i, row in enumerate(cm):
            row_sum = sum(row)
            if row_sum > 0:
                cm_norm.append([val / row_sum for val in row])
            else:
                cm_norm.append([0] * len(row))

        # Create heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Normalized)')
        plt.tight_layout()

        # Save the figure
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Also plot the raw counts
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Counts)')
        plt.tight_layout()

        # Save the figure
        plt.savefig('confusion_matrix_counts.png')
        plt.close()

        print("Confusion matrix plots saved as 'confusion_matrix.png' and 'confusion_matrix_counts.png'")
    except Exception as e:
        print(f"Skipping confusion matrix visualization due to error: {e}")
def plot_roc_curve(metrics, class_names, figsize=(10, 8)):
    """
    Plot ROC curves for each class (one-vs-rest).

    Args:
        metrics: Dictionary containing performance metrics
        class_names: List of class names
        figsize: Figure size (width, height) in inches
    """
    try:
        # Get true labels and probabilities
        true_labels = metrics['true_labels']
        probabilities = metrics['probabilities']

        # Create one-hot encoded true labels
        num_classes = len(class_names)
        y_bin = []
        for label in true_labels:
            one_hot = [0] * num_classes
            one_hot[label] = 1
            y_bin.append(one_hot)

        plt.figure(figsize=figsize)

        # Plot ROC curve for each class
        for i, class_name in enumerate(class_names):
            # Calculate ROC curve points
            fpr = []
            tpr = []
            thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            for threshold in thresholds:
                # True positives, false positives, etc.
                tp = fp = tn = fn = 0

                for j in range(len(probabilities)):
                    pred_positive = probabilities[j][i] >= threshold
                    true_positive = y_bin[j][i] == 1

                    if pred_positive and true_positive:
                        tp += 1
                    elif pred_positive and not true_positive:
                        fp += 1
                    elif not pred_positive and true_positive:
                        fn += 1
                    else:  # not pred_positive and not true_positive
                        tn += 1

                # Calculate rates
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0

                fpr.append(fpr_val)
                tpr.append(tpr_val)

            # Calculate AUC approximation
            auc_val = 0
            for j in range(len(fpr) - 1):
                auc_val += (tpr[j] + tpr[j+1]) * (fpr[j+1] - fpr[j]) / 2

            plt.plot(
                fpr,
                tpr,
                lw=2,
                label=f'{class_name} (AUC = {auc_val:.2f})'
            )

        # Plot random guess line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()

        # Save the figure
        plt.savefig('roc_curves.png')
        plt.close()

        print("ROC curve plot saved as 'roc_curves.png'")
    except Exception as e:
        print(f"Skipping ROC curve visualization due to error: {e}")
def analyze_attention_weights(model, data_loader, device, word_to_idx, idx_to_word=None, num_examples=5):
    """
    Analyze and visualize attention weights for a few examples.

    Args:
        model: The trained model
        data_loader: DataLoader containing evaluation data
        device: Device to run inference on (cpu or cuda)
        word_to_idx: Word to index mapping dictionary
        idx_to_word: Index to word mapping dictionary (if None, will be created from word_to_idx)
        num_examples: Number of examples to visualize
    """
    if idx_to_word is None:
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    model.eval()

    examples = []
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            if count >= num_examples:
                break

            texts = batch['text'].to(device)
            topics = batch['topic'].to(device)
            labels = batch['label'].to(device)

            # Get model predictions and attention weights
            logits, attention_weights = model(texts, topics)
            probabilities = torch.softmax(logits, dim=1)
            _, predictions = torch.max(logits, dim=1)

            # Get a few examples from this batch
            for i in range(min(len(texts), num_examples - count)):
                # Get text words (removing padding)
                text_indices = texts[i].cpu().tolist()  # Convert to Python list instead of numpy
                text_words = [idx_to_word.get(idx, '<unk>') for idx in text_indices if idx != 0]  # 0 is padding

                # Get topic words
                topic_indices = topics[i].cpu().tolist()  # Convert to Python list instead of numpy
                topic_words = [idx_to_word.get(idx, '<unk>') for idx in topic_indices if idx != 0]

                # Get attention weights for this example
                attn = attention_weights[i, :len(text_words)].cpu().tolist()  # Convert to Python list instead of numpy

                # Get prediction and true label
                pred = predictions[i].item()
                true_label = labels[i].item()
                probs = probabilities[i].cpu().tolist()  # Convert to Python list instead of numpy

                examples.append({
                    'text_words': text_words,
                    'topic_words': topic_words,
                    'attention': attn,
                    'prediction': pred,
                    'true_label': true_label,
                    'probabilities': probs
                })

                count += 1

    # Skip visualization if matplotlib is not available
    try:
        # Visualize attention weights for each example
        for i, example in enumerate(examples):
            plt.figure(figsize=(12, 6))

            # Plot attention weights
            plt.barh(range(len(example['text_words'])), example['attention'])
            plt.yticks(range(len(example['text_words'])), example['text_words'])

            topic_str = ' '.join(example['topic_words'])
            pred_label = class_names[example['prediction']]
            true_label = class_names[example['true_label']]

            plt.title(f"Attention Weights for Example {i+1}\nTopic: {topic_str}\nPrediction: {pred_label} (True: {true_label})")
            plt.xlabel('Attention Weight')
            plt.tight_layout()

            # Save the figure
            plt.savefig(f'attention_example_{i+1}.png')
            plt.close()
        print("Attention weight visualizations saved as 'attention_example_X.png'")
    except Exception as e:
        print(f"Skipping attention visualization due to error: {e}")

    return examples
def evaluate_and_visualize(model, test_loader, device, word_to_idx, class_names):
    """
    Complete evaluation and visualization of model performance.

    Args:
        model: The trained model
        test_loader: DataLoader containing test data
        device: Device to run inference on (cpu or cuda)
        word_to_idx: Word to index mapping dictionary
        class_names: List of class names
    """
    print("Evaluating model performance...")

    try:
        # Calculate metrics
        metrics = evaluate_model_performance(model, test_loader, device, class_names)

        # Print metrics summary
        print_metrics_summary(metrics, class_names)

        # Plot confusion matrix
        plot_confusion_matrix(metrics, class_names)

        # Plot ROC curves
        plot_roc_curve(metrics, class_names)

        # Create index to word mapping
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

        # Analyze attention weights
        analyze_attention_weights(model, test_loader, device, word_to_idx, idx_to_word, num_examples=5)

        return metrics
    except Exception as e:
        print(f"Error during evaluation and visualization: {e}")
        return None



def main():
    import os
    import time
    import json
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import DataLoader

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_df = pd.read_excel("train_data_semeval2016.xlsx")
    test_df = pd.read_excel("SemEval2016-Task6-subtaskB-testdata-gold.xlsx")

    all_texts = list(train_df["Tweet"]) + list(train_df["Target"])
    word_to_idx = build_vocab(all_texts, min_freq=2)
    vocab_size = len(word_to_idx)

    from pathlib import Path
    glove_path = r"C:/Users/CSE RGUKT/Desktop/Major_Project/glove.42B.300d/glove.42B.300d.txt"
    embedding_matrix_np = load_glove_embeddings(glove_path, word_to_idx, embedding_dim=300)
    embedding_tensor = torch.tensor(embedding_matrix_np)

    label_encoder = LabelEncoder()
    train_df["label_encoded"] = label_encoder.fit_transform(train_df["Stance"])
    # val_df["label_encoded"] = label_encoder.transform(val_df["Stance"])
    test_df["label_encoded"] = label_encoder.transform(test_df["Stance"])
    class_names = list(label_encoder.classes_)

    train_dataset = StanceDataset(train_df["Tweet"].tolist(), train_df["Target"].tolist(), train_df["label_encoded"].tolist(), word_to_idx)
    # val_dataset = StanceDataset(val_df["Tweet"].tolist(), val_df["Target 1"].tolist(), val_df["label_encoded"].tolist(), word_to_idx)
    test_dataset = StanceDataset(test_df["Tweet"].tolist(), test_df["Target"].tolist(), test_df["label_encoded"].tolist(), word_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_df["label_encoded"]), y=train_df["label_encoded"])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # model = StanceDetectionModel13(vocab_size=vocab_size,
    #                              embedding_dim=300,
    #                              hidden_size=256,
    #                              num_classes=len(class_names),
    #                              word_to_idx=word_to_idx,
    #                              pretrained_embeddings=embedding_tensor,
    #                              freeze_embeddings=True).to(device)

    model = StanceDetectionModel13(
    vocab_size=vocab_size,
    embedding_dim=300,
    hidden_size=256,
    num_classes=len(class_names),
    max_seq_len=128,
    dropout=0.3
        ).to(device)


    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_start = time.time()
    trained_model, train_losses, val_losses = train_model(
        model, train_loader,  criterion, optimizer, device, num_epochs=15, patience=5
    )
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

    inference_start = time.time()
    test_metrics = evaluate_and_visualize(best_model, test_loader, device, word_to_idx, class_names)
    inference_end = time.time()
    inference_time = inference_end - inference_start

    results = {
        "model": "Mini-GRU + Topic-Aware Attention (GloVe)",
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
        "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "metrics": {
            "accuracy": test_metrics['accuracy'],
            "precision": test_metrics['weighted_precision'],
            "recall": test_metrics['weighted_recall'],
            "f1_score": test_metrics['weighted_f1']
        },
        "confusion_matrix": test_metrics['confusion_matrix']
    }

    Path("experiment_results").mkdir(parents=True, exist_ok=True)
    with open("experiment_results/min_gru_attention_glove.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Results saved: experiment_results/min_gru_attention_glove.json")

if __name__ == "__main__":
    main()

