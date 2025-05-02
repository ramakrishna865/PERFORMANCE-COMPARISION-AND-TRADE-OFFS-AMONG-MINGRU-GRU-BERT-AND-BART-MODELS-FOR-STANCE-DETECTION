from transformers import BartTokenizerFast, BartForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from sklearn.preprocessing import label_binarize

class StanceDatasetBART(Dataset):
    def __init__(self, df, tokenizer, label_encoder, max_len=128):
        self.encodings = tokenizer(
            (df['Target'] + " [SEP] " + df['Tweet']).tolist(),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        self.labels = torch.tensor(label_encoder.transform(df['Stance'].tolist()), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Handle tuple outputs (e.g., (logits, other_stuff)) from model
    if isinstance(logits, tuple):
        logits = logits[0]

    # Ensure logits is a numpy array
    logits = np.asarray(logits)

    # Ensure shape is (n_samples, n_classes)
    if logits.ndim == 3:
        logits = np.squeeze(logits, axis=1)

    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("bart_confusion_matrix.png")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig("bart_confusion_matrix_normalized.png")

def plot_roc_curve(true_labels, probs, class_names):
    y_bin = label_binarize(true_labels, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BART ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("bart_roc_curve.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    df_train = pd.read_excel("train_data_semeval2016.xlsx")
    df_test = pd.read_excel("SemEval2016-Task6-subtaskB-testdata-gold.xlsx")

    label_encoder = LabelEncoder()
    label_encoder.fit(df_train["Stance"])
    class_names = label_encoder.classes_

    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    # Prepare datasets
    train_dataset = StanceDatasetBART(df_train, tokenizer, label_encoder)
    test_dataset = StanceDatasetBART(df_test, tokenizer, label_encoder)

    # Load model
    model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=len(class_names)).to(device)

    training_args = TrainingArguments(
        output_dir="./bart_output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    train_start = time.time()
    trainer.train()
    train_end = time.time()
    training_time = train_end - train_start

    # Evaluate
    inference_start = time.time()
    results = trainer.evaluate(eval_dataset=test_dataset)
    inference_end = time.time()
    inference_time = inference_end - inference_start

    print("\nTest Evaluation Results:", results)

    # Predict
    predictions = trainer.predict(test_dataset)
    raw_logits = predictions.predictions
    if isinstance(raw_logits, tuple):
        raw_logits = raw_logits[0]

    if raw_logits.ndim == 3:
        raw_logits = np.squeeze(raw_logits, axis=1)

    probs = torch.nn.functional.softmax(torch.tensor(raw_logits), dim=1).numpy()
    preds = np.argmax(raw_logits, axis=1)
    true_labels = predictions.label_ids

    cm = confusion_matrix(true_labels, preds)
    plot_confusion_matrix(cm, class_names)
    plot_roc_curve(true_labels, probs, class_names)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics_json = {
        "model": "BART",
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
        "total_trainable_params": total_params,
        "accuracy": results["eval_accuracy"],
        "precision": results["eval_precision"],
        "recall": results["eval_recall"],
        "f1_score": results["eval_f1"]
    }

    Path("experiment_results").mkdir(exist_ok=True)
    with open("experiment_results/bart.json", "w") as f:
        json.dump(metrics_json, f, indent=4)

    print("\nMetrics saved to experiment_results/bart.json")

if __name__ == "__main__":
    main()
