# from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
# import pandas as pd
# import numpy as np
# import torch
# import os
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import time


# # Preprocessing
# def preprocess_data(df, label_encoder, tokenizer):
#     df["label"] = label_encoder.transform(df["Stance"])
#     df["text"] = df["Target"] + " [SEP] " + df["Tweet"]
#     hf_dataset = Dataset.from_pandas(df[["text", "label"]])
#     hf_dataset = hf_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
#     return hf_dataset


# # Metrics for Trainer
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     acc = accuracy_score(labels, preds)
#     return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# # Save all visual outputs + results
# def save_bert_results_and_visuals(preds, labels, class_names, metrics_dict, output_dir=str(Path.home() / "Desktop" / "result"), model_name="bert"):
#     os.makedirs(output_dir, exist_ok=True)

#     # Save metrics
#     with open(os.path.join(output_dir, f"{model_name}.json"), "w") as f:
#         json.dump(metrics_dict, f, indent=4)

#     # Confusion matrix
#     cm = confusion_matrix(labels, preds)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.title("Confusion Matrix - BERT")
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
#     plt.close()

#     # Normalized confusion matrix
#     cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.title("Normalized Confusion Matrix - BERT")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix_normalized.png"))
#     plt.close()

#     # ROC Curve
#     y_true_bin = np.eye(len(class_names))[labels]
#     y_scores = np.eye(len(class_names))[preds]  # placeholder scores
#     plt.figure(figsize=(10, 8))
#     for i, class_name in enumerate(class_names):
#         fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.title('ROC Curve - BERT')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
#     plt.close()


# # === MAIN ===
# def main():
#     # Load dataset
#     train_df = pd.read_csv("trainmerged.csv")
#     val_df = pd.read_csv("valmerged.csv")
#     test_df = pd.read_csv("testmerged.csv")

#     # Encode labels
#     label_encoder = LabelEncoder()
#     label_encoder.fit(train_df["Stance"])
#     class_names = list(label_encoder.classes_)

#     # Load tokenizer & model
#     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
#     model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(class_names))

#     # HuggingFace Datasets
#     train_dataset = preprocess_data(train_df, label_encoder, tokenizer)
#     val_dataset = preprocess_data(val_df, label_encoder, tokenizer)
#     test_dataset = preprocess_data(test_df, label_encoder, tokenizer)

#     # Define desktop result path
#     desktop_path = Path.home() / "Desktop" / "result"
#     desktop_path.mkdir(parents=True, exist_ok=True)

#     # TrainingArguments
#     training_args = TrainingArguments(
#         output_dir=str(desktop_path / "bert_output"),
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         logging_strategy="epoch",
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=4,
#         weight_decay=0.01,
#         load_best_model_at_end=True,
#         metric_for_best_model="f1",
#         save_total_limit=1,
#         report_to="none"
#     )

#     # Trainer setup
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         compute_metrics=compute_metrics
#     )

#     # Train with timing
#     train_start = time.time()
#     trainer.train()
#     train_end = time.time()
#     training_time = train_end - train_start

#     # Inference with timing
#     inference_start = time.time()
#     predictions = trainer.predict(test_dataset)
#     inference_end = time.time()
#     inference_time = inference_end - inference_start

#     # Compute metrics
#     metrics = trainer.evaluate(eval_dataset=test_dataset)
#     metrics["training_time_sec"] = training_time
#     metrics["inference_time_sec"] = inference_time

#     # Predictions
#     preds = predictions.predictions.argmax(axis=1)
#     labels = predictions.label_ids

#     # Save visuals + metrics
#     save_bert_results_and_visuals(preds, labels, class_names, metrics, output_dir=str(desktop_path))

#     print("BERT stance detection completed and results saved to Desktop/result!")


# if __name__ == "__main__":
#     main()



from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time


# Preprocessing
def preprocess_data(df, label_encoder, tokenizer):
    df["label"] = label_encoder.transform(df["Stance"])
    df["text"] = df["Target"] + " [SEP] " + df["Tweet"]
    hf_dataset = Dataset.from_pandas(df[["text", "label"]])
    hf_dataset = hf_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
    return hf_dataset


# Metrics for Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# Save results and visuals
def save_bert_results_and_visuals(preds, labels, class_names, metrics_dict, output_dir=str(Path.home() / "Desktop" / "result"), model_name="bert"):
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(output_dir, f"{model_name}.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - BERT")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix - BERT")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix_normalized.png"))
    plt.close()

    # ROC Curve
    y_true_bin = np.eye(len(class_names))[labels]
    y_scores = np.eye(len(class_names))[preds]  # placeholder scores
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - BERT')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
    plt.close()


# === MAIN ===
def main():
    # Load dataset
    train_df = pd.read_csv("trainmerged.csv")
    val_df = pd.read_csv("valmerged.csv")
    test_df = pd.read_csv("testmerged.csv")

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Stance"])
    class_names = list(label_encoder.classes_)

    # Load tokenizer & model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(class_names))

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    # HuggingFace Datasets
    train_dataset = preprocess_data(train_df, label_encoder, tokenizer)
    val_dataset = preprocess_data(val_df, label_encoder, tokenizer)
    test_dataset = preprocess_data(test_df, label_encoder, tokenizer)

    # Define desktop result path
    desktop_path = Path.home() / "Desktop" / "result"
    desktop_path.mkdir(parents=True, exist_ok=True)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(desktop_path / "bert_output"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="none"
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train with timing
    train_start = time.time()
    trainer.train()
    train_end = time.time()
    training_time = train_end - train_start
    print(f"Training Time: {training_time:.2f} seconds")

    # Inference with timing
    inference_start = time.time()
    predictions = trainer.predict(test_dataset)
    inference_end = time.time()
    inference_time = inference_end - inference_start
    print(f"Inference Time: {inference_time:.2f} seconds")

    # Compute metrics
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    metrics["training_time_sec"] = training_time
    metrics["inference_time_sec"] = inference_time
    metrics["trainable_parameters"] = trainable_params
    metrics["total_parameters"] = total_params

    # Predictions
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids

    # Save visuals + metrics
    save_bert_results_and_visuals(preds, labels, class_names, metrics, output_dir=str(desktop_path))

    print("✅ BERT stance detection completed and results saved to Desktop/result!")


if __name__ == "__main__":
    main()
