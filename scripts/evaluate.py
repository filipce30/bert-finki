"""Evaluate a fine-tuned model on the test split.

Prints the classification report, saves the confusion matrix to
docs/figures/ and writes per-sample predictions for error analysis.

    python scripts/evaluate.py --config configs/training_config.yaml --model models/final_model
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import BertForSequenceClassification, BertTokenizer

FIGURE_PATH = "docs/figures/confusion_matrix.png"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model, tokenizer, texts, max_length, batch_size, device):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = {k: v[i : i + batch_size].to(device) for k, v in encodings.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.array(preds)


def plot_confusion_matrix(cm, label_names, path):
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--model", default="models/final_model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    processed_dir = data_cfg["processed_dir"]

    with open(os.path.join(processed_dir, "label_mapping.json")) as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    device = get_device()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertForSequenceClassification.from_pretrained(args.model).to(device).eval()

    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    preds = predict(
        model,
        tokenizer,
        test_df["text"].tolist(),
        data_cfg["max_seq_length"],
        cfg["training"]["eval_batch_size"],
        device,
    )
    y_true = test_df["label"].values

    print(classification_report(y_true, preds, target_names=label_names))
    print(f"accuracy:      {accuracy_score(y_true, preds):.4f}")
    print(f"f1 (macro):    {f1_score(y_true, preds, average='macro'):.4f}")
    print(f"f1 (weighted): {f1_score(y_true, preds, average='weighted'):.4f}")
    print(f"precision:     {precision_score(y_true, preds, average='macro'):.4f}")
    print(f"recall:        {recall_score(y_true, preds, average='macro'):.4f}")

    plot_confusion_matrix(confusion_matrix(y_true, preds), label_names, FIGURE_PATH)

    test_df["pred"] = preds
    test_df["true_label"] = test_df["label"].map(id2label)
    test_df["pred_label"] = test_df["pred"].map(id2label)
    test_df.to_csv(os.path.join(processed_dir, "test_predictions.csv"), index=False)

    errors = test_df[test_df["label"] != test_df["pred"]]
    print(f"\n{len(errors)} misclassified out of {len(test_df)}")
    for _, row in errors.iterrows():
        print(f"  {row['true_label']} -> {row['pred_label']}: {row['text'][:100]}")


if __name__ == "__main__":
    main()
