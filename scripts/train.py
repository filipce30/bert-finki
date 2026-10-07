"""Fine-tune bert-base-uncased on the preprocessed intent data.

    python scripts/train.py --config configs/training_config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_split(processed_dir, name, tokenizer, max_length):
    df = pd.read_csv(os.path.join(processed_dir, f"{name}.csv"))
    return IntentDataset(df["text"].tolist(), df["label"].tolist(), tokenizer, max_length)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["training"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    tokenizer = BertTokenizer.from_pretrained(model_cfg["name"])
    model = BertForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=model_cfg["num_labels"],
        classifier_dropout=model_cfg["classifier_dropout"],
    )

    max_length = data_cfg["max_seq_length"]
    train_ds = load_split(data_cfg["processed_dir"], "train", tokenizer, max_length)
    val_ds = load_split(data_cfg["processed_dir"], "val", tokenizer, max_length)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        eval_strategy=train_cfg["evaluation_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        save_total_limit=cfg["logging"]["save_total_limit"],
        logging_steps=cfg["logging"]["log_steps"],
        fp16=train_cfg["fp16"],
        seed=cfg["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    result = trainer.train()

    final_dir = os.path.join(train_cfg["output_dir"], "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved to {final_dir}")
    print(f"training loss: {result.training_loss:.4f}")

    for key, value in trainer.evaluate().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
