"""Build train/val/test splits from the Bitext customer support dataset.

On first run the dataset is downloaded from the Hugging Face Hub and saved
as a CSV under data/raw/. The text is then cleaned, intent names are mapped
to integer ids and stratified splits are written to data/processed/.

    python scripts/preprocess.py --config configs/training_config.yaml
"""

import argparse
import json
import os
import re

import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RAW_FILENAME = "bitext_customer_support.csv"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_raw(data_cfg):
    path = os.path.join(data_cfg["raw_dir"], RAW_FILENAME)
    if os.path.exists(path):
        return pd.read_csv(path)

    print(f"Downloading {data_cfg['dataset_name']}")
    ds = load_dataset(data_cfg["dataset_name"], split="train")
    df = ds.to_pandas()[["instruction", "intent", "category"]]
    df = df.rename(columns={"instruction": "text", "intent": "label"})
    os.makedirs(data_cfg["raw_dir"], exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()
    data_cfg = load_config(args.config)["data"]
    seed = data_cfg["random_seed"]

    df = load_raw(data_cfg)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    label_names = sorted(df["label"].unique())
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)
    print(f"{len(df)} samples, {len(label_names)} intents")

    train_df, rest = train_test_split(
        df,
        test_size=1 - data_cfg["train_split"],
        random_state=seed,
        stratify=df["label_id"],
    )
    test_frac = data_cfg["test_split"] / (data_cfg["val_split"] + data_cfg["test_split"])
    val_df, test_df = train_test_split(
        rest,
        test_size=test_frac,
        random_state=seed,
        stratify=rest["label_id"],
    )

    out_dir = data_cfg["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = split[["text", "label_id"]].rename(columns={"label_id": "label"})
        out.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        print(f"{name}: {len(out)}")

    with open(os.path.join(out_dir, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)


if __name__ == "__main__":
    main()
