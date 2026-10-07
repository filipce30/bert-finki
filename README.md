# BERT fine-tuning for customer support intent classification

Filip Filipovski, FINKI

Fine-tuning `bert-base-uncased` to classify short customer support messages
into 27 intents (cancel_order, get_refund, recover_password, ...). Trained on
the [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
dataset (26,872 messages, 27 intents, 11 categories).

## Results

Test split, 2,688 messages never seen during training or validation:

| Metric | Value |
|---|---|
| Accuracy | 99.89% |
| F1 (macro) | 0.9989 |
| F1 (weighted) | 0.9989 |
| Misclassified | 3 / 2,688 |

All 27 intents have F1 >= 0.99. Only 3 test messages are wrong: check_invoice
predicted as get_invoice, set_up_shipping_address as change_shipping_address,
and a misspelled complaint ("compoaint") predicted as place_order. Training
took 42 minutes for 4 epochs on Apple Silicon (MPS backend).

Confusion matrix: `docs/figures/confusion_matrix.png`

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.11, Transformers 5.4, scikit-learn 1.8.

## Running

```bash
# download the dataset, clean it and create stratified 80/10/10 splits
python scripts/preprocess.py --config configs/training_config.yaml

# fine-tune (4 epochs, lr 2e-5, batch 16, best epoch by macro F1)
python scripts/train.py --config configs/training_config.yaml

# evaluate on the test split, save confusion matrix and predictions
python scripts/evaluate.py --config configs/training_config.yaml --model models/final_model
```

All hyperparameters are in `configs/training_config.yaml`. The random seed is
fixed to 42, so the splits and the training run are reproducible. The trained
model (~440 MB) is not committed; `train.py` recreates it under `models/final_model`.

## Structure

```
configs/training_config.yaml   hyperparameters and paths
scripts/preprocess.py          download, clean, encode labels, split
scripts/train.py               fine-tuning with the Hugging Face Trainer
scripts/evaluate.py            test metrics, confusion matrix, error list
data/processed/label_mapping.json   intent name <-> id
docs/figures/confusion_matrix.png
```
