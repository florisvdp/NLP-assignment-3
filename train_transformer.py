import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_LABELS = 4
MAX_LEN = 128
DEVICE = torch.device("cpu")  

dataset = load_dataset("ag_news")
train_split = dataset["train"].train_test_split(test_size=0.1, seed=19)
train_texts = list(train_split["train"]["text"])
train_labels = list(train_split["train"]["label"])
dev_texts = list(train_split["test"]["text"])
dev_labels = list(train_split["test"]["label"])
test_texts = list(dataset["test"]["text"])
test_labels = list(dataset["test"]["label"])

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

train_encodings = tokenize(train_texts)
dev_encodings = tokenize(dev_texts)
test_encodings = tokenize(test_texts)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds = NewsDataset(train_encodings, train_labels)
dev_ds = NewsDataset(dev_encodings, dev_labels)
test_ds = NewsDataset(test_encodings, test_labels)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=NUM_LABELS
)
model.to(DEVICE)

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_dir=f"{RESULTS_DIR}/logs",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    disable_tqdm=False,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_metrics
)

trainer.train()

preds_output = trainer.predict(test_ds)
test_preds = preds_output.predictions.argmax(-1)
test_labels_tensor = preds_output.label_ids

acc = accuracy_score(test_labels_tensor, test_preds)
macro_f1 = f1_score(test_labels_tensor, test_preds, average="macro")
print("\nFINAL TEST RESULTS")
print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

cm = confusion_matrix(test_labels_tensor, test_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["World","Sports","Business","Sci/Tech"],
            yticklabels=["World","Sports","Business","Sci/Tech"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/transformer_confusion_matrix.png")
plt.close()

misclassified = []
for text, true, pred in zip(test_texts, test_labels_tensor, test_preds):
    if true != pred:
        misclassified.append({
            "text": text,
            "true_label": ["World","Sports","Business","Sci/Tech"][true],
            "predicted_label": ["World","Sports","Business","Sci/Tech"][pred]
        })

df_mis = pd.DataFrame(misclassified)
df_mis.head(20).to_csv(f"{RESULTS_DIR}/transformer_misclassified_top20.csv", index=False)

print("Saved confusion matrix and top misclassified examples.")
