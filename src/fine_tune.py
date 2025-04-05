import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# Laden der Trainings- und Testdaten
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

# Konvertieren der Daten in das Hugging Face Dataset-Format
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Initialisieren des Tokenizers
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Funktion zum Tokenisieren der Daten
def tokenize_function(examples):
    return tokenizer(examples["review"], padding="max_length", truncation=True)

# Tokenisieren der Trainings- und Testdaten
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Labels zu numerischen Werten konvertieren
label_map = {"positive": 1, "negative": 0}
train_dataset = train_dataset.map(lambda x: {"labels": label_map[x["sentiment"]]})
test_dataset = test_dataset.map(lambda x: {"labels": label_map[x["sentiment"]]})

# Initialisieren des Modells
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# TrainingArguments definieren
training_args = TrainingArguments(
    output_dir="../models/bert-finetuned",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="../logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, p.predictions.argmax(axis=1))},
)

# Modell trainieren
trainer.train()

# Modell evaluieren
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Modell speichern
trainer.save_model("../models/bert-finetuned")
tokenizer.save_pretrained("../models/bert-finetuned")