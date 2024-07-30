import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader, Dataset

# Load the dataset
df = pd.read_csv('dataset.csv', sep=';')

# Basic preprocessing
df['description'] = df['description'].fillna('')

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(df['description'], df['fraudulent'],
                                                                      test_size=0.2, random_state=42)


# Define a custom dataset class
class JobPostingsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        return {key: val.squeeze() for key, val in encoding.items()}, torch.tensor(label)


# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = JobPostingsDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
test_dataset = JobPostingsDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('model')
tokenizer.save_pretrained('model')
