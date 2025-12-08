import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import no_grad
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
import string
from torchinfo import summary

#%%
DATA_PATH = "lab_7/datasets/illegal_surveilance_tagged.csv"
SEQUENCE_LENGTH = 50
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
MAX_VOCAB_SIZE = 50000
DROPOUT = 0.5

#%%
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Please place 'illegal_surveilance_tagged.csv' at the path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded with {len(df)} entries.")

df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

#%%
all_tokens = [token for review in df['sentence'] for token in clean_and_tokenize(review)]
token_counts = Counter(all_tokens)
print(f"Total unique tokens: {len(token_counts)}")

#%%
top_words = token_counts.most_common(MAX_VOCAB_SIZE - 2)

vocab = {word: i + 2 for i, (word, count) in enumerate(top_words)}
vocab['<pad>'] = 0
vocab['<unk>'] = 1
VOCAB_SIZE = len(vocab)
print(f"Vocabulary built with size: {VOCAB_SIZE}")

def text_to_sequence(text, vocab, max_len):
    """Converts a review text to a padded sequence of indices using the clean tokenizer."""
    sequence = [vocab.get(word, vocab['<unk>']) for word in clean_and_tokenize(text)]

    if len(sequence) < max_len:
        sequence.extend([vocab['<pad>']] * (max_len - len(sequence)))
    elif len(sequence) > max_len:
        sequence = sequence[:max_len]

    return sequence

df['sequence'] = df['sentence'].apply(lambda x: text_to_sequence(x, vocab, SEQUENCE_LENGTH))

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

#%%
class PolishDataset(Dataset):
    """Custom Dataset for IMDB sequences and labels."""
    def __init__(self, data_frame):
        self.sequences = torch.tensor(data_frame['sequence'].tolist(), dtype=torch.long)
        self.labels = torch.tensor(data_frame['sentiment'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class SentimentDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the IMDB dataset."""
    def __init__(self, train_df, val_df, batch_size):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = PolishDataset(self.train_df)
        self.val_dataset = PolishDataset(self.val_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

polish_dm = SentimentDataModule(train_df, val_df, BATCH_SIZE)

#%%
class SentimentLSTM(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, learning_rate, dropout):
        super().__init__()
        self.save_hyperparameters()

        # 1. Embedding Layer: Converts word indices to dense vectors
        # Input: (batch, seq_len) -> Output: (batch, seq_len, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Recurrent Layer: LSTM for sequence processing
        # `batch_first=True` means inputs/outputs are (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 3. Classifier Layer: Takes the final hidden state to predict sentiment
        # We need a linear layer with output size 1 for binary classification
        # self.classifier = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy()

    def forward(self, x):
        # x shape: (batch_size, SEQUENCE_LENGTH) - word indices

        # Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch_size, SEQUENCE_LENGTH, EMBEDDING_DIM)

        # LSTM Pass
        # `_` is for the output over all steps (y_t) which we ignore here for classification
        # `(h_n, c_n)` contains the final hidden state (h_n) and cell state (c_n)
        _, (h_n, c_n) = self.lstm(embedded)

        # h_n.shape → (num_layers × num_directions, batch_size, hidden_size)
        final_hidden_state_forward = h_n[-2, :, :]
        final_hidden_state_backward = h_n[-1, :, :]

        combined_final_hidden_state = torch.cat((final_hidden_state_forward, final_hidden_state_backward), dim=1)

        logits = self.classifier(combined_final_hidden_state)
        return logits

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)

        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(logits, labels.long()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(logits, labels.long()), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

#%%
model = SentimentLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    learning_rate=LEARNING_RATE,
    dropout=DROPOUT
)
print("\nLSTM Model instantiated.")
print(model)

#%%
device = next(model.parameters()).device
print(f"Model is currently on device: {device}")

INPUT_TENSOR = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.long, device=device)


summary(model, input_data=INPUT_TENSOR)

#%%
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    logger=True,
    log_every_n_steps=50,
    enable_progress_bar=True
)

#%%
print(f"\nStarting training for {trainer.max_epochs} epochs...")

trainer.fit(model, datamodule=polish_dm)

print("\nTraining complete!")

#%%
DATA_PATH_UNTAGGED = "lab_7/datasets/illegal_surveilance_test_untagged.csv"

if not os.path.exists(DATA_PATH_UNTAGGED):
    raise FileNotFoundError(f"Please place 'illegal_surveilance_test_untagged.csv' at the path: {DATA_PATH_UNTAGGED}")

df_new = pd.read_csv(DATA_PATH_UNTAGGED, header=None, names=["sentence"])
print(f"Dataset loaded with {len(df_new)} entries.")

#%%
df_new['sequence'] = df_new['sentence'].apply(lambda x: text_to_sequence(x, vocab, SEQUENCE_LENGTH))

#%%
class UntaggedDataset(Dataset):
    def __init__(self, data_frame):
        self.sequences = torch.tensor(data_frame['sequence'].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

untagged_loader = DataLoader(UntaggedDataset(df_new), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#%%
model.eval()

all_probs = []
with torch.no_grad():
    for batch in untagged_loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu()
        all_probs.extend(probs.flatten().tolist())

#%%
df_new['probability_positive'] = all_probs
df_new['predicted_sentiment'] = (df_new['probability_positive'] > 0.5).map({True: 'positive', False: 'negative'})

#%%
OUTPUT_PATH = "lab_7/datasets/illegal_surveilance_predicted.csv"
df_new[['sentence', 'predicted_sentiment', 'probability_positive']].to_csv(OUTPUT_PATH, index=False)

#%%
