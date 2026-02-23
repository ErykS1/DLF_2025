import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import re

SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class SubmissionModel(nn.Module):
    def __init__(self):
        super(SubmissionModel, self).__init__()

        self.vocab = self._load_vocab()
        self.vocab_size = len(self.vocab) if self.vocab else 10000
        self.max_len = 40

        # self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = models.resnet50(weights=None)
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.embedding = nn.Embedding(self.vocab_size, 128, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.4)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.text_proj = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _load_vocab(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(current_dir, 'vocab.pt')
        try:
            return torch.load(vocab_path, weights_only=False)
        except FileNotFoundError:
            print(f"WARNING: File {vocab_path} not found. Bad things happened!")
            return {'<pad>': 0, '<unk>': 1}

    def forward(self, images, tokens):
        img_features = self.cnn(images)

        embedded = self.embedding(tokens)
        embedded = self.embedding_dropout(embedded)
        output, _ = self.lstm(embedded)

        text_features = output.mean(dim=1)
        text_features = self.text_proj(text_features)

        combined = img_features * text_features

        prob = self.classifier(combined)
        return prob.squeeze(1)

    def predict(self, image_tensor, text_string):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                image_batch = image_tensor.unsqueeze(0).to(device)
            else:
                image_batch = image_tensor.to(device)

            text = text_string.lower()
            text = re.sub(r"[^a-z0-9\s]", "", text)
            words = text.split()

            indices = [self.vocab.get(w, self.vocab.get('<unk>', 1)) for w in words[:self.max_len]]
            if not indices:
                indices = [self.vocab.get('<pad>', 0)]

            if len(indices) < self.max_len:
                indices += [self.vocab.get('<pad>', 0)] * (self.max_len - len(indices))

            tokens = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

            score = self.forward(image_batch, tokens)
            return score.item()