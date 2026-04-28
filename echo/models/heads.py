"""Output heads for phoneme classification (CTC)."""

import torch
import torch.nn as nn


class PhonemeHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear with blank bias init."""

    def __init__(self, input_dim: int, num_phonemes: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, input_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_phonemes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        with torch.no_grad():
            self.classifier.bias[0] = -1.0

    def forward(self, features):
        return self.classifier(self.dropout(self.activation(self.projection(features))))
