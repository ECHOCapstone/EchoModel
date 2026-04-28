"""Wav2Vec2/HuBERT encoder and feature processing layer."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class Wav2VecEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-base",
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.mask_time_prob = mask_time_prob
        config.mask_time_length = mask_time_length
        config.mask_feature_prob = mask_feature_prob
        config.mask_feature_length = mask_feature_length
        config.layerdrop = 0.0

        self.wav2vec2 = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self._fix_masked_spec_embed()

    def _fix_masked_spec_embed(self):
        if hasattr(self.wav2vec2, 'masked_spec_embed'):
            embed = self.wav2vec2.masked_spec_embed
            if torch.isnan(embed).any() or embed.abs().max() > 1e10:
                nn.init.uniform_(self.wav2vec2.masked_spec_embed)

    def forward(self, waveform, attention_mask=None):
        return self.wav2vec2(waveform, attention_mask=attention_mask).last_hidden_state


class BaseEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        return self.dropout(self.layer_norm(features))
