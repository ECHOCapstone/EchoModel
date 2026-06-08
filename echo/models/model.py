"""Baseline pronunciation assessment model (perceived-only CTC).

Architecture:
  Audio -> Wav2VecEncoder -> BaseEncoder -> PhonemeHead -> logits [B, T, 42]
"""

import torch.nn as nn

from .encoders import Wav2VecEncoder, BaseEncoder
from .heads import PhonemeHead


class BaselineModel(nn.Module):
    """Perceived-only baseline: Wav2Vec2 + CTC phoneme head."""

    # 추론 시 canonical 음소열을 입력으로 요구하는지 여부. baseline 은 음성만으로 인식하므로 False.
    requires_canonical: bool = False

    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-base",
        hidden_dim: int = 768,
        num_phonemes: int = 42,
        dropout: float = 0.1,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
    ):
        super().__init__()
        self.encoder = Wav2VecEncoder(
            pretrained_model_name,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
            mask_feature_prob=mask_feature_prob,
            mask_feature_length=mask_feature_length,
        )
        self.feature_encoder = BaseEncoder(hidden_dim, hidden_dim, dropout)
        self.perceived_head = PhonemeHead(hidden_dim, num_phonemes, dropout)

    @classmethod
    def from_config(cls, config):
        return cls(
            pretrained_model_name=config.pretrained_model,
            hidden_dim=config.hidden_dim,
            num_phonemes=config.num_phonemes,
            dropout=config.dropout,
            mask_time_prob=config.mask_time_prob,
            mask_time_length=config.mask_time_length,
            mask_feature_prob=config.mask_feature_prob,
            mask_feature_length=config.mask_feature_length,
        )

    def encoder_parameters(self):
        """사전학습 음향 인코더(wav2vec2 등) 파라미터. Trainer 가 별도 LR 그룹으로 묶는다."""
        return self.encoder.parameters()

    def head_parameters(self):
        """사전학습 인코더 이외(feature encoder + phoneme head) 파라미터."""
        yield from self.feature_encoder.parameters()
        yield from self.perceived_head.parameters()

    def forward(self, waveform, attention_mask=None, **kwargs):
        features = self.feature_encoder(self.encoder(waveform, attention_mask))
        return {'perceived_logits': self.perceived_head(features)}
