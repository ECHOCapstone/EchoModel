"""FiLM-conditioned pronunciation assessment model.

Architecture:
  Audio -> Wav2VecEncoder -> BaseEncoder -> h [B, T, H]
  Canonical phonemes -> FiLMGen -> (gamma, beta) [B, H]
  h' = (1 + gamma) * h + beta   (broadcast over T)
  h' -> PhonemeHead -> CTC

Identity init: gamma=0, beta=0 → h' = h, so training starts from baseline.
"""

import torch
import torch.nn as nn

from .encoders import Wav2VecEncoder, BaseEncoder
from .heads import PhonemeHead


class FiLMGen(nn.Module):
    """Generates global FiLM parameters from canonical phoneme sequence.

    canonical_ids -> embedding -> masked mean pool -> Linear -> (gamma, beta)
    """

    def __init__(
        self,
        num_phonemes: int,
        hidden_dim: int,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_phonemes, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, hidden_dim * 2)
        self._init_identity()

    def _init_identity(self):
        # Output zeros so that (1 + gamma) = 1 and beta = 0 at init.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        canonical_ids: torch.Tensor,
        canonical_lengths: torch.Tensor,
    ):
        """Generates per-utterance FiLM params.

        Args:
          canonical_ids: [B, L] padded phoneme IDs.
          canonical_lengths: [B] valid phoneme counts.

        Returns:
          gamma, beta: each [B, hidden_dim].
        """
        emb = self.embedding(canonical_ids)  # [B, L, E]

        max_len = emb.size(1)
        mask = (
            torch.arange(max_len, device=emb.device).unsqueeze(0)
            < canonical_lengths.unsqueeze(1)
        ).float().unsqueeze(-1)  # [B, L, 1]

        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, E]
        gamma_beta = self.proj(pooled)  # [B, 2H]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta


class FiLMModel(nn.Module):
    """Wav2Vec2 + global FiLM conditioning on canonical phonemes."""

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
        film_embed_dim: int = 128,
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
        self.film_gen = FiLMGen(num_phonemes, hidden_dim, embed_dim=film_embed_dim)
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
            film_embed_dim=config.film_embed_dim,
        )

    def forward(
        self,
        waveform,
        attention_mask=None,
        canonical_labels=None,
        canonical_lengths=None,
        **kwargs,
    ):
        h = self.feature_encoder(self.encoder(waveform, attention_mask))  # [B, T, H]

        if canonical_labels is not None and canonical_lengths is not None:
            gamma, beta = self.film_gen(canonical_labels, canonical_lengths)
            h = (1.0 + gamma).unsqueeze(1) * h + beta.unsqueeze(1)

        return {'perceived_logits': self.perceived_head(h)}
