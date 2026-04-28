"""Audio utilities for Wav2Vec2-based models."""

import torch
import torch.nn as nn

from .distributed import unwrap_model

WAV2VEC2_STRIDE = 320
WAV2VEC2_SAMPLE_RATE = 16000
FRAME_DURATION = WAV2VEC2_STRIDE / WAV2VEC2_SAMPLE_RATE


def create_attention_mask(
    waveforms: torch.Tensor,
    audio_lengths: torch.Tensor,
) -> torch.Tensor:
    seq_len = waveforms.shape[1]
    clamped = torch.clamp(audio_lengths, max=seq_len)
    positions = torch.arange(seq_len, device=waveforms.device).unsqueeze(0)
    return (positions < clamped.unsqueeze(1)).long()


def enable_specaugment(model: nn.Module, enable: bool = True):
    raw = unwrap_model(model)
    if hasattr(raw, 'encoder') and hasattr(raw.encoder.wav2vec2, 'config'):
        raw.encoder.wav2vec2.config.apply_spec_augment = enable


def compute_output_lengths(
    model: nn.Module,
    input_lengths: torch.Tensor,
) -> torch.Tensor:
    raw = unwrap_model(model)
    return raw.encoder.wav2vec2._get_feat_extract_output_lengths(input_lengths)
