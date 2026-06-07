"""Inference & pronunciation diagnosis for backend integration.

Usage:
    scorer = PronunciationScorer.from_checkpoint(
        "experiments/wav2vec2-base/baseline/20260411_030102/checkpoints/best_mdd_f1.pth",
        "data/phoneme_to_id.json",
    )
    result = scorer.score("path/to/audio.wav", canonical="hh eh l ow")
    # result = {
    #   "recognized":   ["hh", "eh", "l", "ow"],
    #   "canonical":    ["hh", "eh", "l", "ow"],
    #   "peak_softmax": [0.97, 0.93, 0.88, 0.91],
    #   "alignment":    [...per-phoneme ops...],
    #   "errors":       [...only error ops...],
    #   "per":          0.0,
    # }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import logging

import soundfile as sf
import torch
import torchaudio

from .constants import BLANK_TOKEN_ID, SILENCE_TOKENS, infer_num_phonemes
from .evaluation.metrics import get_alignment_with_indices
from .models.film_model import FiLMModel
from .models.model import BaselineModel
from .utils.audio import WAV2VEC2_SAMPLE_RATE, create_attention_mask


logger = logging.getLogger(__name__)


AudioInput = Union[str, torch.Tensor, "np.ndarray"]  # noqa: F821


# 한 발화에 대한 인식 결과 묶음. peak_softmax는 collapsed 후 각 음소의 confidence (max).
@dataclass
class ScoreResult:
    recognized: List[str]
    canonical: Optional[List[str]]
    peak_softmax: List[float] = field(default_factory=list)
    alignment: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    per: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "recognized": self.recognized,
            "canonical": self.canonical,
            "peak_softmax": self.peak_softmax,
            "alignment": self.alignment,
            "errors": self.errors,
            "per": self.per,
        }


class PronunciationScorer:
    """Loads a trained ECHO checkpoint and scores utterances."""

    def __init__(
        self,
        model: torch.nn.Module,
        phoneme_to_id: Dict[str, int],
        id_to_phoneme: Dict[int, str],
        device: Union[str, torch.device] = "cuda",
        sampling_rate: int = WAV2VEC2_SAMPLE_RATE,
        blank_id: int = BLANK_TOKEN_ID,
    ):
        # 요청 device 가 cuda 인데 cuda 가 없으면 cpu 로 떨어진다 — silent migration 이 아니라
        # 명시적으로 경고 로그를 남겨 운영자가 GPU 부재를 알아챌 수 있게 한다.
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU for PronunciationScorer.")
            self.device = torch.device("cpu")
        else:
            self.device = requested
        self.model = model.to(self.device).eval()
        self.phoneme_to_id = phoneme_to_id
        self.id_to_phoneme = id_to_phoneme
        self.sampling_rate = sampling_rate
        self.blank_id = blank_id
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        phoneme_map_path: str,
        model_type: str = "baseline",
        pretrained_model: str = "facebook/wav2vec2-base",
        device: str = "cuda",
    ) -> "PronunciationScorer":
        with open(phoneme_map_path, "r", encoding="utf-8") as f:
            phoneme_to_id: Dict[str, int] = json.load(f)
        id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_cfg = ckpt.get("config", {}) or {}
        # saved_cfg 의 num_phonemes 를 1순위로 신뢰한다 — 구 체크포인트가 다른 vocab 크기로 학습됐을
        # 수 있어 phoneme_to_id 길이로 임의 덮어쓰면 weight shape 가 맞지 않아 load 가 실패한다.
        # saved_cfg 에 없을 때만 phoneme_to_id 길이로 보강. 그것도 없으면 명시적으로 raise.
        if "num_phonemes" in saved_cfg:
            num_phonemes = saved_cfg["num_phonemes"]
        elif phoneme_to_id:
            num_phonemes = infer_num_phonemes(phoneme_to_id)
            logger.warning(
                "checkpoint %s has no num_phonemes in saved config; inferring %d from phoneme map %s",
                checkpoint_path, num_phonemes, phoneme_map_path,
            )
        else:
            raise ValueError(
                f"checkpoint {checkpoint_path} is missing num_phonemes and phoneme map is empty"
            )
        hidden_dim = saved_cfg.get("hidden_dim", 768)
        dropout = saved_cfg.get("dropout", 0.1)
        pretrained = saved_cfg.get("pretrained_model", pretrained_model)
        mtype = saved_cfg.get("model_type", model_type)

        if mtype == "film":
            model = FiLMModel(
                pretrained_model_name=pretrained,
                hidden_dim=hidden_dim,
                num_phonemes=num_phonemes,
                dropout=dropout,
                film_embed_dim=saved_cfg.get("film_embed_dim", 128),
            )
        else:
            model = BaselineModel(
                pretrained_model_name=pretrained,
                hidden_dim=hidden_dim,
                num_phonemes=num_phonemes,
                dropout=dropout,
            )

        model.load_state_dict(ckpt["model_state_dict"])
        if hasattr(model.encoder.wav2vec2, "config"):
            model.encoder.wav2vec2.config.apply_spec_augment = False

        return cls(
            model=model,
            phoneme_to_id=phoneme_to_id,
            id_to_phoneme=id_to_phoneme,
            device=device,
            sampling_rate=saved_cfg.get("sampling_rate", WAV2VEC2_SAMPLE_RATE),
        )

    def _load_waveform(self, audio: AudioInput) -> torch.Tensor:
        if isinstance(audio, str):
            data, sr = sf.read(audio, dtype="float32")
            wav = torch.from_numpy(data)
            if wav.ndim == 2:
                wav = wav.mean(dim=1)
            if sr != self.sampling_rate:
                if sr not in self._resamplers:
                    self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sampling_rate)
                wav = self._resamplers[sr](wav.unsqueeze(0)).squeeze(0)
            return wav

        if isinstance(audio, torch.Tensor):
            wav = audio.detach().float().cpu()
        else:
            wav = torch.as_tensor(audio, dtype=torch.float32)
        if wav.ndim == 2:
            wav = wav.mean(dim=0 if wav.shape[0] < wav.shape[1] else 1)
        return wav

    @torch.no_grad()
    def _forward(self, audio: AudioInput) -> torch.Tensor:
        # 모델 추론 후 유효 프레임 길이만큼 자른 [T, V] logits 반환.
        waveform = self._load_waveform(audio).to(self.device)
        audio_len = torch.tensor([waveform.shape[0]], device=self.device)
        batch = waveform.unsqueeze(0)
        mask = create_attention_mask(batch, audio_len)

        outputs = self.model(batch, attention_mask=mask)
        logits = outputs["perceived_logits"]

        output_lengths = self.model.encoder.wav2vec2._get_feat_extract_output_lengths(audio_len)
        valid_T = int(torch.clamp(output_lengths, min=1, max=logits.size(1))[0].item())
        return logits[0, :valid_T]

    def _decode(
        self,
        frame_logits: torch.Tensor,
        keep_silence: bool,
    ) -> tuple[List[str], List[float]]:
        # CTC argmax + blank/duplicate collapse. 같은 라벨의 연속 프레임은
        # 묶이고, 묶음 내 최대 softmax 값을 그 음소의 peak로 사용한다.
        probs = frame_logits.softmax(dim=-1)
        peaks_per_frame, preds_per_frame = probs.max(dim=-1)
        preds = preds_per_frame.tolist()
        peaks = peaks_per_frame.tolist()

        ids: List[int] = []
        peak_values: List[float] = []
        prev = -1
        for pid, pk in zip(preds, peaks):
            if pid != self.blank_id and pid != prev:
                ids.append(pid)
                peak_values.append(float(pk))
            elif pid == prev and pid != self.blank_id and peak_values:
                if pk > peak_values[-1]:
                    peak_values[-1] = float(pk)
            prev = pid

        phonemes = [self.id_to_phoneme.get(pid, f"UNK_{pid}") for pid in ids]
        if not keep_silence:
            kept = [(ph, pk) for ph, pk in zip(phonemes, peak_values) if ph.lower() not in SILENCE_TOKENS]
            phonemes = [ph for ph, _ in kept]
            peak_values = [pk for _, pk in kept]
        return phonemes, peak_values

    def transcribe(self, audio: AudioInput, keep_silence: bool = False) -> List[str]:
        """Audio -> decoded phoneme sequence."""
        logits = self._forward(audio)
        phonemes, _ = self._decode(logits, keep_silence)
        return phonemes

    def diagnose(
        self,
        recognized: Sequence[str],
        canonical: Sequence[str],
    ) -> Dict:
        """Compare recognized phonemes vs canonical reference, return per-phoneme ops."""
        ref = [p for p in canonical if p.lower() not in SILENCE_TOKENS]
        hyp = [p for p in recognized if p.lower() not in SILENCE_TOKENS]

        align = get_alignment_with_indices(ref, hyp)
        op_label = {"=": "correct", "S": "substitution", "D": "deletion", "I": "insertion"}

        alignment: List[Dict] = []
        errors: List[Dict] = []
        for op, ref_idx, hyp_idx in align:
            item = {
                "op": op_label[op],
                "canonical_index": ref_idx,
                "canonical": ref[ref_idx] if ref_idx is not None else None,
                "recognized_index": hyp_idx,
                "recognized": hyp[hyp_idx] if hyp_idx is not None else None,
            }
            alignment.append(item)
            if op != "=":
                errors.append(item)

        num_errors = sum(1 for op, _, _ in align if op != "=")
        per = num_errors / len(ref) if ref else None
        return {"alignment": alignment, "errors": errors, "per": per}

    def score(
        self,
        audio: AudioInput,
        canonical: Optional[Union[str, Sequence[str]]] = None,
        keep_silence: bool = False,
    ) -> Dict:
        """End-to-end: audio -> recognized phonemes + peak softmax -> (optional) error diagnosis."""
        logits = self._forward(audio)
        recognized, peaks = self._decode(logits, keep_silence)

        can_list: Optional[List[str]] = None
        if canonical is not None:
            can_list = canonical.split() if isinstance(canonical, str) else list(canonical)

        result = ScoreResult(recognized=recognized, canonical=can_list, peak_softmax=peaks)
        if can_list is not None:
            diag = self.diagnose(recognized, can_list)
            result.alignment = diag["alignment"]
            result.errors = diag["errors"]
            result.per = diag["per"]
        return result.to_dict()
