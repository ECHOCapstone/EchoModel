"""eval_echo / eval_slplab 두 스크립트가 공유하는 평가 루프.

ECHO 체크포인트와 HuggingFace 체크포인트는 모델 객체 형태만 다를 뿐 평가 파이프라인
(음성 로드 → recognize → sanitize → PER/MDD 집계) 은 동일하다. recognize 만 콜백으로 받고
나머지는 한곳에서 처리해 두 스크립트의 중복을 줄인다.
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Callable, Dict, FrozenSet, List, Optional

import soundfile as sf
import torch

from ..constants import SILENCE_TOKENS
from ..utils.audio import WAV2VEC2_SAMPLE_RATE
from ..utils.phoneme_sanitize import sanitize_perceived
from .metrics import calculate_mdd_metrics, calculate_sequence_error_rate


# manifest 한 발화 → 음소 시퀀스. recognize 콜백은 waveform 텐서를 받아 raw 토큰을 돌려준다.
RecognizeFn = Callable[[torch.Tensor], List[str]]


def _load_audio(path: pathlib.Path, target_sr: int) -> torch.Tensor:
    """16kHz 1D float32 텐서로 정규화. 다채널은 평균, 다른 sr 은 torchaudio 로 리샘플."""
    waveform, sr = sf.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform), sr, target_sr
        ).numpy()
    return torch.from_numpy(waveform)


def load_arpabet_inventory(phoneme_map_path: str) -> FrozenSet[str]:
    """phoneme_to_id.json 에서 silence/blank/특수 토큰을 제외한 표준 음소 집합."""
    with open(phoneme_map_path, "r", encoding="utf-8") as f:
        phoneme_map = json.load(f)
    return frozenset(
        k.lower() for k in phoneme_map.keys()
        if not k.startswith("<") and k.lower() not in SILENCE_TOKENS
    )


def run_evaluation(
    recognize_fn: RecognizeFn,
    manifest_path: pathlib.Path,
    inventory: FrozenSet[str],
    sampling_rate: int,
    output_path: pathlib.Path,
    model_metadata: Dict,
    data_root: pathlib.Path,
    limit: Optional[int] = None,
    progress_label: str = "eval",
) -> Dict:
    """test split 발화 전체를 평가하고 PER / MDD F1 요약을 JSON 으로 저장 + 반환한다."""
    from tqdm import tqdm

    with manifest_path.open() as f:
        manifest = json.load(f)

    items = [
        (uid, meta) for uid, meta in manifest.items()
        if meta.get("split") == "test" and meta.get("perceived_aligned")
    ]
    if limit:
        items = items[:limit]
    print(f"평가 발화 수: {len(items)} (전체 {len(manifest)} 중 test split)")

    all_ids: List[str] = []
    all_pred_raw: List[List[str]] = []
    all_pred_sanitized: List[List[str]] = []
    all_gt: List[List[str]] = []
    all_can_aligned: List[str] = []
    all_per_aligned: List[str] = []
    skipped = 0

    start = time.time()
    for uid, meta in tqdm(items, desc=progress_label):
        wav_rel = meta.get("wav")
        if not wav_rel:
            skipped += 1
            continue
        wav_path = data_root / wav_rel
        try:
            waveform = _load_audio(wav_path, sampling_rate)
            raw_tokens = recognize_fn(waveform)
        except Exception as exc:
            skipped += 1
            print(f"[skip] {uid}: {exc}")
            continue

        sanitized = sanitize_perceived(raw_tokens, inventory)
        all_ids.append(uid)
        all_pred_raw.append(raw_tokens)
        all_pred_sanitized.append(sanitized)
        all_gt.append(meta["perceived_aligned"].split())
        all_can_aligned.append(meta.get("canonical_aligned", ""))
        all_per_aligned.append(meta.get("perceived_aligned", ""))

    elapsed = time.time() - start
    num_evaluated = len(all_ids)
    print(f"추론 완료: {elapsed/60:.1f} 분 (평가 {num_evaluated} / 스킵 {skipped})")

    # silence 제거 후 raw / sanitized PER 양쪽 계산.
    raw_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_pred_raw]
    pred_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_pred_sanitized]
    gt_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_gt]

    raw_details = calculate_sequence_error_rate(all_ids, gt_clean, raw_clean)
    raw_total_ph = sum(d["num_reference_tokens"] for d in raw_details)
    raw_total_err = sum(d["substitutions"] + d["deletions"] + d["insertions"] for d in raw_details)
    raw_per = raw_total_err / raw_total_ph if raw_total_ph > 0 else 0.0

    per_details = calculate_sequence_error_rate(all_ids, gt_clean, pred_clean)
    total_ph = sum(d["num_reference_tokens"] for d in per_details)
    total_err = sum(d["substitutions"] + d["deletions"] + d["insertions"] for d in per_details)
    per = total_err / total_ph if total_ph > 0 else 0.0

    can_parsed = [s.split() if s else [] for s in all_can_aligned]
    per_parsed = [s.split() if s else [] for s in all_per_aligned]
    has_aligned = (
        len(can_parsed) == len(pred_clean)
        and all(len(c) > 0 for c in can_parsed)
        and all(len(p) > 0 for p in per_parsed)
    )
    if has_aligned:
        mdd = calculate_mdd_metrics(pred_clean, can_parsed, per_parsed)
        mdd_f1 = mdd["f1_score"]
    else:
        mdd = None
        mdd_f1 = 0.0

    summary = {
        **model_metadata,
        "test_json": str(manifest_path),
        # 평가 가능한 utterance 수와 스킵 수를 분리해 노출 — len(items) 와 헷갈리지 않게.
        "num_utterances": num_evaluated,
        "num_total_candidates": len(items),
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "per": per,
        "raw_per": raw_per,
        "total_phonemes": total_ph,
        "total_errors": total_err,
        "raw_total_errors": raw_total_err,
        "mdd_f1": mdd_f1,
        "mdd": mdd,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print(f"PER (raw)       : {raw_per:.4f}")
    print(f"PER (sanitized) : {per:.4f}  ← 공정 비교 기준")
    print(f"MDD F1          : {mdd_f1:.4f}")
    print(f"저장됨          : {output_path}")
    print("=" * 50)

    return summary


__all__ = [
    "RecognizeFn",
    "WAV2VEC2_SAMPLE_RATE",
    "load_arpabet_inventory",
    "run_evaluation",
]
