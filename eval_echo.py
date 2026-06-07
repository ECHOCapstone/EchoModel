"""ECHO 모델 (자체 학습 wav2vec2 체크포인트) 을 test.json 으로 평가.

eval_slplab.py 와 같은 metric 함수 + 같은 출력 schema 를 쓴다. 따라서 두 결과 JSON 을 그대로
diff / 표로 비교하면 모델별 PER · MDD F1 비교가 즉시 가능하다.

ECHO 는 자체 음소 인벤토리 (`data/phoneme_to_id.json`) 외 토큰을 출력하지 않으므로 sanitize 가
실제로는 무손실에 가깝지만, slplab 평가와 동일 파이프라인을 거치도록 sanitize 단계는 그대로 둔다.

사용:
    cd /home/syh/workspace/ECHO_model
    source venv/bin/activate
    python eval_echo.py \\
        --checkpoint experiments/.../best_mdd_f1.pth \\
        --phoneme-map data/phoneme_to_id.json \\
        --test-json data/test.json \\
        --device cuda \\
        --output analysis_output/echo_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from echo.constants import SILENCE_TOKENS
from echo.evaluation.metrics import calculate_mdd_metrics, calculate_sequence_error_rate
from echo.inference import PronunciationScorer
from echo.utils.phoneme_sanitize import sanitize_perceived


# slplab 평가와 같은 인벤토리 로더 — 두 스크립트가 동일한 표준 음소 집합을 본다.
def load_arpabet_inventory(phoneme_map_path: str) -> frozenset:
    with open(phoneme_map_path, "r", encoding="utf-8") as f:
        phoneme_map = json.load(f)
    return frozenset(
        k.lower() for k in phoneme_map.keys()
        if not k.startswith("<") and k.lower() not in SILENCE_TOKENS
    )


def load_audio(path: pathlib.Path, target_sr: int) -> torch.Tensor:
    waveform, sr = sf.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform), sr, target_sr
        ).numpy()
    return torch.from_numpy(waveform)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="ECHO 학습 체크포인트 (.pth) 경로.")
    parser.add_argument("--phoneme-map", default="data/phoneme_to_id.json")
    parser.add_argument("--test-json", default="data/test.json")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="analysis_output/echo_evaluation.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="디버그용. 처음 N 개 발화만 평가.")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root).resolve()
    test_json = pathlib.Path(args.test_json)
    with test_json.open() as f:
        manifest = json.load(f)

    items = [(uid, meta) for uid, meta in manifest.items()
             if meta.get("split") == "test" and meta.get("perceived_aligned")]
    if args.limit:
        items = items[: args.limit]
    print(f"평가 발화 수: {len(items)} (전체 {len(manifest)} 중 test split)")

    print(f"체크포인트 로드: {args.checkpoint} → {args.device}")
    scorer = PronunciationScorer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        phoneme_map_path=args.phoneme_map,
        device=args.device,
    )

    inventory = load_arpabet_inventory(args.phoneme_map)
    print(f"표준 음소 {len(inventory)} 개 — sanitize 후 PER 측정에 사용")

    all_ids, all_pred_raw, all_pred_sanitized, all_gt = [], [], [], []
    all_can_aligned, all_per_aligned = [], []
    errors_loading = 0

    start = time.time()
    for uid, meta in tqdm(items, desc="echo eval"):
        wav_rel = meta.get("wav")
        if not wav_rel:
            errors_loading += 1
            continue
        wav_path = data_root / wav_rel
        try:
            waveform = load_audio(wav_path, scorer.sampling_rate)
            raw_tokens = scorer.transcribe(waveform, keep_silence=False)
        except Exception as exc:
            errors_loading += 1
            print(f"[skip] {uid}: {exc}")
            continue

        sanitized = sanitize_perceived(raw_tokens, inventory)

        gt_aligned = meta["perceived_aligned"].split()
        all_ids.append(uid)
        all_pred_raw.append(raw_tokens)
        all_pred_sanitized.append(sanitized)
        all_gt.append(gt_aligned)
        all_can_aligned.append(meta.get("canonical_aligned", ""))
        all_per_aligned.append(meta.get("perceived_aligned", ""))

    elapsed = time.time() - start
    print(f"추론 완료: {elapsed/60:.1f} 분 (스킵 {errors_loading} 건)")

    # eval_slplab.py 와 같은 파이프라인 — silence 제거 후 raw / sanitized 양쪽 PER 측정.
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
        "model": "echo",
        "checkpoint": str(args.checkpoint),
        "test_json": str(test_json),
        "num_utterances": len(items),
        "skipped": errors_loading,
        "elapsed_sec": elapsed,
        "per": per,
        "raw_per": raw_per,
        "total_phonemes": total_ph,
        "total_errors": total_err,
        "raw_total_errors": raw_total_err,
        "mdd_f1": mdd_f1,
        "mdd": mdd,
    }

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print(f"PER (raw)       : {raw_per:.4f}")
    print(f"PER (sanitized) : {per:.4f}  ← 공정 비교 기준")
    print(f"MDD F1          : {mdd_f1:.4f}")
    print(f"저장됨          : {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
