"""slplab L2-English 모델을 우리 test.json 으로 평가해 PER / MDD F1 을 측정한다.

ECHO 모델 평가(echo.evaluation.evaluator.ModelEvaluator) 와 같은 metric 함수
(calculate_sequence_error_rate, calculate_mdd_metrics) 를 재사용하므로 점수는
바로 비교 가능한 수치로 나온다.

사용:
    cd /home/syh/workspace/ECHO_model
    source venv/bin/activate    # 또는 본인 환경
    python eval_slplab.py \\
        --test-json data/test.json \\
        --hf-id slplab/wav2vec2-large-robust-L2-english-phoneme-recognition \\
        --device cuda \\
        --output analysis_output/slplab_evaluation.json

출력:
    JSON 한 건이 analysis_output/ 에 저장되고 콘솔에 요약 (PER / MDD F1) 이 찍힌다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from echo.constants import SILENCE_TOKENS
from echo.evaluation.metrics import calculate_mdd_metrics, calculate_sequence_error_rate


def load_audio(path: pathlib.Path, target_sr: int = 16000) -> torch.Tensor:
    waveform, sr = sf.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform), sr, target_sr
        ).numpy()
    return torch.from_numpy(waveform)


def recognize(model, processor, waveform: torch.Tensor, device: str) -> list[str]:
    inputs = processor(
        waveform.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0].strip()
    if not text:
        return []
    return [p.replace("_err", "") for p in text.split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", default="data/test.json")
    parser.add_argument("--data-root", default=".",
                        help="test.json 의 wav 상대경로의 기준 디렉터리.")
    parser.add_argument("--hf-id",
                        default="slplab/wav2vec2-large-robust-L2-english-phoneme-recognition")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="analysis_output/slplab_evaluation.json")
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

    print(f"모델 로드: {args.hf_id} → {args.device}")
    processor = Wav2Vec2Processor.from_pretrained(args.hf_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.hf_id).to(args.device).eval()

    all_ids, all_pred, all_gt = [], [], []
    all_can_aligned, all_per_aligned = [], []
    errors_loading = 0

    start = time.time()
    for uid, meta in tqdm(items, desc="slplab eval"):
        wav_rel = meta.get("wav")
        if not wav_rel:
            errors_loading += 1
            continue
        wav_path = data_root / wav_rel
        try:
            waveform = load_audio(wav_path)
            perceived = recognize(model, processor, waveform, args.device)
        except Exception as exc:
            errors_loading += 1
            print(f"[skip] {uid}: {exc}")
            continue

        gt_aligned = meta["perceived_aligned"].split()
        all_ids.append(uid)
        all_pred.append(perceived)
        all_gt.append(gt_aligned)
        all_can_aligned.append(meta.get("canonical_aligned", ""))
        all_per_aligned.append(meta.get("perceived_aligned", ""))

    elapsed = time.time() - start
    print(f"추론 완료: {elapsed/60:.1f} 분 (스킵 {errors_loading} 건)")

    # ECHO evaluator 와 동일하게 silence 제거 후 PER 계산.
    pred_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_pred]
    gt_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_gt]

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
        "model": args.hf_id,
        "test_json": str(test_json),
        "num_utterances": len(items),
        "skipped": errors_loading,
        "elapsed_sec": elapsed,
        "per": per,
        "total_phonemes": total_ph,
        "total_errors": total_err,
        "mdd_f1": mdd_f1,
        "mdd": mdd,
    }

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print(f"PER     : {per:.4f}")
    print(f"MDD F1  : {mdd_f1:.4f}")
    print(f"저장됨  : {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
