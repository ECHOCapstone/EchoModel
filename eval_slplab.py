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
import pathlib

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from echo.constants import DEFAULT_PHONEME_MAP_PATH
from echo.evaluation.runner import load_arpabet_inventory, run_evaluation
from echo.utils.audio import WAV2VEC2_SAMPLE_RATE


def _slplab_recognize(model, processor, waveform: torch.Tensor, device: str) -> list[str]:
    """slplab 모델 추론 결과를 토큰 시퀀스로 돌려준다.

    serve.py 의 `_slplab_recognize` 와 같은 collapsing (frame argmax + duplicate/pad collapse +
    `convert_ids_to_tokens`) 을 써서 두 경로의 출력 분포가 같다 — 디코더 차이로 PER 비교가
    어긋나는 일을 피한다.
    """
    inputs = processor(
        waveform.cpu().numpy(),
        sampling_rate=WAV2VEC2_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()

    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

    tokens: list[str] = []
    prev = -1
    for tid in predicted_ids:
        if tid == pad_id:
            prev = tid
            continue
        if tid == prev:
            continue
        token_text = tokenizer.convert_ids_to_tokens(tid)
        if token_text:
            tokens.append(token_text)
        prev = tid
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", default="data/test.json")
    parser.add_argument("--data-root", default=".",
                        help="test.json 의 wav 상대경로의 기준 디렉터리.")
    parser.add_argument("--hf-id",
                        default="slplab/wav2vec2-large-robust-L2-english-phoneme-recognition")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="analysis_output/slplab_evaluation.json")
    parser.add_argument("--phoneme-map", default=DEFAULT_PHONEME_MAP_PATH,
                        help="표준 ARPABET 인벤토리 — sanitize 매핑의 기준이 된다.")
    parser.add_argument("--limit", type=int, default=None,
                        help="디버그용. 처음 N 개 발화만 평가.")
    args = parser.parse_args()

    print(f"모델 로드: {args.hf_id} → {args.device}")
    processor = Wav2Vec2Processor.from_pretrained(args.hf_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.hf_id).to(args.device).eval()

    print(f"인벤토리 로드: {args.phoneme_map}")
    inventory = load_arpabet_inventory(args.phoneme_map)
    print(f"표준 음소 {len(inventory)} 개 — sanitize 후 PER 측정에 사용")

    def recognize(waveform):
        return _slplab_recognize(model, processor, waveform, args.device)

    run_evaluation(
        recognize_fn=recognize,
        manifest_path=pathlib.Path(args.test_json),
        inventory=inventory,
        sampling_rate=WAV2VEC2_SAMPLE_RATE,
        output_path=pathlib.Path(args.output),
        model_metadata={"model": args.hf_id},
        data_root=pathlib.Path(args.data_root).resolve(),
        limit=args.limit,
        progress_label="slplab eval",
    )


if __name__ == "__main__":
    main()
