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
import pathlib

from echo.constants import DEFAULT_PHONEME_MAP_PATH
from echo.evaluation.runner import load_arpabet_inventory, run_evaluation
from echo.inference import PronunciationScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="ECHO 학습 체크포인트 (.pth) 경로.")
    parser.add_argument("--phoneme-map", default=DEFAULT_PHONEME_MAP_PATH)
    parser.add_argument("--test-json", default="data/test.json")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="analysis_output/echo_evaluation.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="디버그용. 처음 N 개 발화만 평가.")
    args = parser.parse_args()

    print(f"체크포인트 로드: {args.checkpoint} → {args.device}")
    scorer = PronunciationScorer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        phoneme_map_path=args.phoneme_map,
        device=args.device,
    )

    inventory = load_arpabet_inventory(args.phoneme_map)
    print(f"표준 음소 {len(inventory)} 개 — sanitize 후 PER 측정에 사용")

    def recognize(waveform):
        return scorer.transcribe(waveform, keep_silence=False)

    run_evaluation(
        recognize_fn=recognize,
        manifest_path=pathlib.Path(args.test_json),
        inventory=inventory,
        sampling_rate=scorer.sampling_rate,
        output_path=pathlib.Path(args.output),
        model_metadata={"model": "echo", "checkpoint": str(args.checkpoint)},
        data_root=pathlib.Path(args.data_root).resolve(),
        limit=args.limit,
        progress_label="echo eval",
    )


if __name__ == "__main__":
    main()
