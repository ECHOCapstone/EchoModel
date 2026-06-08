"""FiLM 체크포인트를 '올바른' 평가기(canonical 주입)로 test 평가.

eval_echo.py(PronunciationScorer.transcribe)는 canonical_labels 를 모델에 넘기지
않으므로 FiLM 변조가 비활성화돼 FiLM 체크포인트 성능이 붕괴한다. 이 스크립트는
학습/val 과 동일한 echo.evaluation.evaluator.ModelEvaluator 를 써서 canonical 을
주입한 상태로 PER / MDD(FRR/FAR) 를 측정한다.
"""
import argparse, json
import torch
from torch.utils.data import DataLoader

from echo.data.dataset import PronunciationDataset, collate_batch
from echo.evaluation.evaluator import ModelEvaluator
from echo.models.film_model import FiLMModel
from echo.models.model import BaselineModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--phoneme-map", default="data/phoneme_to_id.json")
    ap.add_argument("--test-json", default="data/test.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.phoneme_map, encoding="utf-8") as f:
        phoneme_to_id = json.load(f)
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    num_phonemes = cfg.get("num_phonemes", len(phoneme_to_id))
    hidden_dim = cfg.get("hidden_dim", 768)
    pretrained = cfg.get("pretrained_model", "facebook/wav2vec2-base")
    mtype = cfg.get("model_type", "baseline")

    if mtype == "film":
        model = FiLMModel(pretrained_model_name=pretrained, hidden_dim=hidden_dim,
                          num_phonemes=num_phonemes, dropout=cfg.get("dropout", 0.1),
                          film_embed_dim=cfg.get("film_embed_dim", 128))
    else:
        model = BaselineModel(pretrained_model_name=pretrained, hidden_dim=hidden_dim,
                              num_phonemes=num_phonemes, dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device)

    ds = PronunciationDataset(args.test_json, phoneme_to_id=phoneme_to_id,
                              max_length=cfg.get("max_length", 160000),
                              sampling_rate=cfg.get("sampling_rate", 16000))
    loader = DataLoader(ds, batch_size=16, num_workers=8, shuffle=False,
                        collate_fn=collate_batch)

    evaluator = ModelEvaluator(args.device)
    res = evaluator.evaluate(model, loader, id_to_phoneme)

    out = {
        "checkpoint": args.checkpoint,
        "model_type": mtype,
        "pretrained_model": pretrained,
        "test_json": args.test_json,
        "per": res["per"],
        "total_phonemes": res["total_phonemes"],
        "total_errors": res["total_errors"],
        "mdd_f1": res.get("mdd_f1", 0.0),
        "mdd": res.get("mdd_metrics", {}),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    m = out["mdd"]
    print(f"\n[PROPER EVAL] PER={out['per']:.4f} F1={out['mdd_f1']:.4f} "
          f"P={m.get('precision',0):.4f} R={m.get('recall',0):.4f} "
          f"FRR={m.get('frr',0):.4f} FAR={m.get('far',0):.4f}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
