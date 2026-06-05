"""평가 스크립트가 공유하는 체크포인트 로더.

eval_slplab / measure_noise_robustness / per_source_frr 가 동일한 로딩 로직을 복붙하던 것을
한곳으로 모은다. 체크포인트의 config 에서 모델 종류(baseline/film)를 읽어 해당 모델을 복원한다.
"""

import torch

from echo.config import Config
from echo.models.model import BaselineModel
from echo.models.film_model import FiLMModel
from echo.utils.audio import enable_specaugment


def load_eval_model(ckpt_path, device):
    """체크포인트에서 추론용 모델과 그 Config 를 복원한다. SpecAugment 는 평가용으로 끈다."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model_type = cfg.get('model_type', 'baseline')
    config = Config(
        pretrained_model=cfg.get('pretrained_model', 'facebook/wav2vec2-base'),
        model_type=model_type,
        film_embed_dim=cfg.get('film_embed_dim', 128),
    )
    cls = FiLMModel if model_type == 'film' else BaselineModel
    model = cls.from_config(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    enable_specaugment(model, False)
    return model, config
