"""Configuration for baseline pronunciation assessment model."""

import json
import logging
import os
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .constants import (
    DEFAULT_PHONEME_MAP_PATH,
    MODEL_ALIASES,
    infer_num_phonemes,
    load_phoneme_inventory,
)
from .utils.audio import WAV2VEC2_SAMPLE_RATE


@dataclass
class Config:
    # Pretrained model
    pretrained_model: str = "facebook/wav2vec2-base"
    sampling_rate: int = WAV2VEC2_SAMPLE_RATE
    max_length: int = 160000  # 10s at 16kHz

    # Model
    model_type: str = 'baseline'  # 'baseline' | 'film'
    # phoneme_map 의 토큰 개수로 자동 계산된다. 명시 지정 시 그 값을 우선.
    num_phonemes: Optional[int] = None
    hidden_dim: Optional[int] = None
    dropout: float = 0.1

    # FiLM
    film_embed_dim: int = 128

    # SpecAugment
    wav2vec2_specaug: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10

    # Training
    batch_size: int = 8
    eval_batch_size: int = 16
    num_workers: int = 8
    num_epochs: int = 20
    gradient_accumulation: int = 2
    wav2vec_lr: float = 3e-5
    main_lr: float = 3e-4
    max_grad_norm: float = 1.0
    seed: int = 42
    cudnn_deterministic: bool = False
    gradient_checkpointing: bool = False

    # Distributed
    use_ddp: bool = False
    ddp_backend: str = 'nccl'
    static_graph: bool = False
    find_unused_parameters: bool = False
    master_port: int = 29500
    device_id: Optional[int] = None

    # Data paths
    train_data: str = "data/train.json"
    val_data: str = "data/val.json"
    test_data: str = "data/test.json"
    phoneme_map: str = DEFAULT_PHONEME_MAP_PATH

    # Checkpoint
    save_best_metrics: List[str] = field(default_factory=lambda: ['mdd_f1'])

    # Experiment dirs (auto-generated)
    experiment_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    result_dir: str = ""

    def __post_init__(self):
        if self.pretrained_model in MODEL_ALIASES:
            self.pretrained_model = MODEL_ALIASES[self.pretrained_model]
        if self.hidden_dim is None:
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(self.pretrained_model)
                self.hidden_dim = cfg.hidden_size
            except Exception:
                self.hidden_dim = 768
        if self.num_phonemes is None:
            self.num_phonemes = infer_num_phonemes(load_phoneme_inventory(self.phoneme_map)[0])
        if not self.experiment_dir:
            self._setup_experiment_paths()

    def _setup_experiment_paths(self):
        model_name = self.pretrained_model.split('/')[-1]
        timestamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join("experiments", model_name, self.model_type, timestamp)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')

    @classmethod
    def from_args(cls, args) -> 'Config':
        overrides = {}
        for fi in dataclass_fields(cls):
            val = getattr(args, fi.name, None)
            if val is not None:
                overrides[fi.name] = val
        return cls(**overrides)

    def to_dict(self) -> Dict:
        return {fi.name: getattr(self, fi.name) for fi in dataclass_fields(self) if not fi.name.startswith('_')}

    def save_config(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_phoneme_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        return load_phoneme_inventory(self.phoneme_map)
