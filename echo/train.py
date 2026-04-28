"""Training script for baseline pronunciation assessment model."""

import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import Config
from .data.dataset import PronunciationDataset, collate_batch
from .evaluation.evaluator import ModelEvaluator
from .models.model import BaselineModel
from .models.film_model import FiLMModel
from .training.trainer import Trainer


def build_model(config: Config):
    if config.model_type == 'baseline':
        return BaselineModel.from_config(config)
    if config.model_type == 'film':
        return FiLMModel.from_config(config)
    raise ValueError(f"Unknown model_type: {config.model_type}")
from .utils.distributed import (
    is_distributed, is_main_process, get_rank,
    get_local_rank, get_world_size, barrier, get_device, unwrap_model,
)

logger = logging.getLogger(__name__)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def save_checkpoint(model, wav2vec_opt, main_opt, epoch, val_loss, train_loss, metrics, path, config=None):
    raw = unwrap_model(model)
    ckpt = {
        'epoch': epoch,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'metrics': metrics,
        'model_state_dict': raw.state_dict(),
    }
    if config:
        ckpt['config'] = config.to_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def train_model(config: Config):
    device = get_device(config.device_id)
    set_random_seed(config.seed, config.cudnn_deterministic)

    if is_main_process():
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.result_dir, exist_ok=True)
        config.save_config(os.path.join(config.experiment_dir, 'config.json'))

        # File logging
        fh = logging.FileHandler(os.path.join(config.log_dir, 'training.log'), mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)

    if is_distributed():
        barrier()

    phoneme_to_id, id_to_phoneme = config.load_phoneme_mappings()

    if is_main_process():
        logger.info(f"Model: {config.pretrained_model}")
        logger.info(f"Device: {device}")
        logger.info(f"Wav2Vec LR: {config.wav2vec_lr}, Main LR: {config.main_lr}")

    # Model
    model = build_model(config)
    if config.gradient_checkpointing:
        if hasattr(model.encoder.wav2vec2, 'gradient_checkpointing_enable'):
            model.encoder.wav2vec2.gradient_checkpointing_enable()
    model = model.to(device)

    if is_distributed():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank(),
                     static_graph=config.static_graph,
                     find_unused_parameters=config.find_unused_parameters)
        set_random_seed(config.seed + get_rank(), config.cudnn_deterministic)

    # Data
    ds_kwargs = dict(phoneme_to_id=phoneme_to_id, max_length=config.max_length,
                     sampling_rate=config.sampling_rate, is_main_process=is_main_process())
    train_ds = PronunciationDataset(config.train_data, **ds_kwargs)
    val_ds = PronunciationDataset(config.val_data, **ds_kwargs)
    test_ds = PronunciationDataset(config.test_data, **ds_kwargs)

    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=config.seed) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None

    def worker_init(wid):
        np.random.seed(config.seed + wid)
        random.seed(config.seed + wid)

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, num_workers=config.num_workers,
                              shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=pin,
                              collate_fn=collate_batch, drop_last=True,
                              persistent_workers=config.num_workers > 0, worker_init_fn=worker_init)
    val_loader = DataLoader(val_ds, batch_size=config.eval_batch_size, num_workers=config.num_workers,
                            shuffle=False, sampler=val_sampler, pin_memory=pin,
                            collate_fn=collate_batch, persistent_workers=config.num_workers > 0,
                            worker_init_fn=worker_init)
    test_loader = DataLoader(test_ds, batch_size=config.eval_batch_size, num_workers=config.num_workers,
                             shuffle=False, pin_memory=pin, collate_fn=collate_batch,
                             persistent_workers=config.num_workers > 0, worker_init_fn=worker_init)

    # Trainer & evaluator
    trainer = Trainer(model, config, device, logger)
    evaluator = ModelEvaluator(device)
    wav2vec_opt, main_opt = trainer.get_optimizers()

    best_metrics = {'perceived_per': float('inf'), 'mdd_f1': 0.0, 'val_loss': float('inf')}

    for epoch in range(1, config.num_epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate_epoch(val_loader)

        if is_main_process():
            logger.info(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            raw = unwrap_model(model)
            metrics = evaluator.evaluate(raw, test_loader, id_to_phoneme)
            per = metrics.get('per', float('inf'))
            mdd_f1 = metrics.get('mdd_f1', 0.0)
            logger.info(f"Eval — PER: {per:.4f} | MDD F1: {mdd_f1:.4f}")

            # Save best
            improved = False
            if val_loss < best_metrics['val_loss']:
                best_metrics['val_loss'] = val_loss
            if per < best_metrics['perceived_per']:
                best_metrics['perceived_per'] = per
            if mdd_f1 > best_metrics['mdd_f1']:
                best_metrics['mdd_f1'] = mdd_f1
                improved = True

            if improved:
                save_checkpoint(model, wav2vec_opt, main_opt, epoch, val_loss, train_loss,
                                best_metrics, os.path.join(config.checkpoint_dir, 'best_mdd_f1.pth'), config)
                logger.info("Saved best MDD F1 checkpoint")

        if is_distributed():
            barrier()

    if is_main_process():
        # Save final metrics
        final = {**best_metrics, **config.to_dict()}
        os.makedirs(config.result_dir, exist_ok=True)
        with open(os.path.join(config.result_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final, f, indent=2)
        logger.info(f"Training completed! Best: {best_metrics}")
