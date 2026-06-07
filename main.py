"""CLI entry point for ECHO baseline training."""

import argparse
import logging
import os

import torch

from echo.config import Config
from echo.train import train_model
from echo.utils.distributed import setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description='ECHO Baseline Training')

    # Model
    parser.add_argument('--pretrained_model', type=str, default='wav2vec2-base',
                        help='Pretrained model name or alias')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'film'],
                        help='Model architecture')
    parser.add_argument('--film_embed_dim', type=int, default=128,
                        help='FiLM phoneme embedding dim')

    # Data
    parser.add_argument('--train_data', type=str, default='data/train.json')
    parser.add_argument('--val_data', type=str, default='data/val.json')
    parser.add_argument('--test_data', type=str, default='data/test.json')
    parser.add_argument('--phoneme_map', type=str, default='data/phoneme_to_id.json')

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--wav2vec_lr', type=float, default=3e-5)
    parser.add_argument('--main_lr', type=float, default=3e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=160000)
    parser.add_argument('--gradient_checkpointing', action='store_true')

    # Device
    parser.add_argument('--device_id', type=int, default=None)

    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    args = parse_args()
    # cpu 환경에서 num_workers 가 큰 값으로 들어오면 IPC 비용이 학습 step 보다 커진다.
    # cuda 가 없을 때만 cpu_count 기준으로 줄여 OS 한도 초과를 막는다 (cuda 환경은 사용자가
    # 명시 지정한 값을 그대로 신뢰).
    if not torch.cuda.is_available() and args.num_workers > 0:
        safe_workers = min(args.num_workers, max(1, (os.cpu_count() or 2) // 2))
        if safe_workers != args.num_workers:
            logging.getLogger(__name__).warning(
                "CUDA unavailable; capping --num_workers from %d to %d.", args.num_workers, safe_workers,
            )
            args.num_workers = safe_workers
    setup_distributed()
    config = Config.from_args(args)

    try:
        train_model(config)
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
