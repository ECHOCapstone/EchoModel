"""CLI entry point for ECHO baseline training."""

import argparse
import logging
import sys

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
    parser.add_argument('--data_dir', type=str, default='data')
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
    setup_distributed()
    config = Config.from_args(args)

    try:
        train_model(config)
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
