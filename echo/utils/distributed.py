"""Distributed training utilities."""

import os
import socket

import torch
import torch.distributed as dist
from typing import Optional


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = 'nccl', master_port: Optional[int] = None) -> bool:
    if dist.is_initialized():
        return True
    if 'RANK' not in os.environ:
        return False

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        if master_port is not None:
            os.environ['MASTER_PORT'] = str(master_port)
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                os.environ['MASTER_PORT'] = str(s.getsockname()[1])

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']),
        device_id=torch.device(f'cuda:{local_rank}'),
    )
    return True


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def barrier():
    if is_distributed():
        dist.barrier()


def get_device(device_id: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        if is_distributed():
            return torch.device(f'cuda:{get_local_rank()}')
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        return torch.device('cuda:0')
    return torch.device('cpu')


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model
