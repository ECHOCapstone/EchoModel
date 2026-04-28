"""Baseline CTC trainer for perceived phoneme recognition."""

from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.audio import create_attention_mask, compute_output_lengths, enable_specaugment
from ..utils.distributed import is_distributed, is_main_process, unwrap_model

import torch.distributed as dist


class Trainer:
    def __init__(self, model: nn.Module, config, device, logger=None):
        self.model = model
        self.config = config
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.logger = logger
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self._setup_optimizers()
        self.scaler = torch.amp.GradScaler(self.device.type)

    def _setup_optimizers(self):
        raw = unwrap_model(self.model)
        self.wav2vec_params = []
        self.main_params = []
        for name, param in raw.named_parameters():
            if 'encoder.wav2vec2' in name:
                self.wav2vec_params.append(param)
            else:
                self.main_params.append(param)

        self.wav2vec_optimizer = optim.AdamW(self.wav2vec_params, lr=self.config.wav2vec_lr)
        self.main_optimizer = optim.AdamW(self.main_params, lr=self.config.main_lr)

    def get_optimizers(self):
        return self.wav2vec_optimizer, self.main_optimizer

    def _gradient_step(self):
        if self.wav2vec_params:
            self.scaler.unscale_(self.wav2vec_optimizer)
        if self.main_params:
            self.scaler.unscale_(self.main_optimizer)

        if self.config.max_grad_norm > 0:
            if self.wav2vec_params:
                nn.utils.clip_grad_norm_(self.wav2vec_params, self.config.max_grad_norm)
            if self.main_params:
                nn.utils.clip_grad_norm_(self.main_params, self.config.max_grad_norm)

        if self.wav2vec_params:
            self.scaler.step(self.wav2vec_optimizer)
        if self.main_params:
            self.scaler.step(self.main_optimizer)
        self.scaler.update()
        self.wav2vec_optimizer.zero_grad()
        self.main_optimizer.zero_grad()

    def _compute_loss(self, batch_data) -> Tuple[torch.Tensor, Dict]:
        waveforms = batch_data['waveforms'].to(self.device)
        audio_lengths = batch_data['audio_lengths'].to(self.device)
        attention_mask = create_attention_mask(waveforms, audio_lengths)
        input_lengths = compute_output_lengths(self.model, audio_lengths)

        perceived_labels = batch_data['perceived_labels'].to(self.device)
        perceived_lengths = batch_data['perceived_lengths'].to(self.device)

        canonical_labels = batch_data.get('canonical_labels')
        canonical_lengths = batch_data.get('canonical_lengths')
        if canonical_labels is not None:
            canonical_labels = canonical_labels.to(self.device)
            canonical_lengths = canonical_lengths.to(self.device)

        outputs = self.model(
            waveforms,
            attention_mask=attention_mask,
            canonical_labels=canonical_labels,
            canonical_lengths=canonical_lengths,
        )
        logits = outputs['perceived_logits']
        clamped = torch.clamp(input_lengths, min=1, max=logits.size(1))

        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        target_flat = torch.cat([
            perceived_labels[i, :perceived_lengths[i]]
            for i in range(perceived_labels.size(0))
        ])
        loss = self.ctc_loss(log_probs, target_flat, clamped, perceived_lengths)
        return loss, {'loss': f'{loss.item():.4f}'}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        if self.config.wav2vec2_specaug:
            enable_specaugment(self.model, True)

        total_loss = 0.0
        num_batches = 0
        accumulated = 0

        self.wav2vec_optimizer.zero_grad()
        self.main_optimizer.zero_grad()

        progress = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not is_main_process())
        for batch_data in progress:
            if batch_data is None or 'perceived_labels' not in batch_data:
                continue

            is_last = (accumulated + 1) % self.config.gradient_accumulation == 0
            sync_ctx = self.model.no_sync() if (is_distributed() and not is_last) else nullcontext()

            with sync_ctx:
                with torch.amp.autocast(self.device.type):
                    loss, info = self._compute_loss(batch_data)
                    scaled = loss / self.config.gradient_accumulation
                self.scaler.scale(scaled).backward()

            total_loss += loss.item()
            num_batches += 1
            accumulated += 1

            if accumulated % self.config.gradient_accumulation == 0:
                self._gradient_step()
                accumulated = 0
                if is_main_process():
                    progress.set_postfix(info)

        if accumulated > 0:
            self._gradient_step()

        if is_distributed():
            stats = torch.tensor([total_loss, float(num_batches)], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches = stats[0].item(), int(stats[1].item())

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self, dataloader: DataLoader) -> float:
        self.model.eval()
        enable_specaugment(self.model, False)
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc='Val', disable=not is_main_process()):
                if batch_data is None or 'perceived_labels' not in batch_data:
                    continue
                with torch.amp.autocast(self.device.type):
                    loss, _ = self._compute_loss(batch_data)
                total_loss += loss.item()
                num_batches += 1

        if is_distributed():
            stats = torch.tensor([total_loss, float(num_batches)], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches = stats[0].item(), int(stats[1].item())

        return total_loss / num_batches if num_batches > 0 else 0.0
