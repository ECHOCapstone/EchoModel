"""Noise robustness measurement (eval-only, no retraining).

Mixes deterministic MUSAN noise into the test set at fixed SNRs and measures
PER / MDD F1 degradation. Same noise per file across all evaluations →
fair comparison.

Robustness metrics:
  - PER@SNR for SNR ∈ {clean, 20, 15, 10, 5} dB
  - ΔPER = PER@SNR - PER@clean   (smaller = more robust)
  - PER ratio = PER@SNR / PER@clean

Usage:
  python measure_noise_robustness.py \
      --checkpoint experiments/.../best_mdd_f1.pth \
      --snrs clean,20,15,10,5

Note: This script does not modify any source files. Noise is mixed in-memory
at evaluation time only.
"""

import argparse
import json
import os
import random
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from echo.config import Config
from echo.data.dataset import PronunciationDataset, collate_batch
from echo.models.model import BaselineModel
from echo.models.film_model import FiLMModel
from echo.evaluation.evaluator import frame_collapse_decode
from echo.evaluation.metrics import calculate_sequence_error_rate, calculate_mdd_metrics
from echo.utils.audio import create_attention_mask, compute_output_lengths, enable_specaugment
from echo.constants import SILENCE_TOKENS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test_data', default='data/test.json')
    p.add_argument('--phoneme_map', default='data/phoneme_to_id.json')
    p.add_argument('--snrs', default='clean,20,15,10,5',
                   help='Comma-separated SNRs in dB (or "clean")')
    p.add_argument('--musan_split_json', default='data/musan/split.json')
    p.add_argument('--noise_split', default='test',
                   help='MUSAN split for noise (test = unseen noise files)')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--device_id', type=int, default=0)
    p.add_argument('--output', default=None)
    return p.parse_args()


def load_model(ckpt_path, device):
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


def mix_noise_deterministic(
    waveform: torch.Tensor,
    file_key: str,
    noise_files: list,
    snr_db: float,
    sampling_rate: int,
) -> torch.Tensor:
    """Mixes one MUSAN noise clip into waveform at fixed SNR.
    Same file_key always gets same noise (reproducible across runs)."""
    seed = abs(hash(file_key)) % (2**32)
    rng = random.Random(seed)
    path = rng.choice(noise_files)

    data, sr = sf.read(path, dtype='float32')
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != sampling_rate:
        ratio = sampling_rate / sr
        new_len = int(len(data) * ratio)
        data = np.interp(
            np.linspace(0, len(data) - 1, new_len),
            np.arange(len(data)),
            data,
        ).astype(np.float32)

    n = waveform.shape[0]
    if len(data) >= n:
        start = rng.randint(0, len(data) - n)
        data = data[start:start + n]
    else:
        reps = int(np.ceil(n / len(data)))
        data = np.tile(data, reps)[:n]

    noise = torch.from_numpy(data)
    speech_p = waveform.pow(2).mean().clamp(min=1e-10)
    noise_p = noise.pow(2).mean().clamp(min=1e-10)
    snr_lin = 10 ** (snr_db / 10)
    scale = torch.sqrt(speech_p / (noise_p * snr_lin))
    return waveform + scale * noise


def evaluate(model, loader, id2p, device, snr, noise_files, sampling_rate):
    """snr=None for clean."""
    all_pred, all_gt = [], []
    all_can, all_per = [], []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'SNR={snr}', leave=False):
            if batch is None:
                continue
            wavs = batch['waveforms']

            if snr is not None:
                aug = []
                for i in range(wavs.size(0)):
                    fp = batch['file_paths'][i]
                    L = batch['audio_lengths'][i].item()
                    clean = wavs[i, :L]
                    noisy = mix_noise_deterministic(
                        clean, fp, noise_files, float(snr), sampling_rate)
                    pad = torch.zeros_like(wavs[i])
                    pad[:L] = noisy
                    aug.append(pad)
                wavs = torch.stack(aug)

            wavs = wavs.to(device)
            audio_lens = batch['audio_lengths'].to(device)
            mask = create_attention_mask(wavs, audio_lens)
            in_lens = compute_output_lengths(model, audio_lens)

            fwd = dict(waveform=wavs, attention_mask=mask)
            if 'canonical_labels' in batch:
                fwd['canonical_labels'] = batch['canonical_labels'].to(device)
                fwd['canonical_lengths'] = batch['canonical_lengths'].to(device)

            out = model(**fwd)
            logits = out['perceived_logits']
            clamped = torch.clamp(in_lens, min=1, max=logits.size(1))
            pred_ids = frame_collapse_decode(logits, clamped)

            gt = batch['perceived_labels']
            gt_lens = batch['perceived_lengths']
            for i in range(len(pred_ids)):
                all_pred.append([id2p.get(p, 'UNK') for p in pred_ids[i]])
                all_gt.append([id2p.get(g, 'UNK') for g in gt[i, :gt_lens[i].item()].tolist()])
            all_ids.extend(batch['file_paths'])
            all_can.extend(batch['canonical_aligned'])
            all_per.extend(batch['perceived_aligned'])

    pred_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_pred]
    gt_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_gt]

    details = calculate_sequence_error_rate(all_ids, gt_clean, pred_clean)
    total_ph = sum(d['num_reference_tokens'] for d in details)
    total_s = sum(d['substitutions'] for d in details)
    total_d = sum(d['deletions'] for d in details)
    total_i = sum(d['insertions'] for d in details)
    per = (total_s + total_d + total_i) / total_ph if total_ph > 0 else 0

    can_p = [s.split() if s else [] for s in all_can]
    per_p = [s.split() if s else [] for s in all_per]
    mdd = calculate_mdd_metrics(pred_clean, can_p, per_p)

    return {
        'PER': per,
        'sub_rate': total_s / total_ph if total_ph > 0 else 0,
        'del_rate': total_d / total_ph if total_ph > 0 else 0,
        'ins_rate': total_i / total_ph if total_ph > 0 else 0,
        'MDD_F1': mdd['f1_score'],
        'Precision': mdd['precision'],
        'Recall': mdd['recall'],
        'FRR': mdd['frr'],
        'FAR': mdd['far'],
    }


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.device_id}')

    print(f'Loading: {args.checkpoint}')
    model, config = load_model(args.checkpoint, device)

    with open(args.phoneme_map) as f:
        p2id = json.load(f)
    id2p = {v: k for k, v in p2id.items()}

    ds = PronunciationDataset(args.test_data, p2id, max_length=config.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=8,
                        shuffle=False, collate_fn=collate_batch, pin_memory=True)

    with open(args.musan_split_json) as f:
        noise_files = json.load(f)[args.noise_split]
    print(f'Using {len(noise_files)} unseen MUSAN files (split={args.noise_split})')

    snrs = [None if s == 'clean' else float(s) for s in args.snrs.split(',')]

    results = {}
    print(f'\n{"SNR":>8s}  {"PER":>7s}  {"S":>6s}  {"D":>6s}  {"I":>6s}  '
          f'{"MDD_F1":>7s}  {"Prec":>6s}  {"Rec":>6s}  {"FRR":>6s}  {"FAR":>6s}')
    print('-' * 80)
    for snr in snrs:
        label = 'clean' if snr is None else f'{int(snr) if snr.is_integer() else snr}dB'
        m = evaluate(model, loader, id2p, device, snr, noise_files, config.sampling_rate)
        results[label] = m
        print(f'  {label:>6s}  {m["PER"]*100:>6.2f}%  {m["sub_rate"]*100:>5.2f}%  '
              f'{m["del_rate"]*100:>5.2f}%  {m["ins_rate"]*100:>5.2f}%  '
              f'{m["MDD_F1"]*100:>6.2f}%  {m["Precision"]*100:>5.2f}%  '
              f'{m["Recall"]*100:>5.2f}%  {m["FRR"]*100:>5.2f}%  {m["FAR"]*100:>5.2f}%')

    # Robustness deltas
    if 'clean' in results:
        clean_per = results['clean']['PER']
        clean_f1 = results['clean']['MDD_F1']
        print(f'\nRobustness (vs clean):')
        for label, m in results.items():
            if label == 'clean':
                continue
            d_per = (m['PER'] - clean_per) * 100
            d_f1 = (m['MDD_F1'] - clean_f1) * 100
            ratio = m['PER'] / clean_per if clean_per > 0 else 1
            print(f'  {label}: ΔPER={d_per:+.2f}%p, ΔMDD_F1={d_f1:+.2f}%p, '
                  f'PER ratio={ratio:.2f}x')

    out = args.output or os.path.join(
        os.path.dirname(args.checkpoint), '..', 'results', 'noise_robustness.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'results': results}, f, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
