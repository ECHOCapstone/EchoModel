"""Model evaluator: PER + MDD metrics."""

from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import calculate_sequence_error_rate, calculate_mdd_metrics
from ..constants import SILENCE_TOKENS
from ..utils.audio import compute_output_lengths, enable_specaugment, create_attention_mask
from ..utils.distributed import is_main_process


def frame_collapse_decode(logits, input_lengths, blank=0):
    batch_size = logits.size(0)
    decoded = []
    for b in range(batch_size):
        T = input_lengths[b].item()
        preds = logits[b, :T].argmax(dim=-1)
        seq = []
        prev = -1
        for t in range(T):
            p = preds[t].item()
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        decoded.append(seq)
    return decoded


class ModelEvaluator:
    def __init__(self, device='cuda'):
        self.device = device

    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        id_to_phoneme: Dict[int, str],
    ) -> Dict:
        model.eval()
        enable_specaugment(model, False)

        all_pred, all_gt = [], []
        all_can_aligned, all_per_aligned = [], []
        all_ids, all_spk = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Eval', disable=not is_main_process()):
                if batch is None or 'perceived_labels' not in batch:
                    continue

                waveforms = batch['waveforms'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                mask = create_attention_mask(waveforms, audio_lengths)
                input_lengths = compute_output_lengths(model, audio_lengths)

                canonical_labels = batch.get('canonical_labels')
                canonical_lengths = batch.get('canonical_lengths')
                if canonical_labels is not None:
                    canonical_labels = canonical_labels.to(self.device)
                    canonical_lengths = canonical_lengths.to(self.device)

                outputs = model(
                    waveforms,
                    attention_mask=mask,
                    canonical_labels=canonical_labels,
                    canonical_lengths=canonical_lengths,
                )
                logits = outputs['perceived_logits']
                clamped = torch.clamp(input_lengths, min=1, max=logits.size(1))

                pred_ids = frame_collapse_decode(logits, clamped)
                gt_labels = batch['perceived_labels']
                gt_lengths = batch['perceived_lengths']

                for i in range(len(pred_ids)):
                    length = gt_lengths[i].item()
                    gt_seq = gt_labels[i, :length].tolist()
                    all_pred.append([id_to_phoneme.get(p, f'UNK_{p}') for p in pred_ids[i]])
                    all_gt.append([id_to_phoneme.get(g, f'UNK_{g}') for g in gt_seq])

                all_ids.extend(batch['file_paths'])
                all_spk.extend(batch['speaker_ids'])
                if 'canonical_aligned' in batch:
                    all_can_aligned.extend(batch['canonical_aligned'])
                    all_per_aligned.extend(batch['perceived_aligned'])

        # PER
        pred_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_pred]
        gt_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in all_gt]

        per_details = calculate_sequence_error_rate(all_ids, gt_clean, pred_clean)
        total_ph = sum(d['num_reference_tokens'] for d in per_details)
        total_err = sum(d['substitutions'] + d['deletions'] + d['insertions'] for d in per_details)
        per = total_err / total_ph if total_ph > 0 else 0.0

        results = {
            'per': per,
            'total_phonemes': total_ph,
            'total_errors': total_err,
        }

        # MDD
        can_parsed = [s.split() if s else [] for s in all_can_aligned]
        per_parsed = [s.split() if s else [] for s in all_per_aligned]

        has_aligned = (
            len(can_parsed) == len(pred_clean)
            and all(len(c) > 0 for c in can_parsed)
            and all(len(p) > 0 for p in per_parsed)
        )
        if has_aligned:
            mdd = calculate_mdd_metrics(pred_clean, can_parsed, per_parsed)
            results['mdd_f1'] = mdd['f1_score']
            results['mdd_metrics'] = mdd
        else:
            results['mdd_f1'] = 0.0

        return results
