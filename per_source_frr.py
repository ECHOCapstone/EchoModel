"""Per-source FRR/MDD metrics computation for all checkpoints."""

import argparse
import json
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from echo.data.dataset import PronunciationDataset, collate_batch
from echo.evaluation.evaluator import frame_collapse_decode
from echo.evaluation.loading import load_eval_model
from echo.evaluation.metrics import calculate_mdd_metrics, calculate_sequence_error_rate
from echo.utils.audio import create_attention_mask, compute_output_lengths
from echo.constants import SILENCE_TOKENS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test_data', default='data/test.json')
    p.add_argument('--phoneme_map', default='data/phoneme_to_id.json')
    p.add_argument('--device_id', type=int, default=0)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.device_id}')

    print(f'Loading: {args.checkpoint}')
    model, config = load_eval_model(args.checkpoint, device)

    with open(args.phoneme_map) as f:
        p2id = json.load(f)
    id2p = {v: k for k, v in p2id.items()}

    # Load aihub keys for source detection
    aihub_keys = set()
    if os.path.exists('data/aihub_clean_v2.json'):
        with open('data/aihub_clean_v2.json') as f:
            for k in json.load(f).keys():
                aihub_keys.add(k)

    ds = PronunciationDataset(args.test_data, p2id, max_length=config.max_length)
    loader = DataLoader(ds, batch_size=16, num_workers=8,
                        shuffle=False, collate_fn=collate_batch, pin_memory=True)

    # Collect per-source data
    per_source = defaultdict(lambda: {'pred': [], 'gt': [], 'can': [], 'per_': [], 'ids': []})

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            waves = batch['waveforms'].to(device)
            lens = batch['audio_lengths'].to(device)
            mask = create_attention_mask(waves, lens)
            in_lens = compute_output_lengths(model, lens)

            fwd = dict(waveform=waves, attention_mask=mask)
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
                fp = batch['file_paths'][i]
                key = os.path.basename(fp).replace('.wav', '')
                source = 'aihub' if key in aihub_keys else 'l2arctic'

                pred_ph = [id2p.get(p, 'UNK') for p in pred_ids[i]]
                gt_ph = [id2p.get(g, 'UNK') for g in gt[i, :gt_lens[i].item()].tolist()]
                per_source[source]['pred'].append(pred_ph)
                per_source[source]['gt'].append(gt_ph)
                per_source[source]['can'].append(batch['canonical_aligned'][i])
                per_source[source]['per_'].append(batch['perceived_aligned'][i])
                per_source[source]['ids'].append(fp)

    # Compute per-source metrics
    results = {}
    for src, d in per_source.items():
        pred_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in d['pred']]
        gt_clean = [[p for p in s if p.lower() not in SILENCE_TOKENS] for s in d['gt']]

        det = calculate_sequence_error_rate(d['ids'], gt_clean, pred_clean)
        total_ph = sum(x['num_reference_tokens'] for x in det)
        total_s = sum(x['substitutions'] for x in det)
        total_d = sum(x['deletions'] for x in det)
        total_i = sum(x['insertions'] for x in det)
        per = (total_s + total_d + total_i) / total_ph if total_ph > 0 else 0

        can_p = [s.split() if s else [] for s in d['can']]
        per_p = [s.split() if s else [] for s in d['per_']]
        mdd = calculate_mdd_metrics(pred_clean, can_p, per_p)

        results[src] = {
            'n': len(d['pred']),
            'total_phonemes': total_ph,
            'PER': per,
            'sub_rate': total_s/total_ph if total_ph>0 else 0,
            'del_rate': total_d/total_ph if total_ph>0 else 0,
            'ins_rate': total_i/total_ph if total_ph>0 else 0,
            'MDD_F1': mdd['f1_score'],
            'Precision': mdd['precision'],
            'Recall': mdd['recall'],
            'FRR': mdd['frr'],
            'FAR': mdd['far'],
            'TA': mdd['true_acceptance'],
            'TR': mdd['true_rejection'],
            'FA': mdd['false_acceptance'],
            'FR': mdd['false_rejection'],
        }
        print(f'{src}: n={results[src]["n"]} PER={per*100:.2f}% '
              f'MDD_F1={mdd["f1_score"]*100:.2f}% FRR={mdd["frr"]*100:.2f}% '
              f'FAR={mdd["far"]*100:.2f}%')

    with open(args.output, 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'per_source': results}, f, indent=2)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
