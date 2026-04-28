"""PER and MDD evaluation metrics."""

from typing import List, Dict, Tuple, Optional
from ..constants import SILENCE_TOKENS


def get_alignment_with_indices(reference, hypothesis):
    ref_len, hyp_len = len(reference), len(hypothesis)
    if ref_len == 0:
        return [('I', None, j) for j in range(hyp_len)]
    if hyp_len == 0:
        return [('D', i, None) for i in range(ref_len)]

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    alignment = []
    i, j = ref_len, hyp_len
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i-1] == hypothesis[j-1]:
            alignment.append(('=', i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append(('S', i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            alignment.append(('D', i-1, None)); i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append(('I', None, j-1)); j -= 1
        else:
            break
    alignment.reverse()
    return alignment


def calculate_sequence_error_rate(ids, references, hypotheses):
    details = []
    for sid, ref, hyp in zip(ids, references, hypotheses):
        ref_t = [str(t) for t in ref]
        hyp_t = [str(t) for t in hyp]
        if not ref_t:
            details.append({'id': sid, 'error_rate': 0, 'num_reference_tokens': 0,
                            'substitutions': 0, 'deletions': 0, 'insertions': len(hyp_t)})
            continue
        align = get_alignment_with_indices(ref_t, hyp_t)
        s = sum(1 for op, _, _ in align if op == 'S')
        d = sum(1 for op, _, _ in align if op == 'D')
        ins = sum(1 for op, _, _ in align if op == 'I')
        total = s + d + ins
        details.append({
            'id': sid, 'error_rate': total / len(ref_t) if ref_t else 0,
            'num_reference_tokens': len(ref_t),
            'substitutions': s, 'deletions': d, 'insertions': ins,
        })
    return details


def _mdd_stats_aligned(canonical_aligned, perceived_aligned, model_output):
    align = get_alignment_with_indices(perceived_aligned, model_output)
    ta, fr, fa, tr = 0, 0, 0, 0

    for op, ref_idx, hyp_idx in align:
        if ref_idx is None:
            fr += 1
            continue

        can = canonical_aligned[ref_idx]
        per = perceived_aligned[ref_idx]
        is_can_sil = can.lower() in SILENCE_TOKENS
        is_per_sil = per.lower() in SILENCE_TOKENS

        if is_can_sil and is_per_sil:
            continue

        gt_is_error = (can != per)

        if hyp_idx is not None and hyp_idx < len(model_output):
            pred_is_error = (model_output[hyp_idx] != can) if not is_can_sil else True
        else:
            pred_is_error = not is_can_sil

        if not gt_is_error and not pred_is_error:
            ta += 1
        elif gt_is_error and pred_is_error:
            tr += 1
        elif gt_is_error and not pred_is_error:
            fa += 1
        else:
            fr += 1

    return ta, tr, fa, fr


def calculate_mdd_metrics(
    all_predictions: List[List[str]],
    all_canonical_aligned: List[List[str]],
    all_perceived_aligned: List[List[str]],
) -> Dict:
    total_ta, total_tr, total_fa, total_fr = 0, 0, 0, 0

    for pred, can_al, per_al in zip(all_predictions, all_canonical_aligned, all_perceived_aligned):
        if not can_al or not per_al or len(can_al) != len(per_al):
            continue
        ta, tr, fa, fr = _mdd_stats_aligned(can_al, per_al, pred)
        total_ta += ta; total_tr += tr; total_fa += fa; total_fr += fr

    precision = total_tr / (total_tr + total_fr) if (total_tr + total_fr) > 0 else 0.0
    recall = total_tr / (total_tr + total_fa) if (total_tr + total_fa) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_correct = total_ta + total_fr
    total_errors = total_tr + total_fa
    total = total_ta + total_tr + total_fa + total_fr
    frr = total_fr / total_correct if total_correct > 0 else 0.0
    far = total_fa / total_errors if total_errors > 0 else 0.0
    da = (total_ta + total_tr) / total if total > 0 else 0.0

    return {
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'frr': frr, 'far': far, 'da': da,
        'true_acceptance': total_ta, 'false_rejection': total_fr,
        'false_acceptance': total_fa, 'true_rejection': total_tr,
    }
