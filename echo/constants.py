"""Domain constants for pronunciation assessment."""

from typing import Dict, FrozenSet

MODEL_ALIASES: Dict[str, str] = {
    'wav2vec2-base': 'facebook/wav2vec2-base',
    'wav2vec2-large': 'facebook/wav2vec2-large',
    'hubert-base': 'facebook/hubert-base-ls960',
    'hubert-large': 'facebook/hubert-large-ls960-ft',
}

SILENCE_TOKENS: FrozenSet[str] = frozenset({'sil', 'sp', 'spn', 'pau', ''})
