"""Domain constants for pronunciation assessment."""

import json
from typing import Dict, FrozenSet, Tuple

MODEL_ALIASES: Dict[str, str] = {
    'wav2vec2-base': 'facebook/wav2vec2-base',
    'wav2vec2-large': 'facebook/wav2vec2-large',
    'hubert-base': 'facebook/hubert-base-ls960',
    'hubert-large': 'facebook/hubert-large-ls960-ft',
}

SILENCE_TOKENS: FrozenSet[str] = frozenset({'sil', 'sp', 'spn', 'pau', ''})

# CTC blank id. dataset / inference / trainer 가 동일한 값을 보도록 단일 출처화.
BLANK_TOKEN_ID: int = 0

# 음소 인벤토리 단일 출처. 코드에서 phoneme_to_id.json 경로를 직접 적지 말고 이 상수를 쓴다.
DEFAULT_PHONEME_MAP_PATH: str = "data/phoneme_to_id.json"


def load_phoneme_inventory(path: str = DEFAULT_PHONEME_MAP_PATH) -> Tuple[Dict[str, int], Dict[int, str]]:
    """`phoneme_to_id.json` 을 로드해 (token->id, id->token) 양방향 사전을 반환한다."""
    with open(path, "r", encoding="utf-8") as f:
        p2id: Dict[str, int] = json.load(f)
    return p2id, {v: k for k, v in p2id.items()}


def infer_num_phonemes(phoneme_to_id: Dict[str, int]) -> int:
    """vocab 크기 = 토큰 개수. 모델 출력 차원과 직결되므로 두 곳에서 따로 정의하지 말고 이 함수로 계산."""
    return len(phoneme_to_id)
