"""모델 출력 음소 시퀀스를 학습 채점에 적합한 표준 ARPABET 시퀀스로 정규화.

음성인식 모델(특히 slplab) 은 학습 도메인의 영향으로 한국식 자음 표기 (`b*`)나
신경망 vocab 특수 토큰 (`<unk>`) 을 출력에 섞을 수 있다. 이 시퀀스가 그대로 정렬에
들어가면 strict equals 비교 단계에서 정답 음소와 무조건 불일치 처리되어 학습자에게
실제와 다른 낮은 점수가 노출된다.

본 모듈은 모델 raw 출력의 다음 잡음을 흡수한다.
- `_err` / `_xxx` suffix 표기 제거.
- vocab 특수 토큰 (`<pad>`, `<unk>`, `<s>`, `</s>`) 드롭.
- 한국식 fortis 표기 (`b*`, `d*`, `g*`) → 기본 자음으로 매핑.
- slplab vocab 의 비표준 모음 (`ax`, `eu`, `o`) → 가까운 ARPABET 모음으로 매핑.

인벤토리에 정확히 있는 표준 음소는 그대로 통과시키며, 매핑 사전에도 없는 토큰은
백엔드의 `PhonemeNormalizer` 가 `ts → t s` 같은 분해를 책임지므로 원형을 유지한다.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set


# slplab vocab 에 섞여 나오는 비표준 토큰의 ARPABET 매핑. 인벤토리에 정확히 있는 표준
# 음소는 손대지 않고, 인벤토리 외 토큰만 가까운 표준 음소로 흡수한다.
_NON_STANDARD_MAP = {
    "b*": "b",
    "d*": "d",
    "g*": "g",
    "ax": "ah",
    "eu": "uw",
    "o": "ow",
}

# 합성된 시퀀스에 섞이면 정렬에 잡음만 일으키므로 항상 드롭하는 vocab 특수 토큰.
_DROP_TOKENS = frozenset({"<s>", "</s>", "<pad>", "<unk>", "|"})

# 일부 모델은 오인식 결과를 `AA1_err` 형태로 표기한다. 같은 음소의 confidence 표기 차이일
# 뿐 음소 자체는 동일하므로 suffix 를 제거해 표준 음소로 흡수한다.
_SUFFIX_RE = re.compile(r"_[a-zA-Z0-9]+$")


def sanitize_perceived(raw: Iterable[str], inventory: Set[str]) -> List[str]:
    """모델 raw 출력 시퀀스를 학습 채점용 음소 시퀀스로 정규화한다.

    인벤토리에 정확히 있으면 그대로, 매핑 사전에 있으면 표준 음소로 변환, 둘 다 아니면
    원형 유지(백엔드의 `PhonemeNormalizer` 가 분해 책임을 가진다).

    Args:
        raw: 모델 토크나이저가 돌려준 음소/특수토큰 시퀀스.
        inventory: 우리 ARPABET 인벤토리(소문자) 집합. 빈 집합이면 매핑 사전만으로 판정한다.

    Returns:
        정리된 표준 음소 시퀀스. 빈 토큰과 드롭 대상은 제거된다.
    """
    out: List[str] = []
    for token in raw:
        if not token:
            continue
        cleaned = _SUFFIX_RE.sub("", token).strip().lower()
        if not cleaned or cleaned in _DROP_TOKENS:
            continue
        if cleaned in inventory:
            out.append(cleaned)
            continue
        mapped = _NON_STANDARD_MAP.get(cleaned)
        if mapped is not None:
            out.append(mapped)
            continue
        out.append(cleaned)
    return out
