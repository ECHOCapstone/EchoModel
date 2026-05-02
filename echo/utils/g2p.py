"""Grapheme-to-phoneme 변환을 모델 음소 인벤토리에 맞춰 정규화하여 제공.

배경
    추론 모델은 오디오만 받아 perceived 음소를 뱉을 뿐, 정답 음소(canonical)는 외부 입력이다.
    따라서 사용자가 임의의 영문 텍스트를 입력하면 누군가는 텍스트를 ARPAbet 시퀀스로
    바꿔 넣어 줘야 한다. 이 모듈이 그 책임을 단일 클래스로 가둔다.

설계
    - 모델 인벤토리(phoneme_to_id.json) 가 단일 진실. 출력은 항상 그 집합으로 필터링.
    - G2P 백엔드는 g2p_en (CMU Pronouncing Dictionary 룩업 + OOV 시 내장 신경망 폴백).
      이 모듈은 백엔드 교체에 대비해 G2P 추상만 노출한다.
    - stress digit (AA0/AA1/AA2 -> aa) 제거를 정규식 한 줄로 통일.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set


_STRESS_DIGITS = re.compile(r"\d+$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


@dataclass(frozen=True)
class WordPhonemes:
    """원문 단어와 그에 대응하는 ARPAbet 음소 목록."""

    word: str
    phonemes: List[str]


class G2P:
    """입력 영문 텍스트를 모델이 인식하는 ARPAbet 음소 시퀀스로 변환한다.

    인벤토리 외 음소는 결과에서 제외된다. 비단어 토큰(구두점, 공백) 은 단어 경계로만
    쓰이고 음소 출력에는 포함되지 않는다.
    """

    def __init__(self, phoneme_inventory: Set[str]):
        self._inventory = {p.lower() for p in phoneme_inventory}
        self._engine = None

    @classmethod
    def from_phoneme_map(cls, phoneme_map_path: str) -> "G2P":
        """phoneme_to_id.json 의 키 집합을 모델 인벤토리로 사용한다."""
        path = Path(phoneme_map_path)
        with path.open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        return cls(set(mapping.keys()))

    def convert(self, text: str) -> List[WordPhonemes]:
        """원문을 단어 단위로 쪼개 각 단어의 음소 시퀀스를 채운다."""
        engine = self._ensure_engine()
        results: List[WordPhonemes] = []
        for word in _WORD_RE.findall(text):
            tokens = engine(word)
            phonemes = self._sanitize(tokens)
            results.append(WordPhonemes(word=word, phonemes=phonemes))
        return results

    def convert_flat(self, text: str) -> str:
        """모든 단어의 음소를 공백 한 칸으로 이어 붙인 문자열을 반환한다.

        모델 서버 /analyze 의 canonical 인자 포맷과 일치한다.
        """
        flat: List[str] = []
        for entry in self.convert(text):
            flat.extend(entry.phonemes)
        return " ".join(flat)

    def _sanitize(self, tokens: Iterable[str]) -> List[str]:
        """g2p_en 출력에서 stress digit 을 떼고 인벤토리에 속한 음소만 남긴다."""
        cleaned: List[str] = []
        for token in tokens:
            normalized = _STRESS_DIGITS.sub("", token).lower()
            if normalized in self._inventory:
                cleaned.append(normalized)
        return cleaned

    def _ensure_engine(self):
        """g2p_en 의 G2p 인스턴스를 한 번만 생성해 재사용한다.

        라이브러리 import 비용과 초기 nltk 데이터 로딩이 무겁기 때문에
        실제 호출 시점까지 미루고, 한 번 만들면 프로세스 동안 보관한다.
        """
        if self._engine is None:
            from g2p_en import G2p  # 지연 import: 의존성 부재 시 생성 시점에만 실패.
            self._engine = G2p()
        return self._engine


__all__ = ["G2P", "WordPhonemes"]
