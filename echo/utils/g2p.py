"""CMU 기반 결정적 baseline canonical 생성기.

백엔드 LLM 의 hallucination 위험과 비결정성을 보완하기 위한 결정적 1차 후보를 만든다.
문맥 의존 발음(예: `the`+vowel → `DH IY`) 같은 정교한 후처리는 백엔드 LLM 이 담당하며,
이 모듈은 단어 단독 표준 ARPABET 시퀀스만 돌려준다.

핵심 동작
- g2p_en.G2p 인스턴스는 NLTK 자원과 LSTM weight 를 동반해 init 비용이 큼 → 앱 부팅 시
  한 번만 init 해 재사용.
- 입력 텍스트는 g2p_en 의 `normalize_numbers` 로 숫자/서수/통화를 영어 단어로 풀고
  구두점을 떼어낸 뒤 공백/하이픈으로 split. contraction(`don't`)은 보존.
- 단어마다 G2p 호출 → 강세 숫자(AH0/AH1/AH2) 제거 → 학습 인벤토리(소문자) 와 정합되도록
  소문자 변환을 거쳐 SIL/SP 같은 silence 토큰을 드롭 → 응답 시 UPPERCASE 로 반환.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from g2p_en import G2p as _G2pEn
from g2p_en.expand import normalize_numbers

from echo.constants import SILENCE_TOKENS


# 단어 분리 패턴. 공백/하이픈을 경계로 두고 알파벳·apostrophe 만 유지해 구두점을 떼어낸다.
_WORD_SPLIT_RE = re.compile(r"[\s\-]+")
# 단어 좌우의 잔여 구두점 제거. 내부 apostrophe(don't)는 보존.
_PUNCT_STRIP_RE = re.compile(r"^[^A-Za-z']+|[^A-Za-z']+$")
# 강세 숫자(AH0/AH1/AH2) 제거 — 인벤토리는 강세 비포함.
_STRESS_RE = re.compile(r"\d+$")


@dataclass(frozen=True)
class G2pWord:
    word: str
    phonemes: List[str]


def _normalize_text(text: str) -> str:
    """숫자/서수/통화 → 영어 단어로 풀어 단어 토큰화에 안전한 텍스트를 만든다."""
    return normalize_numbers(text)


def _split_words(text: str) -> List[str]:
    """공백/하이픈 분리 후 단어 좌우 구두점을 떼어 빈 토큰을 거른다."""
    out: List[str] = []
    for raw in _WORD_SPLIT_RE.split(text):
        if not raw:
            continue
        stripped = _PUNCT_STRIP_RE.sub("", raw)
        if stripped:
            out.append(stripped)
    return out


def _strip_stress(phoneme: str) -> str:
    """ARPABET 끝의 강세 숫자를 제거. 자음/silence 처럼 숫자가 없으면 그대로."""
    return _STRESS_RE.sub("", phoneme)


class G2p:
    """g2p_en 의 G2p 인스턴스를 한 번 init 후 reuse 하는 thin wrapper."""

    def __init__(self, inventory: Optional[Set[str]] = None) -> None:
        # g2p_en 의 G2p() init 은 NLTK 자원 로드 + LSTM weight 로드로 1~2초 소요.
        # 앱 부팅 시 1회만 호출되도록 외부에서 보장한다.
        self._g2p = _G2pEn()
        # 인벤토리는 소문자 ARPABET (sil/blank 제외). 비어 있으면 통과시키고 호출 측이 책임.
        self._inventory: Set[str] = set(inventory or set())

    def convert(self, text: str) -> List[G2pWord]:
        """입력 텍스트를 단어별 ARPABET (UPPERCASE) 시퀀스로 변환한다."""
        if not text or not text.strip():
            return []
        normalized = _normalize_text(text)
        words = _split_words(normalized)
        results: List[G2pWord] = []
        for word in words:
            phonemes = self._convert_word(word)
            if not phonemes:
                # OOV 인데 LSTM 예측도 비면 단어 자체는 응답에 포함하되 phonemes 만 비운다.
                # 백엔드 LLM 이 이를 보고 refine 단계에서 보강할 수 있게 한다.
                results.append(G2pWord(word=word, phonemes=[]))
                continue
            results.append(G2pWord(word=word, phonemes=phonemes))
        return results

    def _convert_word(self, word: str) -> List[str]:
        """단일 단어를 ARPABET 시퀀스로 변환 + 인벤토리 정규화 적용."""
        raw = self._g2p(word)
        out: List[str] = []
        for tok in raw:
            base = _strip_stress(tok).strip()
            if not base:
                continue
            lower = base.lower()
            # silence 토큰(SIL/SP/SPN/PAU)은 단어 단독 발음에 끼면 잡음이므로 제거.
            if lower in SILENCE_TOKENS:
                continue
            # 인벤토리가 주어졌고 해당 토큰이 없으면 드롭 — 비표준 토큰이 채점 단계로 새지 않게.
            if self._inventory and lower not in self._inventory:
                continue
            out.append(lower.upper())
        return out
