"""ECHO 모델 서버 - 발음 평가(/analyze) + G2P(/g2p) + TTS(/tts).

Run:
    python serve.py \
        --checkpoint experiments/wav2vec2-base/film/20260506_004510/checkpoints/best_mdd_f1.pth \
        --phoneme-map data/phoneme_to_id.json \
        --host 0.0.0.0 --port 8001
        -- device cuda

Endpoints:
    GET  /healthz   서버/모델/TTS/G2P 가용 상태
    GET  /phonemes  학습된 음소 리스트
    POST /analyze   발음 평가 - perceived/canonical/peak_softmax/alignment/errors/per
    POST /score     기존 호환 - recognized 키 사용
    POST /g2p       텍스트 → ARPAbet 음소 시퀀스 (CMUDict + OOV 신경망 폴백)
    POST /tts       텍스트 → mp3 음성 (gTTS)
    GET  /          데모 웹 UI
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from echo.inference import PronunciationScorer
from echo.utils.g2p import G2P

logger = logging.getLogger("echo.serve")

# 추론 자원의 디폴트 위치 및 서버 포트.
DEFAULT_CHECKPOINT = "experiments/wav2vec2-base/baseline/20260411_030102/checkpoints/best_mdd_f1.pth"
DEFAULT_PHONEME_MAP = "data/phoneme_to_id.json"
DEFAULT_PORT = 8001

# 프론트엔드 및 mock 백엔드의 개발 origin.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]

# gTTS는 선택 의존성. 미설치 환경에서도 추론 엔드포인트는 정상 동작한다.
try:
    from gtts import gTTS  # type: ignore
    _TTS_AVAILABLE = True
except Exception:
    gTTS = None  # type: ignore
    _TTS_AVAILABLE = False

# lifespan 동안 유지되는 추론기/G2P 핸들과 부팅 설정.
state: dict = {"scorer": None, "g2p": None, "config": {}}


def _decode_upload(raw: bytes, target_sr: int) -> torch.Tensor:
    """업로드 바이트를 1D float32 mono 텐서(target_sr)로 변환."""
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio: {e}. Upload WAV/FLAC/OGG.",
        )
    wav = torch.from_numpy(data)
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = state["config"]
    logger.info("Loading scorer from %s", cfg["checkpoint"])
    state["scorer"] = PronunciationScorer.from_checkpoint(
        checkpoint_path=cfg["checkpoint"],
        phoneme_map_path=cfg["phoneme_map"],
        device=cfg["device"],
    )
    # G2P 는 모델과 동일한 phoneme_map 을 SSOT 로 공유한다. 모델이 인식할 수 있는
    # 음소만 canonical 로 내보내야 정렬·오류 비교가 무결해진다.
    state["g2p"] = G2P.from_phoneme_map(cfg["phoneme_map"])
    logger.info(
        "Scorer ready on %s (tts_available=%s, g2p_ready=True)",
        state["scorer"].device, _TTS_AVAILABLE,
    )
    yield
    state["scorer"] = None
    state["g2p"] = None


app = FastAPI(title="ECHO Pronunciation Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    path = os.path.join(WEB_DIR, "index.html")
    if not os.path.isfile(path):
        return JSONResponse({"ok": True, "hint": "web/index.html not found"})
    return FileResponse(path)


@app.get("/healthz")
def healthz():
    scorer = state["scorer"]
    return {
        "ok": scorer is not None,
        "device": str(scorer.device) if scorer else None,
        "num_phonemes": len(scorer.phoneme_to_id) if scorer else None,
        "tts_available": _TTS_AVAILABLE,
        "g2p_ready": state["g2p"] is not None,
    }


@app.get("/phonemes")
def phonemes():
    scorer = state["scorer"]
    if scorer is None:
        raise HTTPException(503, "Scorer not ready")
    return {"phonemes": sorted(scorer.phoneme_to_id.keys())}


def _score_upload(
    raw: bytes,
    canonical: Optional[str],
    keep_silence: bool,
    filename: Optional[str],
) -> dict:
    # 모델 호출의 공통 처리 - 디코딩, 길이 검증, score 호출, 메타 추가.
    scorer = state["scorer"]
    if scorer is None:
        raise HTTPException(503, "Scorer not ready")
    if not raw:
        raise HTTPException(400, "Empty audio upload")
    waveform = _decode_upload(raw, scorer.sampling_rate)
    if waveform.numel() == 0:
        raise HTTPException(400, "Decoded audio is empty")

    result = scorer.score(waveform, canonical=canonical, keep_silence=keep_silence)
    result["filename"] = filename
    result["duration_sec"] = float(waveform.shape[0] / scorer.sampling_rate)
    return result


@app.post("/score")
async def score(
    audio: UploadFile = File(...),
    canonical: Optional[str] = Form(None),
    keep_silence: bool = Form(False),
):
    """기존 호환 응답: recognized 키 + peak_softmax 포함."""
    raw = await audio.read()
    return _score_upload(raw, canonical, keep_silence, audio.filename)


# 영어 평균 발화 속도 (음소/초). 사용자 데이터에서 fine-tune 가능.
# 이 값을 기준으로 expected_sec 을 계산하고 실제 duration 과 비율로 fast/normal/slow 를 분기한다.
_PHONEMES_PER_SECOND_NORMAL = 14.0
_FAST_RATIO_THRESHOLD = 0.6   # 실제 / 예상 < 0.6 이면 빠른 발화
_SLOW_RATIO_THRESHOLD = 1.6   # 실제 / 예상 > 1.6 이면 느린 발화


def _classify_speech_rate(canonical_phonemes: list, duration_sec: float) -> tuple[str, float]:
    """canonical 음소 개수 대비 실제 duration 비율로 발화 속도 레이블을 정한다."""
    if not canonical_phonemes or duration_sec <= 0.0:
        return ("normal", 1.0)
    expected_sec = max(0.2, len(canonical_phonemes) / _PHONEMES_PER_SECOND_NORMAL)
    ratio = duration_sec / expected_sec
    if ratio < _FAST_RATIO_THRESHOLD:
        return ("fast", round(ratio, 2))
    if ratio > _SLOW_RATIO_THRESHOLD:
        return ("slow", round(ratio, 2))
    return ("normal", round(ratio, 2))


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    canonical: Optional[str] = Form(None),
    keep_silence: bool = Form(False),
):
    """백엔드 연동용 응답: perceived 키 + peak_softmax + 정렬 결과 + 발화 속도 분류."""
    raw = await audio.read()
    raw_result = _score_upload(raw, canonical, keep_silence, audio.filename)
    speech_rate, speech_rate_ratio = _classify_speech_rate(
        raw_result["canonical"], raw_result["duration_sec"]
    )
    return {
        "perceived": raw_result["recognized"],
        "canonical": raw_result["canonical"],
        "peak_softmax": raw_result.get("peak_softmax", []),
        "alignment": raw_result["alignment"],
        "errors": raw_result["errors"],
        "per": raw_result["per"],
        "duration_sec": raw_result["duration_sec"],
        "speech_rate": speech_rate,
        "speech_rate_ratio": speech_rate_ratio,
    }


@app.post("/g2p")
async def g2p(text: str = Form(...)):
    """텍스트(단어 또는 문장) 를 모델 인벤토리에 맞춘 ARPAbet 음소 시퀀스로 변환한다.

    응답:
        phonemes: 공백으로 이어 붙인 전체 음소 시퀀스. /analyze 의 canonical 인자에 그대로 사용.
        words:    원문 단어와 그 단어의 음소 목록. UI 에서 단어별 강조에 사용.
    """
    converter = state["g2p"]
    if converter is None:
        raise HTTPException(503, "G2P not ready")
    if not text or not text.strip():
        raise HTTPException(400, "text is required")
    entries = converter.convert(text)
    return {
        "phonemes": " ".join(p for entry in entries for p in entry.phonemes),
        "words": [{"word": entry.word, "phonemes": entry.phonemes} for entry in entries],
    }


@app.post("/tts")
async def tts(text: str = Form(...), lang: str = Form("en")):
    """텍스트를 mp3 스트림으로 합성. gTTS 미설치 시 503."""
    if not _TTS_AVAILABLE:
        raise HTTPException(503, "TTS not available - install with: pip install gTTS")
    if not text or not text.strip():
        raise HTTPException(400, "text is required")
    try:
        engine = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        engine.write_to_fp(buf)
        buf.seek(0)
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")
    return StreamingResponse(buf, media_type="audio/mpeg")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.environ.get("ECHO_CHECKPOINT", DEFAULT_CHECKPOINT))
    p.add_argument("--phoneme-map", default=os.environ.get("ECHO_PHONEME_MAP", DEFAULT_PHONEME_MAP))
    p.add_argument("--device", default=os.environ.get("ECHO_DEVICE", "cuda"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    state["config"] = {
        "checkpoint": args.checkpoint,
        "phoneme_map": args.phoneme_map,
        "device": args.device,
    }
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
