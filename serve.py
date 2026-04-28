"""ECHO 모델 서버 - 발음 평가(/analyze) + TTS(/tts).

Run:
    python serve.py \
        --checkpoint experiments/wav2vec2-base/baseline/20260411_030102/checkpoints/best_mdd_f1.pth \
        --phoneme-map data/phoneme_to_id.json \
        --host 0.0.0.0 --port 8001

Endpoints:
    GET  /healthz   서버/모델/TTS 가용 상태
    GET  /phonemes  학습된 음소 리스트
    POST /analyze   발음 평가 - perceived/canonical/peak_softmax/alignment/errors/per
    POST /score     기존 호환 - recognized 키 사용
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

# lifespan 동안 유지되는 추론기 핸들과 부팅 설정.
state: dict = {"scorer": None, "config": {}}


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
    logger.info("Scorer ready on %s (tts_available=%s)", state["scorer"].device, _TTS_AVAILABLE)
    yield
    state["scorer"] = None


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


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    canonical: Optional[str] = Form(None),
    keep_silence: bool = Form(False),
):
    """백엔드 연동용 응답: perceived 키 + peak_softmax + 정렬 결과."""
    raw = await audio.read()
    raw_result = _score_upload(raw, canonical, keep_silence, audio.filename)
    return {
        "perceived": raw_result["recognized"],
        "canonical": raw_result["canonical"],
        "peak_softmax": raw_result.get("peak_softmax", []),
        "alignment": raw_result["alignment"],
        "errors": raw_result["errors"],
        "per": raw_result["per"],
        "duration_sec": raw_result["duration_sec"],
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
