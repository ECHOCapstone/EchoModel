"""ECHO 모델 서버 - 발음 평가(/analyze) + G2P(/g2p) + TTS(/tts).

Run:
    ECHO_DUMP_DIR=/home/syh/workspace/ECHO_model/_dumps serve.py \
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
import datetime
import io
import logging
import os
import pathlib
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from echo.inference import PronunciationScorer
from echo.utils.g2p import G2P

logger = logging.getLogger("echo.serve")

# 추론 자원의 디폴트 위치 및 서버 포트.
DEFAULT_CHECKPOINT = "experiments/wav2vec2-base/baseline/20260411_030102/checkpoints/best_mdd_f1.pth"
DEFAULT_PHONEME_MAP = "data/phoneme_to_id.json"
DEFAULT_PORT = 8001

# slplab 비교 모델 (한국인 L2 영어 발음으로 fine-tune 된 wav2vec2-large-robust).
# 환경변수로 모델 ID / 디바이스 지정 가능. 로드하지 않으려면 ECHO_SLPLAB_MODEL=none 으로 설정.
DEFAULT_SLPLAB_MODEL = "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition"

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
state: dict = {"scorer": None, "g2p": None, "slplab_model": None, "slplab_processor": None, "config": {}}


# 디버그용 dump 디렉토리. 환경변수 ECHO_DUMP_DIR 가 설정되어 있을 때만 활성화.
# 활성화되면 매 요청마다 raw upload 와 denoise+VAD 적용 후 신호를 함께 떨어뜨려 비교 청취 가능.
_DUMP_DIR = os.environ.get("ECHO_DUMP_DIR")


def _dump_audio(raw: bytes, processed: torch.Tensor, sr: int, filename: Optional[str]) -> None:
    """원본 업로드 바이트와 전처리 후 텐서를 디스크에 함께 저장한다 (ECHO_DUMP_DIR 설정 시만)."""
    if not _DUMP_DIR:
        return
    try:
        out_dir = pathlib.Path(_DUMP_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = (filename or "rec").rsplit(".", 1)[0].replace("/", "_").replace("\\", "_")
        raw_path = out_dir / f"{ts}_{stem}_raw.wav"
        proc_path = out_dir / f"{ts}_{stem}_processed.wav"
        raw_path.write_bytes(raw)
        sf.write(str(proc_path), processed.cpu().numpy(), sr)
        logger.info("Dumped audio: %s, %s", raw_path.name, proc_path.name)
    except Exception as e:
        logger.warning("Audio dump failed (ignored): %s", e)


# 잡음 / 무음 제거 파라미터. 학습자가 "녹음 시작" 직후 망설이며 생기는 앞쪽 무음과
# 발음 종료 후 종료 버튼까지의 뒤쪽 무음을 컷해 wav2vec2 인식률을 끌어올린다.
# 너무 공격적이면 onset/offset 자음 (특히 무성 파열음 /p/ /t/ /k/) 이 잘리므로 가드 패딩을 둔다.
_VAD_FRAME_MS = 25         # 프레임 길이
_VAD_HOP_MS = 10           # 프레임 홉
_VAD_DB_THRESH = -40.0     # peak 대비 -40dB 미만이면 무음으로 본다
_VAD_ABS_FLOOR = 1e-3      # peak 자체가 너무 작으면 무음 트림 자체를 포기 (전구간 잡음일 가능성)
_VAD_PAD_MS = 80           # 트림 후 양끝에 다시 붙여주는 가드 패딩

# 배경 잡음 제거 파라미터. 보수적 — wav2vec2 가 학습한 도메인에서 너무 벗어나지 않게.
# 1) HPF: 80Hz 미만 (에어컨 / 팬 / 험) 제거. 음성 fundamental 은 80Hz 이상.
# 2) Spectral gating: 처음 _DENOISE_PROFILE_MS 를 잡음 프로파일로 잡아, 각 frequency bin 에서
#    프로파일 평균 + k*std 이하는 _DENOISE_FLOOR 비율까지만 남긴다 (완전 제거하면 musical noise 발생).
_DENOISE_HPF_HZ = 80.0
_DENOISE_FRAME_MS = 32
_DENOISE_HOP_MS = 8
_DENOISE_PROFILE_MS = 200  # 처음 0.2초를 잡음 프로파일로 가정
_DENOISE_K = 1.5           # noise floor 위로 k*std 까지는 잡음으로 본다
_DENOISE_FLOOR = 0.1       # 잡음 bin 도 10% 는 남겨 artifact 줄이기


def _denoise(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """저주파 험 제거 + STFT spectral gating 으로 정상 잡음을 약하게 줄인다.

    프로파일링 가능한 분량 (>= _DENOISE_PROFILE_MS) 이 없거나, 신호가 너무 짧으면 HPF 만 적용한다.
    """
    if wav.numel() == 0:
        return wav
    # 1) Highpass: 험·팬 같은 60~80Hz 저주파 잡음 제거.
    wav = torchaudio.functional.highpass_biquad(wav, sample_rate=sr, cutoff_freq=_DENOISE_HPF_HZ)

    profile_samples = int(sr * _DENOISE_PROFILE_MS / 1000)
    n_fft = max(256, int(sr * _DENOISE_FRAME_MS / 1000))
    hop = max(64, int(sr * _DENOISE_HOP_MS / 1000))
    if wav.shape[0] < profile_samples + n_fft * 2:
        return wav  # 신호가 너무 짧으면 spectral gating 생략

    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = spec.abs()
    phase = spec / (mag + 1e-10)

    # 프로파일 구간 (시작 _DENOISE_PROFILE_MS) 의 평균/표준편차를 freq bin 별로.
    profile_frames = max(1, profile_samples // hop)
    noise_mag = mag[:, :profile_frames]
    noise_mean = noise_mag.mean(dim=1, keepdim=True)
    noise_std = noise_mag.std(dim=1, keepdim=True)
    threshold = noise_mean + _DENOISE_K * noise_std

    # threshold 미만은 _DENOISE_FLOOR 비율로 감쇠, 그 이상은 그대로 둔다.
    mask = torch.where(mag > threshold, torch.ones_like(mag), torch.full_like(mag, _DENOISE_FLOOR))
    cleaned_spec = mag * mask * phase
    cleaned = torch.istft(cleaned_spec, n_fft=n_fft, hop_length=hop, window=window, length=wav.shape[0])
    return cleaned


def _trim_silence(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """앞뒤 무음을 RMS 기반 VAD 로 제거. 중간 무음은 보존한다 (발음 사이 자연스러운 호흡)."""
    if wav.numel() == 0:
        return wav
    peak = wav.abs().max().item()
    if peak < _VAD_ABS_FLOOR:
        return wav  # 전체가 너무 작다 → 트림하지 않고 그대로 (모델이 판단)

    frame_len = max(1, int(sr * _VAD_FRAME_MS / 1000))
    hop_len = max(1, int(sr * _VAD_HOP_MS / 1000))
    pad_len = int(sr * _VAD_PAD_MS / 1000)

    # 프레임별 RMS 를 dBFS (peak 기준) 로 환산해 thresh 와 비교한다.
    frames = wav.unfold(0, frame_len, hop_len)  # [num_frames, frame_len]
    if frames.shape[0] == 0:
        return wav
    rms = frames.pow(2).mean(dim=1).clamp_min(1e-12).sqrt()
    db = 20.0 * torch.log10(rms / peak)
    voiced = db > _VAD_DB_THRESH

    if not voiced.any():
        return wav  # 모두 무음 판정 → 그대로 보낸다 (잘못된 트림보다 안전)

    first = int(voiced.nonzero()[0].item())
    last = int(voiced.nonzero()[-1].item())
    start = max(0, first * hop_len - pad_len)
    end = min(wav.shape[0], last * hop_len + frame_len + pad_len)
    return wav[start:end]


def _decode_upload(raw: bytes, target_sr: int) -> torch.Tensor:
    """업로드 바이트를 1D float32 mono 텐서(target_sr)로 변환하고 앞뒤 무음을 트림한다."""
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
    # 잡음 → 무음 트림 순서. 잡음을 먼저 줄여야 VAD 가 발화 구간을 더 정확히 잡는다.
    wav = _denoise(wav, target_sr)
    before = wav.shape[0]
    wav = _trim_silence(wav, target_sr)
    after = wav.shape[0]
    if before != after:
        logger.info(
            "VAD trim: %.2fs -> %.2fs (cut %.2fs)",
            before / target_sr, after / target_sr, (before - after) / target_sr,
        )
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
    # slplab 비교 모델 (선택 로드). ECHO_SLPLAB_MODEL=none 이면 건너뛴다.
    slplab_model_id = os.environ.get("ECHO_SLPLAB_MODEL", DEFAULT_SLPLAB_MODEL)
    if slplab_model_id and slplab_model_id.lower() != "none":
        slplab_device = os.environ.get("ECHO_SLPLAB_DEVICE", cfg["device"])
        logger.info("Loading slplab model: %s on %s", slplab_model_id, slplab_device)
        state["slplab_processor"] = Wav2Vec2Processor.from_pretrained(slplab_model_id)
        state["slplab_model"] = Wav2Vec2ForCTC.from_pretrained(slplab_model_id).to(slplab_device).eval()
        logger.info("slplab model ready (%d params)", sum(p.numel() for p in state["slplab_model"].parameters()))
    yield
    state["scorer"] = None
    state["g2p"] = None
    state["slplab_model"] = None
    state["slplab_processor"] = None


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
        "slplab_ready": state["slplab_model"] is not None,
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
    _dump_audio(raw, waveform, scorer.sampling_rate, filename)

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


def _slplab_recognize(waveform: torch.Tensor) -> list[str]:
    """slplab 모델로 음소 시퀀스 인식. _err 태그 포함 그대로 반환한다."""
    model = state["slplab_model"]
    proc = state["slplab_processor"]
    if model is None or proc is None:
        raise HTTPException(503, "slplab model not loaded")
    device = next(model.parameters()).device
    inputs = proc(waveform.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    raw_text = proc.batch_decode(predicted_ids)[0]
    return raw_text.strip().split() if raw_text.strip() else []


def _simple_per(perceived: list[str], canonical: list[str]) -> float:
    """Levenshtein 기반 PER. 두 시퀀스를 비교해 음소 오류율을 반환한다."""
    if not canonical:
        return 0.0 if not perceived else 1.0
    n, m = len(perceived), len(canonical)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if perceived[i - 1] == canonical[j - 1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
    return round(dp[m] / m, 4)


@app.post("/analyze-slplab")
async def analyze_slplab(
    audio: UploadFile = File(...),
    canonical: Optional[str] = Form(None),
):
    """slplab (한국인 L2 영어 fine-tune) 모델로 음소 인식. 기존 ECHO 모델과 비교 테스트용.

    응답의 perceived 에 _err 접미사가 붙은 음소는 모델이 오류로 판정한 발음이다.
    perceived_clean 은 _err 제거 후 순수 음소 시퀀스, per 은 perceived_clean 과 canonical 비교.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty audio upload")
    scorer = state["scorer"]
    sr = scorer.sampling_rate if scorer else 16000
    waveform = _decode_upload(raw, sr)
    if waveform.numel() == 0:
        raise HTTPException(400, "Decoded audio is empty")
    _dump_audio(raw, waveform, sr, audio.filename)

    perceived = _slplab_recognize(waveform)
    perceived_clean = [p.replace("_err", "") for p in perceived]
    error_indices = [i for i, p in enumerate(perceived) if "_err" in p]

    canonical_list: list[str] = []
    if canonical and canonical.strip():
        canonical_list = canonical.strip().split()

    per = _simple_per(perceived_clean, canonical_list) if canonical_list else None

    return {
        "model": "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition",
        "perceived": perceived,
        "perceived_clean": perceived_clean,
        "error_indices": error_indices,
        "canonical": canonical_list,
        "per": per,
        "duration_sec": float(waveform.shape[0] / sr),
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
