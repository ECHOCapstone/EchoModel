"""ECHO 모델 서버 - 발음 평가(/analyze) + G2P(/g2p) + TTS(/tts).

Run:
    # 모델 후보는 data/models.json 레지스트리에서 로드된다. 외부 노출은 ECHO_HOST=0.0.0.0 으로.
    python serve.py --models-config data/models.json --device cuda --port 8001

    # 레지스트리 없이 슬랩 모델만 띄우는 예:
    python serve.py \
      --slplab-model slplab/wav2vec2-large-robust-L2-english-phoneme-recognition \
      --device cuda --port 8001

모델 선택:
    data/models.json (--models-config) 에 후보 모델을 두고, /analyze 의 model 폼 필드로 고른다.
    요청 시 교체 로드되므로 한 번에 하나만 메모리에 올라간다. 후보는 GET /models 로 조회.

Endpoints:
    GET  /healthz   서버/모델/TTS/G2P 가용 상태 (active_model 포함)
    GET  /models    선택 가능한 음소인식 모델 후보 + 활성 모델
    GET  /phonemes  학습된 음소 리스트
    POST /analyze   발음 평가 - perceived/canonical/peak_softmax/alignment/errors/per (model 폼 필드로 모델 선택)
    POST /score     기존 호환 - recognized 키 사용
    POST /g2p       텍스트 → ARPAbet 음소 시퀀스 (CMUDict + OOV 신경망 폴백)
    POST /tts       텍스트 → mp3 음성 (gTTS)
    GET  /          데모 웹 UI
"""

from __future__ import annotations

import argparse
import datetime
import gc
import io
import json
import logging
import os
import pathlib
import threading
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from echo.constants import SILENCE_TOKENS
from echo.evaluation.metrics import get_alignment_with_indices
from echo.inference import PronunciationScorer
from echo.utils.audio import WAV2VEC2_SAMPLE_RATE
from echo.utils.g2p import G2P
from echo.utils.phoneme_sanitize import sanitize_perceived

logger = logging.getLogger("echo.serve")

# 추론 자원의 디폴트 위치 및 서버 포트.
# 모델 후보의 단일 출처는 data/models.json 레지스트리다. checkpoint 를 따로 지정하지 않으면("none")
# 레지스트리에서 모델을 로드하며, slplab 모델만으로도 서버가 동작한다. 머신 종속 절대경로를 기본값으로
# 박아두면 모델 재학습 시 즉시 깨지므로 기본값은 "none" 으로 둔다.
DEFAULT_CHECKPOINT = "none"
DEFAULT_PHONEME_MAP = "data/phoneme_to_id.json"
DEFAULT_PORT = 8001

# 업로드 오디오 최대 크기(바이트). 무제한 read 로 인한 메모리 고갈을 막는다. 환경변수로 조정 가능.
MAX_UPLOAD_BYTES = int(os.environ.get("ECHO_MAX_UPLOAD_BYTES", str(16 * 1024 * 1024)))

# slplab 비교 모델 (한국인 L2 영어 발음으로 fine-tune 된 wav2vec2-large-robust).
# 환경변수로 모델 ID / 디바이스 지정 가능. 로드하지 않으려면 ECHO_SLPLAB_MODEL=none 으로 설정.
DEFAULT_SLPLAB_MODEL = "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition"

# 음소인식 모델 레지스트리 파일. 어드민에서 고를 수 있는 모델 후보 목록의 단일 출처.
# 없으면 CLI 인자(--checkpoint / --slplab-model)로 1~2개를 합성한다.
DEFAULT_MODELS_CONFIG = "data/models.json"

# CORS 허용 origin. 기본값은 프론트엔드/mock 백엔드의 개발 origin 이며, 운영에서는
# ECHO_ALLOWED_ORIGINS(쉼표 구분)로 덮어쓴다.
_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173,http://127.0.0.1:8080"
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ECHO_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]

# gTTS는 선택 의존성. 미설치 환경에서도 추론 엔드포인트는 정상 동작한다.
try:
    from gtts import gTTS  # type: ignore
    _TTS_AVAILABLE = True
except Exception:
    gTTS = None  # type: ignore
    _TTS_AVAILABLE = False

# lifespan 동안 유지되는 추론기/G2P 핸들과 부팅 설정.
# registry: {"models": [entry...], "default": id}, active_model_id: 현재 메모리에 올라온 모델 id.
# arpabet_inventory: phoneme_to_id.json 에서 추출한 표준 음소 집합 — sanitize 가 인벤토리 외
# 토큰만 매핑/드롭하도록 한다.
state: dict = {
    "scorer": None,
    "g2p": None,
    "slplab_model": None,
    "slplab_processor": None,
    "config": {},
    "registry": {"models": [], "default": None},
    "active_model_id": None,
    "arpabet_inventory": frozenset(),
}


# 모델 교체와 추론을 직렬화하는 락. 모델 A 추론 도중 다른 요청이 _unload_current 를 호출해
# GPU 텐서가 GC 되거나 두 모델이 동시에 GPU 로 올라가 OOM 이 발생하는 race 를 차단한다.
# RLock 이라 같은 스레드에서 ensure_loaded 와 추론을 중첩으로 잡아도 안전하다.
_MODEL_SWITCH_LOCK = threading.RLock()


def _load_registry(cfg: dict) -> dict:
    """models.json 이 있으면 그걸로 모델 레지스트리를 만들고, 없으면 CLI 인자로 합성한다.
    반환: {"models": [entry...], "default": id}. entry = {id,label,type, checkpoint?,phoneme_map?,hf_id?}."""
    path = cfg.get("models_config", "")
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = [m for m in data.get("models", []) if isinstance(m, dict) and m.get("id")]
        default = data.get("default") or (models[0]["id"] if models else None)
        logger.info("Loaded model registry from %s (%d models, default=%s)", path, len(models), default)
        return {"models": models, "default": default}

    # 폴백: CLI 인자로 합성. checkpoint(ECHO) / slplab-model(slplab) 각각 한 개씩.
    models = []
    ckpt = cfg.get("checkpoint", "")
    if ckpt and ckpt.lower() != "none":
        models.append({"id": "echo", "label": "ECHO", "type": "echo",
                       "checkpoint": ckpt, "phoneme_map": cfg.get("phoneme_map", DEFAULT_PHONEME_MAP)})
    slp = cfg.get("slplab_model", "")
    if slp and slp.lower() != "none":
        models.append({"id": "slplab", "label": "slplab", "type": "slplab", "hf_id": slp})
    default = models[0]["id"] if models else None
    return {"models": models, "default": default}


def _entry_by_id(model_id: Optional[str]) -> Optional[dict]:
    if not model_id:
        return None
    for m in state["registry"].get("models", []):
        if m.get("id") == model_id:
            return m
    return None


def _unload_current() -> None:
    """현재 로드된 인식 모델을 내려 GPU/RAM 을 회수한다 (G2P 는 모델과 무관하므로 유지).

    호출자는 반드시 `_MODEL_SWITCH_LOCK` 을 잡은 상태여야 한다 — 추론 중인 다른 요청과의
    race 를 막기 위한 전제다.
    """
    state["scorer"] = None
    state["slplab_model"] = None
    state["slplab_processor"] = None
    state["active_model_id"] = None
    # 파이썬 GC 를 한 번 돌려 모델이 참조하던 텐서가 즉시 해제되도록 한다. 그 후 CUDA 캐시 회수.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(entry: dict) -> None:
    """레지스트리 entry 의 type 에 따라 인식 모델을 메모리에 올린다."""
    cfg = state["config"]
    mtype = entry.get("type")
    if mtype == "echo":
        state["scorer"] = PronunciationScorer.from_checkpoint(
            checkpoint_path=entry["checkpoint"],
            phoneme_map_path=entry.get("phoneme_map", cfg.get("phoneme_map", DEFAULT_PHONEME_MAP)),
            device=cfg["device"],
        )
    elif mtype == "slplab":
        device = cfg.get("slplab_device", cfg["device"])
        state["slplab_processor"] = Wav2Vec2Processor.from_pretrained(entry["hf_id"])
        state["slplab_model"] = Wav2Vec2ForCTC.from_pretrained(entry["hf_id"]).to(device).eval()
    else:
        raise HTTPException(400, f"unknown model type: {mtype}")
    state["active_model_id"] = entry["id"]
    logger.info("Recognition model ready: %s (type=%s)", entry["id"], mtype)


def ensure_loaded(model_id: Optional[str]) -> dict:
    """요청한 모델을 보장 로드한다. 이미 활성이면 그대로, 다르면 기존 것을 내리고 교체 로드한다.

    model_id 가 None 이면 활성 모델, 그것도 없으면 레지스트리 기본 모델을 쓴다. 활성 entry 를 반환.
    동시에 들어온 두 요청이 서로 다른 모델을 부르거나 한 요청이 추론 중인데 다른 요청이 교체를
    시도하는 race 를 막기 위해 `_MODEL_SWITCH_LOCK` 안에서만 상태를 갱신한다.
    """
    with _MODEL_SWITCH_LOCK:
        target_id = model_id or state["active_model_id"] or state["registry"].get("default")
        entry = _entry_by_id(target_id)
        if entry is None:
            raise HTTPException(400, f"unknown model: {target_id}")
        if state["active_model_id"] != entry["id"]:
            logger.info("Switching recognition model: %s -> %s", state["active_model_id"], entry["id"])
            _unload_current()
            _load_model(entry)
        return entry


# 디버그용 dump 디렉토리. 환경변수 ECHO_DUMP_DIR 가 설정되어 있을 때만 활성화.
# 활성화되면 매 요청마다 raw upload 와 denoise+VAD 적용 후 신호를 함께 떨어뜨려 비교 청취 가능.
_DUMP_DIR = os.environ.get("ECHO_DUMP_DIR")
# dump 디렉토리에 남길 최대 파일 개수. 환경변수 ECHO_DUMP_KEEP 으로 조정 가능 (기본 200).
# 초과 시 오래된 파일부터 삭제 — 무한 누적으로 디스크가 포화되는 것을 막는다.
_DUMP_KEEP = int(os.environ.get("ECHO_DUMP_KEEP", "200"))


def _prune_dump_dir(out_dir: pathlib.Path) -> None:
    """덤프 디렉토리의 파일 수가 _DUMP_KEEP 을 넘으면 오래된 것부터 삭제한다."""
    try:
        files = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        excess = len(files) - _DUMP_KEEP
        for stale in files[: max(0, excess)]:
            stale.unlink(missing_ok=True)
    except OSError as e:
        # 파일 정리는 best-effort — 실패해도 추론은 그대로 계속.
        logger.warning("Audio dump prune failed (ignored): %s", e)


def _dump_audio(raw: bytes, processed: torch.Tensor, sr: int, filename: Optional[str]) -> None:
    """원본 업로드 바이트와 전처리 후 텐서를 디스크에 함께 저장한다 (ECHO_DUMP_DIR 설정 시만).

    저장 후 디렉토리 파일 수가 한도(_DUMP_KEEP)를 넘으면 오래된 파일부터 정리해 누적을 막는다.
    파일명은 PurePath.stem 으로 정규화해 경로 분리자가 포함된 입력에도 안전하다.
    """
    if not _DUMP_DIR:
        return
    try:
        out_dir = pathlib.Path(_DUMP_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = pathlib.PurePath(filename or "rec").stem.replace("/", "_").replace("\\", "_") or "rec"
        raw_path = out_dir / f"{ts}_{stem}_raw.wav"
        proc_path = out_dir / f"{ts}_{stem}_processed.wav"
        raw_path.write_bytes(raw)
        sf.write(str(proc_path), processed.cpu().numpy(), sr)
        logger.info("Dumped audio: %s, %s", raw_path.name, proc_path.name)
        _prune_dump_dir(out_dir)
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
    """앞뒤 무음을 RMS 기반 VAD 로 제거. 중간 무음은 보존한다 (발음 사이 자연스러운 호흡).

    peak 가 _VAD_ABS_FLOOR 미만이면 트림하지 않는다 — 노트북 내장 마이크처럼 게인이 낮으면
    음성과 잡음의 peak 차이가 거의 없어, peak 기준 dBFS 임계가 발화를 통째로 잘라낼 수 있다.
    이 경우는 보수적으로 원본을 모델에 맡긴다 (잘못된 트림보다 안전).
    """
    if wav.numel() == 0:
        return wav
    peak = wav.abs().max().item()
    if peak < _VAD_ABS_FLOOR:
        return wav

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

    # 모두 무음 판정이거나 voiced 프레임이 전체의 5% 미만이면 신뢰할 수 없는 트림 — 원본을 그대로 보낸다.
    # 보수적인 안전망: 발화 자체가 통째로 잘리는 worst case 를 막는다.
    if not voiced.any() or voiced.sum().item() < max(2, int(0.05 * frames.shape[0])):
        return wav

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

    # G2P 는 인식 모델과 무관하게 독립 동작 — phoneme_map 만 있으면 로드한다.
    # 백엔드가 /g2p 로 canonical 시퀀스를 만들어 /analyze 에 보내므로 항상 필요하다.
    # 같은 파일을 sanitize 의 인벤토리로도 활용한다 (인벤토리 = blank/silence/<...> 제외한 표준 음소).
    pmap = cfg.get("phoneme_map", "")
    if pmap and pmap.lower() != "none" and os.path.isfile(pmap):
        state["g2p"] = G2P.from_phoneme_map(pmap)
        with open(pmap, "r", encoding="utf-8") as f:
            phoneme_map_data = json.load(f)
        state["arpabet_inventory"] = frozenset(
            k.lower() for k in phoneme_map_data.keys()
            if not k.startswith("<") and k.lower() not in SILENCE_TOKENS
        )
        logger.info(
            "G2P ready (phoneme_map=%s, inventory=%d phonemes)",
            pmap, len(state["arpabet_inventory"]),
        )

    # 모델 레지스트리를 읽고 기본 모델을 메모리에 올린다. 이후 /analyze?model=<id> 로 교체된다.
    # 기본 모델 로드 실패가 서버 부팅 자체를 막지 않게 한다 — /g2p / /tts 는 계속 동작하고
    # /analyze 만 active_model_id 가 None 인 상태로 503 을 돌려준다.
    state["registry"] = _load_registry(cfg)
    default_id = state["registry"].get("default")
    if default_id:
        try:
            ensure_loaded(default_id)
        except Exception as exc:
            logger.error("Failed to load default model '%s': %s", default_id, exc)
    else:
        logger.warning("No recognition model in registry — /analyze will return 503")

    yield
    with _MODEL_SWITCH_LOCK:
        _unload_current()
    state["g2p"] = None
    state["arpabet_inventory"] = frozenset()


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
        "ok": scorer is not None or state["slplab_model"] is not None,
        "active_model": state["active_model_id"],
        "device": str(scorer.device) if scorer else None,
        "num_phonemes": len(scorer.phoneme_to_id) if scorer else None,
        "tts_available": _TTS_AVAILABLE,
        "g2p_ready": state["g2p"] is not None,
        "slplab_ready": state["slplab_model"] is not None,
    }


@app.get("/models")
def models():
    """선택 가능한 음소인식 모델 후보 + 현재 활성 모델. 백엔드/어드민이 드롭다운을 그릴 때 쓴다."""
    reg = state["registry"]
    return {
        "active": state["active_model_id"],
        "models": [
            {"id": m["id"], "label": m.get("label", m["id"]), "type": m.get("type")}
            for m in reg.get("models", [])
        ],
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
    expected_model_id: Optional[str] = None,
) -> dict:
    """ECHO 추론기로 발음 평가. 디코딩 → 모델 호출 → 메타 부착의 공통 처리."""
    if not raw:
        raise HTTPException(400, "Empty audio upload")
    # 추론 전 모델 교체 race 를 막기 위해 lock 안에서 모델 참조를 캡처하고 추론까지 마친다.
    with _MODEL_SWITCH_LOCK:
        if expected_model_id and state["active_model_id"] != expected_model_id:
            ensure_loaded(expected_model_id)
        scorer = state["scorer"]
        if scorer is None:
            raise HTTPException(503, "Scorer not ready")
        waveform = _decode_upload(raw, scorer.sampling_rate)
        if waveform.numel() == 0:
            raise HTTPException(400, "Decoded audio is empty")
        _dump_audio(raw, waveform, scorer.sampling_rate, filename)

        result = scorer.score(waveform, canonical=canonical, keep_silence=keep_silence)
        result["filename"] = filename
        result["duration_sec"] = float(waveform.shape[0] / scorer.sampling_rate)
        return result


# 업로드를 최대 크기까지만 읽어 메모리 고갈을 막는다. 한도를 넘으면 413 으로 거절한다.
async def _read_limited(audio: UploadFile) -> bytes:
    data = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Audio upload exceeds {MAX_UPLOAD_BYTES} bytes")
    return data


@app.post("/score")
async def score(
    audio: UploadFile = File(...),
    canonical: Optional[str] = Form(None),
    keep_silence: bool = Form(False),
):
    """기존 호환 응답: recognized 키 + peak_softmax 포함. 모델 선택은 활성 모델을 그대로 따른다."""
    raw = await _read_limited(audio)
    # 활성 모델 그대로 사용 — expected_model_id 를 비워두면 추론 함수가 현재 active_model 만 검증한다.
    return await run_in_threadpool(
        _score_upload, raw, canonical, keep_silence, audio.filename, state["active_model_id"],
    )


# 발화 속도 분류 기준. 학습자 데이터를 토대로 다음과 같이 보정한다.
#   - normal phonemes/sec: 원어민 빠른 발화는 14 음소/초 수준이나, 한국인 L2 학습자는
#     보통 8~12 음소/초 영역에서 발화하므로 10.0 을 기준으로 둔다. 이 값을 낮추면 한국인
#     사용자가 정상 발화에 가까울수록 ratio 가 1.0 에 가까워져 잘못된 "느림" 분류가 줄어든다.
#   - fast/slow threshold: 정상 윈도우를 더 넉넉히 잡아 평소 페이스의 학습자가 매번 fast/slow 로
#     분류되는 것을 막는다 (0.55~1.8).
_PHONEMES_PER_SECOND_NORMAL = 10.0
_FAST_RATIO_THRESHOLD = 0.55
_SLOW_RATIO_THRESHOLD = 1.8


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


def _build_alignment(canonical_list: list[str], perceived_list: list[str]) -> dict:
    """Levenshtein 정렬로 기존 PronunciationScorer.diagnose 와 동일한 alignment/errors 구조를 만든다."""
    op_label = {"=": "correct", "S": "substitution", "D": "deletion", "I": "insertion"}
    align = get_alignment_with_indices(canonical_list, perceived_list)
    alignment, errors = [], []
    for op, ref_idx, hyp_idx in align:
        item = {
            "op": op_label[op],
            "canonical_index": ref_idx,
            "canonical": canonical_list[ref_idx] if ref_idx is not None else None,
            "recognized_index": hyp_idx,
            "recognized": perceived_list[hyp_idx] if hyp_idx is not None else None,
        }
        alignment.append(item)
        if op != "=":
            errors.append(item)
    num_errors = sum(1 for op, _, _ in align if op != "=")
    per = round(num_errors / len(canonical_list), 4) if canonical_list else 0.0
    return {"alignment": alignment, "errors": errors, "per": per}


def _analyze_with_slplab(
    raw: bytes,
    canonical: Optional[str],
    keep_silence: bool,
    filename: Optional[str],
    expected_model_id: Optional[str] = None,
) -> dict:
    """slplab 모델로 인식 후 ECHO 모델 응답과 동일한 형식으로 정렬·메타까지 묶어 돌려준다.

    sanitize 단계가 vocab 잡음을 흡수하므로 학습자에게 표준 ARPABET 시퀀스만 노출된다.
    keep_silence 가 False 면 silence 토큰을 제거해 ECHO 경로와 정합한 채점 결과를 만든다.
    """
    if not raw:
        raise HTTPException(400, "Empty audio upload")
    # slplab vocab 은 16kHz 고정이라 ECHO scorer 의 sampling_rate 가 아닌 상수를 사용한다.
    sr = WAV2VEC2_SAMPLE_RATE
    with _MODEL_SWITCH_LOCK:
        if expected_model_id and state["active_model_id"] != expected_model_id:
            ensure_loaded(expected_model_id)
        if state["slplab_model"] is None or state["slplab_processor"] is None:
            raise HTTPException(503, "slplab model not loaded")
        waveform = _decode_upload(raw, sr)
        if waveform.numel() == 0:
            raise HTTPException(400, "Decoded audio is empty")
        _dump_audio(raw, waveform, sr, filename)

        raw_tokens, raw_peaks = _slplab_recognize(waveform)

    # vocab 잡음 정리는 락 밖에서 — 모델 자원을 더 이상 만지지 않는다.
    inventory = state["arpabet_inventory"]
    sanitized: list[str] = []
    sanitized_peaks: list[float] = []
    for token, peak in zip(raw_tokens, raw_peaks):
        for clean_token in sanitize_perceived([token], inventory):
            sanitized.append(clean_token)
            sanitized_peaks.append(peak)

    if keep_silence:
        perceived = sanitized
        peak_softmax = sanitized_peaks
    else:
        kept = [(p, pk) for p, pk in zip(sanitized, sanitized_peaks) if p not in SILENCE_TOKENS]
        perceived = [p for p, _ in kept]
        peak_softmax = [pk for _, pk in kept]

    canonical_list: list[str] = []
    if canonical and canonical.strip():
        canonical_list = canonical.strip().split()

    diag = _build_alignment(canonical_list, perceived) if canonical_list else {
        "alignment": [], "errors": [], "per": 0.0,
    }
    duration_sec = float(waveform.shape[0] / sr)
    speech_rate, speech_rate_ratio = _classify_speech_rate(canonical_list, duration_sec)

    return {
        "perceived": perceived,
        "canonical": canonical_list,
        "peak_softmax": peak_softmax,
        "alignment": diag["alignment"],
        "errors": diag["errors"],
        "per": diag["per"],
        "duration_sec": duration_sec,
        "speech_rate": speech_rate,
        "speech_rate_ratio": speech_rate_ratio,
    }


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    canonical: Optional[str] = Form(None),
    keep_silence: bool = Form(False),
    model: Optional[str] = Form(None),
):
    """백엔드 연동용 응답: perceived 키 + peak_softmax + 정렬 결과 + 발화 속도 분류.

    model 폼 필드로 레지스트리의 모델 id 를 지정하면 그 모델로 (필요 시 교체 로드 후) 인식한다.
    생략하면 현재 활성 모델(없으면 레지스트리 기본 모델) 을 쓴다.
    """
    raw = await _read_limited(audio)
    entry = ensure_loaded(model)
    entry_id = entry["id"]

    # 추론 함수는 자체 락 안에서 active_model_id 가 expected 와 다르면 다시 ensure_loaded 를
    # 부르므로, ensure_loaded 와 추론 사이에 다른 요청이 모델을 바꿔도 안전하다.
    if entry.get("type") == "slplab":
        return await run_in_threadpool(
            _analyze_with_slplab, raw, canonical, keep_silence, audio.filename, entry_id,
        )

    # 블로킹 추론(STFT/CTC)을 스레드풀로 보내 이벤트 루프가 멈추지 않게 한다.
    raw_result = await run_in_threadpool(
        _score_upload, raw, canonical, keep_silence, audio.filename, entry_id,
    )
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


def _slplab_recognize(waveform: torch.Tensor) -> tuple[list[str], list[float]]:
    """slplab 모델 추론 후 토큰 시퀀스와 토큰별 softmax peak 를 함께 돌려준다.

    `batch_decode` 는 토큰 단위 confidence 를 노출하지 않으므로 frame-level argmax 를 직접
    duplicate/pad collapse 해 토큰 시퀀스와 같은 길이의 peak 배열을 만든다.
    """
    model = state["slplab_model"]
    proc = state["slplab_processor"]
    if model is None or proc is None:
        raise HTTPException(503, "slplab model not loaded")
    device = next(model.parameters()).device
    inputs = proc(
        waveform.cpu().numpy(),
        sampling_rate=WAV2VEC2_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits  # [B, T, V]

    probs = logits.softmax(dim=-1)
    peaks_per_frame, predicted_ids = probs.max(dim=-1)
    pred_ids = predicted_ids[0].tolist()
    pred_peaks = peaks_per_frame[0].tolist()

    tokenizer = proc.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

    tokens: list[str] = []
    peaks: list[float] = []
    prev = -1
    for tid, pk in zip(pred_ids, pred_peaks):
        if tid == pad_id:
            prev = tid
            continue
        if tid == prev:
            # 같은 라벨이 연속하는 프레임 묶음의 confidence 는 최댓값으로 갱신.
            if peaks and pk > peaks[-1]:
                peaks[-1] = float(pk)
            continue
        token_text = tokenizer.convert_ids_to_tokens(tid)
        if not token_text:
            prev = tid
            continue
        tokens.append(token_text)
        peaks.append(float(pk))
        prev = tid
    return tokens, peaks


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

    # gTTS 는 외부 네트워크 합성을 수행하는 블로킹 호출이라 스레드풀로 보낸다.
    def _synthesize() -> io.BytesIO:
        engine = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        engine.write_to_fp(buf)
        buf.seek(0)
        return buf

    try:
        buf = await run_in_threadpool(_synthesize)
    except Exception as e:
        # gTTS 는 외부 Google 서버에 의존하므로 네트워크 단절 / 쿼터 초과는 자체 장애가 아닌
        # 업스트림 가용성 문제다. 5xx 가 아닌 503 으로 분리해 호출 측이 재시도 정책을 정확히 세울 수 있게 한다.
        raise HTTPException(503, f"TTS upstream failed: {e}")
    return StreamingResponse(buf, media_type="audio/mpeg")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.environ.get("ECHO_CHECKPOINT", DEFAULT_CHECKPOINT))
    p.add_argument("--phoneme-map", default=os.environ.get("ECHO_PHONEME_MAP", DEFAULT_PHONEME_MAP))
    p.add_argument("--device", default=os.environ.get("ECHO_DEVICE", "cuda"))
    p.add_argument("--slplab-model", default=os.environ.get("ECHO_SLPLAB_MODEL", DEFAULT_SLPLAB_MODEL),
                    help="HuggingFace model ID for slplab comparison. 'none' to skip.")
    p.add_argument("--slplab-device", default=os.environ.get("ECHO_SLPLAB_DEVICE"),
                    help="Device for slplab model. Defaults to --device value.")
    p.add_argument("--models-config", default=os.environ.get("ECHO_MODELS_CONFIG", DEFAULT_MODELS_CONFIG),
                    help="음소인식 모델 레지스트리 JSON. 없으면 --checkpoint / --slplab-model 로 합성한다.")
    # 기본은 로컬 바인딩. 리버스 프록시 뒤에서 외부 노출하려면 ECHO_HOST=0.0.0.0 으로 지정한다.
    p.add_argument("--host", default=os.environ.get("ECHO_HOST", "127.0.0.1"))
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
        "slplab_model": args.slplab_model,
        "slplab_device": args.slplab_device or args.device,
        "models_config": args.models_config,
    }
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
