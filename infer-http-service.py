#!/usr/bin/env python3
"""Offline voice conversion over HTTP: one request converts one whole utterance.

The websocket servers in this repo are built for realtime: fixed-size blocks with
heavy overlap, latency measured in tens of milliseconds. This service is the
opposite trade-off — it takes a complete phrase and converts it in a single pass,
which is what a non-realtime caller wants.

Written for the AI Radio Agent: it synthesizes a full reply with Piper, sends it
here to be re-voiced, and only then keys the transmitter. It lives in a separate
process (and a separate venv) because RVC needs numpy 1.23 + fairseq, while the
agent runs on numpy 2.x for faster-whisper and piper.

Usage:
    .venv/bin/python infer-http-service.py --voice tsar.pth --port 8081

Then:
    curl -s --data-binary @in.wav -H 'Content-Type: audio/wav' \\
         'http://127.0.0.1:8081/convert?pitch=0' -o out.wav
"""
import argparse
import io
import logging
import os
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("infer-http-service")
for noisy in ("numba", "matplotlib", "faiss", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI(title="RVC offline voice conversion")


class TargetVoice:
    """Same presets as infer-batch-websocket-server.py, so both paths sound alike."""

    def __init__(self, model: str, pitch: int, formant_shift: float = 0.0):
        self.model = model
        self.pitch = pitch
        self.formant_shift = formant_shift


TARGET_VOICES = {
    # formant_shift=1 picked by ear for Piper ru_RU-irina input: it thins the
    # timbre without the tempo running away (the phrase ends up ~6% shorter)
    "voicevox_speaker_43": TargetVoice(
        "voicevox_speaker_43.pth", pitch=8, formant_shift=1.0),
    "xiangling_eng": TargetVoice(
        "xiangling_eng_30_epochs_with_pitch.pth", pitch=12, formant_shift=1.0),
    "citlali_jap": TargetVoice("citlali_jap.pth", pitch=6),
}

# Transpose is target.pitch - INPUT_VOICES_PITCH[input voice]. In the realtime
# server shimmer sits at 0, and Piper's ru_RU-irina is about as low, so the
# preset pitch applies unchanged.
INPUT_VOICES_PITCH = {"shimmer": 0, "irina": 0, "coral": 4, "sage": 5}

# vc_single walks shared model state, so requests are serialized. Conversion is
# short (well under a second for a radio-length phrase) and callers are few.
_lock = threading.Lock()
_state = {}


def _select_voice(name: str):
    if name not in TARGET_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown voice {name!r}; available: {', '.join(TARGET_VOICES)}",
        )
    if _state.get("voice") != name:
        # switching reloads net_g (~0.4 s); hubert and rmvpe stay resident
        logger.info("switching voice %s -> %s", _state.get("voice"), name)
        _state["vc"].get_vc(TARGET_VOICES[name].model)
        _state["voice"] = name
    return TARGET_VOICES[name]


def _apply_formant_shift(audio: np.ndarray, factor: float) -> np.ndarray:
    """Speed the audio up by `factor`, lifting formants and pitch together.

    The realtime path gets this for free: it asks net_g for return_length*factor
    frames and plays them back over return_length. Offline there is no such knob,
    so the same shift is applied by resampling afterwards — at the cost of the
    phrase getting shorter by (1 - 1/factor), about 6% at formant_shift=1.
    """
    if abs(factor - 1.0) < 1e-6 or len(audio) == 0:
        return audio
    n_out = max(1, int(round(len(audio) / factor)))
    src = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    dst = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(dst, src, audio.astype(np.float64)).astype(audio.dtype)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "voice": _state.get("voice"),
        "voices": {
            n: {"pitch": v.pitch, "formant_shift": v.formant_shift}
            for n, v in TARGET_VOICES.items()
        },
        "index": _state.get("index") or None,
        "device": _state.get("device"),
        "is_half": _state.get("is_half"),
        "target_sr": _state.get("target_sr"),
    }


@app.post("/convert")
async def convert(
    request: Request,
    voice: str = Query(None, description=f"one of: {', '.join(TARGET_VOICES)}"),
    pitch: int = Query(None, description="transpose in semitones; preset value if omitted"),
    input_voice: str = Query(
        "irina", description=f"source voice for pitch correction: {', '.join(INPUT_VOICES_PITCH)}"),
    formant_shift: float = Query(None, description="preset value if omitted"),
    # pm по умолчанию: экстрактор F0 на praat, целиком на CPU. Нейросетевой rmvpe
    # держит ~335 МБ VRAM, а входом здесь идёт чистый синтез TTS, не шумная запись,
    # так что разницы на слух нет — тем более после узкой полосы рации.
    f0_method: str = Query("pm"),
    index_rate: float = Query(0.5, ge=0.0, le=1.0),
    filter_radius: int = Query(3, ge=0, le=7),
    rms_mix_rate: float = Query(0.25, ge=0.0, le=1.0),
    protect: float = Query(0.33, ge=0.0, le=0.5),
    resample_sr: int = Query(
        0, description="0 keeps the model rate (40k); set 16000 to get radio rate back"
    ),
):
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="empty request body, expected WAV")

    if input_voice not in INPUT_VOICES_PITCH:
        raise HTTPException(
            status_code=400,
            detail=f"unknown input_voice {input_voice!r}; "
                   f"available: {', '.join(INPUT_VOICES_PITCH)}",
        )

    with _lock:
        preset = _select_voice(voice or _state["voice"])
        shift = preset.formant_shift if formant_shift is None else formant_shift
        base_pitch = preset.pitch if pitch is None else pitch
        # same arithmetic as the realtime server
        transpose_by = base_pitch - INPUT_VOICES_PITCH[input_voice]
        f0_up_key = transpose_by - shift
        factor = pow(2, shift / 12)

        # vc_single takes a path (it decodes via ffmpeg), so input lands on disk first
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            started = time.perf_counter()
            info, (out_sr, audio) = _state["vc"].vc_single(
                0,                      # speaker id inside the model
                tmp_path,
                f0_up_key,
                None,                   # f0_file
                f0_method,
                _state["index"],        # file_index
                "",                     # file_index2
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
            )
            elapsed = time.perf_counter() - started
        finally:
            os.unlink(tmp_path)

    if audio is None:
        # vc_single swallows exceptions and returns the traceback as text
        logger.error("conversion failed: %s", info)
        raise HTTPException(status_code=500, detail=info)

    audio = _apply_formant_shift(audio, factor)

    buf = io.BytesIO()
    sf.write(buf, audio, out_sr, format="WAV")
    payload = buf.getvalue()
    logger.info(
        "%s: %.2f s -> %.2f s @ %d Hz in %.2f s (transpose %+d, formant %+.1f, %s)",
        preset.model, len(raw) / 32000.0, len(audio) / out_sr, out_sr,
        elapsed, transpose_by, shift, f0_method,
    )
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(out_sr),
            "X-Convert-Seconds": f"{elapsed:.3f}",
            "X-Voice": preset.model,
            "X-Transpose": str(transpose_by),
        },
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voice", default="voicevox_speaker_43", choices=list(TARGET_VOICES),
                   help="voice loaded at startup; requests may switch it")
    p.add_argument("--index", default="",
                   help="path to a .index file for retrieval (optional)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--fp32", action="store_true",
                   help="force fp32; Pascal cards are detected automatically")
    return p.parse_args()


def main():
    args = parse_args()

    # Config.__init__ runs its own argparse over sys.argv and would choke on ours
    sys.argv = [sys.argv[0]]

    load_dotenv()
    from configs.config import Config
    from infer.modules.vc.modules import VC

    config = Config()
    if args.fp32 and config.is_half:
        logger.info("forcing fp32 on request")
        config.is_half = False
        config.use_fp32_config()

    vc = VC(config)
    _state.update(vc=vc, index=args.index, voice=None)

    started = time.perf_counter()
    _select_voice(args.voice)
    preset = TARGET_VOICES[args.voice]
    logger.info(
        "loaded %s in %.1f s (device=%s, is_half=%s, target_sr=%s, pitch %+d, formant %+.1f)",
        preset.model, time.perf_counter() - started, config.device, config.is_half,
        vc.tgt_sr, preset.pitch, preset.formant_shift,
    )

    _state.update(
        device=str(config.device),
        is_half=bool(config.is_half),
        target_sr=vc.tgt_sr,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
