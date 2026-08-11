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


def _chunking_for_fp32(config) -> tuple:
    """Segment sizes matching the precision we actually run at.

    `Config.device_config()` picks x_pad/x_query/x_center/x_max from `is_half`, and
    it does so **before** anything can override the precision. A card it fails to
    recognize (Quadro P2000 matches none of its name patterns) therefore keeps the
    fp16 "6 GB" layout — 3 s of padding per segment and no splitting below 65 s —
    while `--fp32` doubles the memory each activation takes. On a 5 GB card that
    combination is what a long phrase dies of: a 14.76 s reply reaches the decoder
    as 20.76 s in one piece, peaks at 1.16 GiB, and takes the whole stack down.

    These are upstream's own fp32 numbers, applied after the precision is settled.
    """
    return 1, 6, 38, 41


def _free_vram() -> None:
    """Hand the caching allocator's pool back to the driver.

    After an OOM torch keeps every segment it ever reserved, so this process goes on
    holding ~1.16 GiB instead of its usual 706 MiB — on a shared 5 GB card that is
    enough to keep the neighbours (Whisper in the agent) failing to allocate long
    after the phrase that caused it is gone. Observed twice on the прод machine:
    the agent then restart-looped until the whole box was rebooted.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                # noqa: BLE001 — cleanup must not mask the error
        logger.warning("empty_cache failed", exc_info=True)


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
        # how input gets split: x_max is the length above which it is split at all,
        # x_pad the padding added to every segment. Both drive peak VRAM.
        "chunking": _state.get("chunking"),
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
        _free_vram()      # an OOM here would otherwise poison the card for everyone
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
    p.add_argument("--max-segment", type=int, default=None, metavar="SECONDS",
                   help="longest piece the decoder gets; longer input is split at its "
                        "quietest points (fp32 default: 38). Sets x_center and x_max "
                        "together — x_max alone would never trigger, since the split "
                        "points themselves are placed every x_center seconds")
    p.add_argument("--x-pad", type=int, default=None, metavar="SECONDS",
                   help="padding added to every segment, counts twice per piece "
                        "(fp32 default: 1)")
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
        # ...and with it the segment sizes, which device_config() already chose for
        # fp16 and does not revisit. See _chunking_for_fp32 for what that costs.
        config.x_pad, config.x_query, config.x_center, config.x_max = _chunking_for_fp32(config)

    if args.max_segment is not None:
        # x_center is the spacing of the split points, x_max the length above which
        # splitting happens at all. Upstream keeps them 3 s apart; do the same, or a
        # phrase between the two would take the unsplit path anyway.
        config.x_center = args.max_segment
        config.x_max = args.max_segment + 3
        # x_query is the half-window the split point is searched in, and it has to fit
        # before the *first* one at t=x_center. Upstream's 6 s does not once segments
        # get short: the slice audio_sum[t - t_query : t + t_query] then starts at a
        # negative index, wraps to the end of the array, comes back empty, and .min()
        # raises "zero-size array to reduction operation minimum". The caller sees a
        # bare 500 and falls back to the Piper voice — for every long phrase, silently.
        config.x_query = max(1, min(config.x_query, args.max_segment // 2))
    if args.x_pad is not None:
        config.x_pad = args.x_pad
    logger.info(
        "chunking: x_pad=%s x_query=%s x_center=%s x_max=%s (peak VRAM grows with"
        " x_max + 2*x_pad, the longest piece the decoder ever sees)",
        config.x_pad, config.x_query, config.x_center, config.x_max,
    )

    # Pipeline reads these in its constructor, which runs inside the first get_vc()
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
        chunking={
            "x_pad": config.x_pad, "x_query": config.x_query,
            "x_center": config.x_center, "x_max": config.x_max,
        },
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
