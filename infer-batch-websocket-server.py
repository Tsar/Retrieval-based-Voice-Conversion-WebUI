#!/usr/bin/env python3

import os
import json
import time
import ssl
import uuid
import itertools
import logging
from urllib.parse import urlparse, parse_qs
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from multiprocessing import cpu_count
import asyncio
import websockets
import numpy as np

import fairseq.checkpoint_utils
from fairseq.models.hubert import HubertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as tat
from torchfcpe import spawn_bundled_infer_model
from torchfcpe.models_infer import InferCFNaiveMelPE

from infer.lib.jit.get_synthesizer import get_synthesizer

logger = logging.getLogger('infer-batch-websocket-server')

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

PORT = int(os.environ.get('PORT', 7411))

SSL_CERT = os.environ.get('SSL_CERT_FILENAME')
SSL_KEY  = os.environ.get('SSL_KEY_FILENAME')
ALLOW_UNENCRYPTED_SERVING = int(os.environ.get('ALLOW_UNENCRYPTED_SERVING', 0))

INPUT_VOICES_PITCH = {
    'coral': 4,
    'shimmer': 0,
    'sage': 5,
}

TARGET_VOICES_PITCH = {
    'voicevox_speaker_43': 8,
}

SUPPORTED_TARGET_VOICES = list(TARGET_VOICES_PITCH.keys())

MODEL_PTH_PATH = 'assets/weights/voicevox_speaker_43.pth'

GPU = 'cuda:0'
IS_HALF = True
MAX_INFERENCE_BATCH_SIZE = 10

SAMPLE_RATE = 24000  # for both input and output
BLOCK_DURATION_MS = 150
CROSSFADE_DURATION_MS = 80
EXTRA_TIME_MS = 2000  # TODO: Reduce

BLOCK_FRAME = SAMPLE_RATE * BLOCK_DURATION_MS // 1000
BLOCK_FRAME_16K = 16000 * BLOCK_FRAME // SAMPLE_RATE
F0_EXTRACTOR_FRAME = BLOCK_FRAME_16K + 800
BLOCK_SIZE_BYTES = BLOCK_FRAME * 2  # PCM16
CROSSFADE_FRAME = SAMPLE_RATE * CROSSFADE_DURATION_MS // 1000
ZC = SAMPLE_RATE // 100
SOLA_BUFFER_FRAME = min(CROSSFADE_FRAME, 4 * ZC)
SOLA_SEARCH_FRAME = ZC
EXTRA_FRAME = SAMPLE_RATE * EXTRA_TIME_MS // 1000
SKIP_HEAD = EXTRA_FRAME // ZC
RETURN_LENGTH = (BLOCK_FRAME + SOLA_BUFFER_FRAME + SOLA_SEARCH_FRAME) // ZC
PITCH_SHIFT = BLOCK_FRAME_16K // 160

INPUT_WAV_LEN = EXTRA_FRAME + CROSSFADE_FRAME + SOLA_SEARCH_FRAME + BLOCK_FRAME
INPUT_WAV_RES_LEN = 160 * INPUT_WAV_LEN // ZC
P_LEN = INPUT_WAV_RES_LEN // 160

F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)

RESAMPLER_TO_16K = tat.Resample(
    orig_freq=SAMPLE_RATE,
    new_freq=16000,
    dtype=torch.float32,
).to(GPU)

FADE_IN_WINDOW: torch.Tensor = (
    torch.sin(
        0.5 * np.pi * torch.linspace(0.0, 1.0, steps=SOLA_BUFFER_FRAME, device=GPU, dtype=torch.float32)
    ) ** 2
)
FADE_OUT_WINDOW: torch.Tensor = 1 - FADE_IN_WINDOW

class PreprocessContext:
    def __init__(self, pitch: int, formant_shift: float = 0.0):
        self.pitch = pitch
        self.formant_shift = formant_shift
        self.factor = pow(2, self.formant_shift / 12)
        self.return_length2 = int(np.ceil(RETURN_LENGTH * self.factor))
        self.input_wav: torch.Tensor = torch.zeros(INPUT_WAV_LEN, device=GPU, dtype=torch.float32)
        self.input_wav_res: torch.Tensor = torch.zeros(INPUT_WAV_RES_LEN, device=GPU, dtype=torch.float32)
        self.cache_pitch = torch.zeros(1024, device=GPU, dtype=torch.long)
        self.cache_pitchf = torch.zeros(1024, device=GPU, dtype=torch.float32)

    # TODO: Fill with zeros again on message end?

    def prepare_input_buffers(self, block_i16: bytes):
        block_f32 = np.frombuffer(block_i16, dtype=np.int16).astype(np.float32) / 32768.0
        assert block_f32.shape[0] == BLOCK_FRAME
        self.input_wav[:-BLOCK_FRAME] = self.input_wav[BLOCK_FRAME:].clone()
        self.input_wav[-BLOCK_FRAME:] = torch.from_numpy(block_f32).to(GPU)
        self.input_wav_res[:-BLOCK_FRAME_16K] = self.input_wav_res[BLOCK_FRAME_16K:].clone()
        self.input_wav_res[-BLOCK_FRAME_16K - 160:] = RESAMPLER_TO_16K(self.input_wav[-BLOCK_FRAME - 2 * ZC:])[160:]

    def update_pitch_caches(self, pitch: torch.Tensor, pitchf: torch.Tensor):
        self.cache_pitch[:-PITCH_SHIFT] = self.cache_pitch[PITCH_SHIFT:].clone()
        self.cache_pitchf[:-PITCH_SHIFT] = self.cache_pitchf[PITCH_SHIFT:].clone()
        self.cache_pitch[4 - pitch.shape[0]:] = pitch[3:-1]
        self.cache_pitchf[4 - pitch.shape[0]:] = pitchf[3:-1]

class PostprocessContext:
    def __init__(self):
        self.sola_buffer: torch.Tensor = torch.zeros(SOLA_BUFFER_FRAME, device=GPU, dtype=torch.float32)

class Task:
    def __init__(
        self,
        priority: int,
        sequence: int,
        is_last_for_message: bool,
        feats: torch.Tensor,
        cache_pitch: torch.Tensor,
        cache_pitchf: torch.Tensor,
        factor: float,
        return_length2: int,
        future: asyncio.Future,
    ):
        self.priority = priority
        self.sequence = sequence
        self.is_last_for_message = is_last_for_message
        self.feats = feats
        self.cache_pitch = cache_pitch
        self.cache_pitchf = cache_pitchf
        self.factor = factor
        self.return_length2 = return_length2
        self.future = future

    def __lt__(self, other) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.sequence < other.sequence

global_sequence = itertools.count()
executor = ThreadPoolExecutor(max_workers=cpu_count() * 2)  # TODO: Replace with ProcessPoolExecutor?

# Each target voice has its own priority queue
inference_queues = {}
for voice in SUPPORTED_TARGET_VOICES:
    inference_queues[voice] = asyncio.PriorityQueue()

result_resamplers = {}
result_resamplers_lock = Lock()

hubert_model: Optional[HubertModel] = None
fcpe_model: Optional[InferCFNaiveMelPE] = None
net_g_model: Optional[nn.Module] = None  # TODO: Support multiple net_g models
net_g_tgt_sr: Optional[int] = None

def load_hubert_model():
    global hubert_model
    load_start_time = time.perf_counter()

    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(GPU)
    if IS_HALF:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

    load_done_time = time.perf_counter()
    logger.info(f'Loaded Hubert model in {(load_done_time - load_start_time) * 1000:.1f} ms')

def load_fcpe_model():
    global fcpe_model
    load_start_time = time.perf_counter()
    fcpe_model = spawn_bundled_infer_model(device=GPU)
    load_done_time = time.perf_counter()
    logger.info(f'Loaded fcpe model in {(load_done_time - load_start_time) * 1000:.1f} ms')

def load_net_g_model(pth_path):
    global net_g_model, net_g_tgt_sr
    load_start_time = time.perf_counter()

    net_g_model, cpt = get_synthesizer(pth_path=pth_path, device=GPU)
    net_g_tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    assert cpt.get("f0", 1) == 1
    assert cpt.get("version", "v1") == "v2"
    if IS_HALF:
        net_g_model = net_g_model.half()
    else:
        net_g_model = net_g_model.float()

    load_done_time = time.perf_counter()
    logger.info(f'Loaded net_g model in {(load_done_time - load_start_time) * 1000:.1f} ms')

def extract_features(input_wav: torch.Tensor, version="v2"):
    with torch.no_grad():
        if IS_HALF:
            feats = input_wav.half().view(1, -1)
        else:
            feats = input_wav.float().view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(GPU).fill_(False)
        logits = hubert_model.extract_features(
            source=feats,
            padding_mask=padding_mask,
            output_layer=9 if version == "v1" else 12,
        )
        feats = hubert_model.final_proj(logits[0]) if version == "v1" else logits[0]
        feats = torch.cat((feats, feats[:, -1:, :]), 1)
    return feats

def get_f0_fcpe(input_wav: torch.Tensor, f0_up_key: float):
    f0 = fcpe_model.infer(
        input_wav.to(GPU).unsqueeze(0).float(),
        sr=16000,
        decoder_mode="local_argmax",
        threshold=0.006,
    )
    f0 *= pow(2, f0_up_key / 12)
    f0 = f0.float().to(GPU).squeeze()
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = torch.round(f0_mel).long()
    return f0_coarse, f0

# Should be executed sequentially for each context, can be executed in parallel for different contexts
def prepare_task_for_inference(
    context: PreprocessContext,
    block_i16: bytes,
    priority: int,
    is_last_for_message: bool,
    future: asyncio.Future,
) -> Task:
    t0 = time.perf_counter()
    context.prepare_input_buffers(block_i16=block_i16)
    input_wav = context.input_wav_res

    t1 = time.perf_counter()
    feats = extract_features(input_wav=input_wav)  # TODO: Use batched inference here if Hubert can

    t2 = time.perf_counter()
    pitch, pitchf = get_f0_fcpe(  # TODO: Use batched inference here if fcpe can
        input_wav=input_wav[-F0_EXTRACTOR_FRAME:],
        f0_up_key=context.pitch - context.formant_shift,
    )
    t3 = time.perf_counter()
    context.update_pitch_caches(pitch, pitchf)
    cache_pitch = context.cache_pitch[None, -P_LEN:]
    cache_pitchf = context.cache_pitchf[None, -P_LEN:] * context.return_length2 / RETURN_LENGTH

    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    feats = feats[:, :P_LEN, :]
    task = Task(
        priority=priority,
        sequence=next(global_sequence),
        is_last_for_message=is_last_for_message,
        feats=feats,
        cache_pitch=cache_pitch,
        cache_pitchf=cache_pitchf,
        factor=context.factor,
        return_length2=context.return_length2,
        future=future,
    )
    t4 = time.perf_counter()
    print(
        f'preprocessing: {(t4 - t0) * 1000:.1f} ms [hubert: {(t2 - t1) * 1000:.1f} ms, '
        f'get_f0_fcpe: {(t3 - t2) * 1000:.1f} ms, other: {(t1 - t0 + t4 - t3) * 1000:.1f} ms]'
    )
    return task

async def net_g_inference_worker(net_g: nn.Module, tasks_queue: asyncio.PriorityQueue):
    while True:
        t0 = time.perf_counter()
        task1: Task = await tasks_queue.get()
        tasks_batch = [task1]
        while len(tasks_batch) < MAX_INFERENCE_BATCH_SIZE:
            try:
                taskN: Task = tasks_queue.get_nowait()
                if taskN.return_length2 != task1.return_length2:
                    tasks_queue.put_nowait(taskN)  # put it back
                    logger.warning(
                        f'Stopped filling inference batch from priority queue at size {len(tasks_batch)} because '
                        f'next task has different return_length2: {taskN.return_length2} != {task1.return_length2}'
                    )
                    break
                tasks_batch.append(taskN)
            except asyncio.QueueEmpty:
                break
        B = len(tasks_batch)

        t1 = time.perf_counter()
        feats = torch.cat([task.feats for task in tasks_batch], dim=0)
        p_len = torch.full((B,), P_LEN, dtype=torch.long, device=GPU)
        cache_pitch = torch.cat([task.cache_pitch for task in tasks_batch], dim=0)
        cache_pitchf = torch.cat([task.cache_pitchf for task in tasks_batch], dim=0)
        sid = torch.zeros(B, dtype=torch.long, device=GPU)
        skip_head = torch.LongTensor([SKIP_HEAD])
        return_length = torch.LongTensor([RETURN_LENGTH])
        return_length2 = torch.LongTensor([task1.return_length2])

        t2 = time.perf_counter()
        with torch.no_grad():
            infered_audio_batch, _, _ = net_g.infer(
                feats,
                p_len,
                cache_pitch,
                cache_pitchf,
                sid,
                skip_head,
                return_length,
                return_length2,
            )
        assert infered_audio_batch.size(0) == B
        t3 = time.perf_counter()
        for task, infered_audio in zip(tasks_batch, infered_audio_batch):
            task.future.set_result(infered_audio.float())
        t4 = time.perf_counter()
        print(
            f'inference: {(t4 - t1) * 1000:.1f} ms [build batch: {(t2 - t1) * 1000:.1f} ms, '
            f'net_g.infer: {(t3 - t2) * 1000:.1f} ms, split res: {(t4 - t3) * 1000:.1f} ms], '
            f'waited: {(t1 - t0) * 1000:.1f} ms'
        )

def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result

# Should be executed sequentially for each context, can be executed in parallel for different contexts
def postprocess_inference_result(context: PostprocessContext, infered_audio: torch.Tensor, task: Task) -> bytes:
    t0 = time.perf_counter()
    upp_res = int(np.floor(task.factor * net_g_tgt_sr // 100))
    if upp_res != SAMPLE_RATE // 100:
        with result_resamplers_lock:
            if upp_res not in result_resamplers:
                logger.info(f'Created new resampler: {upp_res * 100} Hz -> {SAMPLE_RATE} Hz')
                result_resamplers[upp_res] = tat.Resample(
                    orig_freq=upp_res * 100,
                    new_freq=SAMPLE_RATE,
                    dtype=torch.float32,
                ).to(GPU)
            resampler = result_resamplers[upp_res]
        infered_audio = resampler(infered_audio[:, :RETURN_LENGTH * upp_res])  # TODO: Move to batch processing?
    infer_wav = infered_audio.squeeze()

    # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
    t1 = time.perf_counter()
    conv_input = infer_wav[None, None, :SOLA_BUFFER_FRAME + SOLA_SEARCH_FRAME]
    cor_nom = F.conv1d(conv_input, context.sola_buffer[None, None, :])
    cor_den = torch.sqrt(F.conv1d(conv_input**2, torch.ones(1, 1, SOLA_BUFFER_FRAME, device=GPU)) + 1e-8)
    sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
    infer_wav = infer_wav[sola_offset:]
    infer_wav[:SOLA_BUFFER_FRAME] = phase_vocoder(
        context.sola_buffer,
        infer_wav[:SOLA_BUFFER_FRAME],
        FADE_OUT_WINDOW,
        FADE_IN_WINDOW,
    )
    context.sola_buffer[:] = infer_wav[BLOCK_FRAME:BLOCK_FRAME + SOLA_BUFFER_FRAME]

    t2 = time.perf_counter()
    processed_f32 = infer_wav[:BLOCK_FRAME].t().cpu().numpy()
    processed_i16 = (np.clip(processed_f32, -1.0, 1.0 - 1.0 / 32768.0) * 32768.0).astype(np.int16).tobytes()
    t3 = time.perf_counter()
    print(
        f'postprocessing: {(t3 - t0) * 1000:.1f} ms [resample: {(t1 - t0) * 1000:.1f} ms, '
        f'SOLA: {(t2 - t1) * 1000:.1f} ms, f32->i16: {(t3 - t2) * 1000:.1f} ms]'
    )
    return processed_i16

def error_message(message, log_prefix='', details_to_log=None):
    if details_to_log:
        logger.error(f'{log_prefix}{message} "{details_to_log}"')
    else:
        logger.error(f'{log_prefix}{message}')
    return json.dumps({'error': message})

async def handler(websocket):
    session_id = str(uuid.uuid4())
    log_prefix = f'Session {session_id}: '
    logger.info(f'{log_prefix}Incoming connection')

    parsed_url = urlparse(websocket.request.path)
    if parsed_url.path != '/v1/voice_conversion':
        await websocket.send(error_message(f'Unsupported path {parsed_url.path}', log_prefix=log_prefix))
        return

    auth_header = websocket.request.headers.get('Authorization')
    if auth_header is None:
        await websocket.send(error_message('No Authorization header', log_prefix=log_prefix))
        return
    if not auth_header.startswith(BEARER_PREFIX):
        await websocket.send(error_message('Bad Authorization header', log_prefix=log_prefix, details_to_log=auth_header))
        return
    auth_token = auth_header[len(BEARER_PREFIX):]
    if auth_token != AUTH_TOKEN:
        await websocket.send(error_message('Bad authentication token', log_prefix=log_prefix, details_to_log=auth_token))
        return

    params = parse_qs(parsed_url.query)
    if 'target_voice' not in params:
        await websocket.send(error_message('No target_voice in query params', log_prefix=log_prefix, details_to_log=params))
        return
    target_voice = params['target_voice'][0]
    if target_voice not in TARGET_VOICES_PITCH:
        await websocket.send(error_message(
            f'Unsupported target_voice, only the following are supported: {list(TARGET_VOICES_PITCH.keys())}',
            log_prefix=log_prefix,
            details_to_log=params,
        ))
        return

    if 'transpose_by' in params:
        transpose_by_str = params['transpose_by'][0]
        try:
            transpose_by = int(transpose_by_str)
        except ValueError:
            await websocket.send(error_message('Bad transpose_by value', log_prefix=log_prefix, details_to_log=transpose_by_str))
            return
    elif 'input_voice' in params:
        input_voice = params['input_voice'][0]
        if input_voice not in INPUT_VOICES_PITCH:
            await websocket.send(error_message(
                f'Unsupported input_voice, only the following are supported: {list(INPUT_VOICES_PITCH.keys())}',
                log_prefix=log_prefix,
                details_to_log=params,
            ))
            return
        transpose_by = TARGET_VOICES_PITCH[target_voice] - INPUT_VOICES_PITCH[input_voice]
    else:
        await websocket.send(error_message('No transpose_by or input_voice in query params', log_prefix=log_prefix, details_to_log=params))
        return

    logger.info(f'{log_prefix}Starting voice conversion to {target_voice} transposed by {transpose_by}')

    buffer = b''
    preprocess_blocks_queue = asyncio.Queue()
    postprocess_tasks_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    try:
        async def preprocessing_loop():
            msg_start_ts_ms: Optional[int] = None
            block_num = 0
            context = PreprocessContext(pitch=transpose_by)
            while not stop_event.is_set():
                try:
                    block, is_last = await asyncio.wait_for(preprocess_blocks_queue.get(), timeout=1)
                    if msg_start_ts_ms is None:
                        msg_start_ts_ms = int(time.time() * 1000)

                    assert isinstance(block, bytes)
                    assert isinstance(is_last, bool)
                    assert len(block) == BLOCK_SIZE_BYTES

                    loop = asyncio.get_running_loop()
                    task = await loop.run_in_executor(
                        executor,
                        prepare_task_for_inference,
                        context,
                        block,
                        msg_start_ts_ms + BLOCK_DURATION_MS * block_num,
                        is_last,
                        loop.create_future(),
                    )
                    postprocess_tasks_queue.put_nowait(task)
                    inference_queues[target_voice].put_nowait(task)
                    block_num += 1

                    if is_last:
                        msg_start_ts_ms = None
                        block_num = 0
                except asyncio.TimeoutError:
                    continue  # periodically checking if we need to stop
            logger.info(f'{log_prefix}Preprocessing loop stopped gracefully')

        async def postprocessing_loop():
            context = PostprocessContext()
            while not stop_event.is_set():
                try:
                    task: Task = await asyncio.wait_for(postprocess_tasks_queue.get(), timeout=1)
                    result = await task.future

                    loop = asyncio.get_running_loop()
                    result_audio_block = await loop.run_in_executor(
                        executor,
                        postprocess_inference_result,
                        context,
                        result,
                        task,
                    )
                    await websocket.send(result_audio_block)
                    if task.is_last_for_message:
                        await websocket.send('end_message')
                except asyncio.TimeoutError:
                    continue  # periodically checking if we need to stop
            logger.info(f'{log_prefix}Postprocessing loop stopped gracefully')

        asyncio.create_task(preprocessing_loop())
        asyncio.create_task(postprocessing_loop())

        while True:
            data = await websocket.recv()
            if isinstance(data, bytes):
                buffer += data
                while len(buffer) >= BLOCK_SIZE_BYTES:
                    preprocess_blocks_queue.put_nowait((buffer[:BLOCK_SIZE_BYTES], False))
                    buffer = buffer[BLOCK_SIZE_BYTES:]
            elif isinstance(data, str):
                if data == 'end_message':
                    assert len(buffer) < BLOCK_SIZE_BYTES
                    preprocess_blocks_queue.put_nowait((buffer.ljust(BLOCK_SIZE_BYTES, b'\x00'), True))
                    buffer = b''
                else:
                    logger.error(f'{log_prefix}Unrecognized text received: "{data}"')
            else:
                await websocket.send(error_message('Received unrecognized data type', details_to_log=data))
                continue
    except websockets.exceptions.ConnectionClosedOK as ex:
        logger.info(f'{log_prefix}Disconnected OK: {ex}')
    except websockets.exceptions.ConnectionClosedError as ex:
        logger.error(f'{log_prefix}Disconnected with error: {ex}')
    finally:
        stop_event.set()

async def perform_warmup():
    block = bytes(BLOCK_SIZE_BYTES)
    context = PreprocessContext(pitch=4)
    for _ in range(3):
        task = prepare_task_for_inference(
            context=context,
            block_i16=block,
            priority=0,
            is_last_for_message=False,
            future=asyncio.get_running_loop().create_future(),
        )
        inference_queues['voicevox_speaker_43'].put_nowait(task)
        await task.future

async def main():
    ssl_context = None
    if SSL_CERT and SSL_KEY and os.path.isfile(SSL_CERT) and os.path.isfile(SSL_KEY):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(SSL_CERT, keyfile=SSL_KEY)
        logger.info('SSL certificates found, using encryption')
    else:
        if ALLOW_UNENCRYPTED_SERVING == 1:
            logger.warning('SSL certificates NOT FOUND, launching without encryption')
        else:
            logger.warning('SSL certificates NOT FOUND, unencrypted serving prohibited')
            return

    load_hubert_model()
    load_fcpe_model()
    load_net_g_model(pth_path=MODEL_PTH_PATH)  # TODO: Load multiple models

    # TODO: Create task for each voice
    asyncio.create_task(net_g_inference_worker(net_g_model, inference_queues['voicevox_speaker_43']))
    await perform_warmup()

    try:
        async with websockets.serve(handler, host='', port=PORT, ssl=ssl_context) as server:
            logger.info(f'WebSocket server started on port {PORT}')
            await server.serve_forever()
    except Exception as e:
        logger.error('Unhandled exception', exc_info=e)
    finally:
        logger.info('WebSocket server stopped')

if __name__ == '__main__':
    asyncio.run(main())
