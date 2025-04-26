#!/usr/bin/env python3

import os
import sys
import time
import random
from typing import Optional

import torch
import torch.nn as nn

from torch.profiler import profile, record_function, ProfilerActivity

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
from infer.lib.jit.get_synthesizer import get_synthesizer

USE_PROFILER = bool(int(os.environ.get('USE_PROFILER', 0)))
RAND_BATCH_SIZE = bool(int(os.environ.get('RAND_BATCH_SIZE', 0)))
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 10))  # will always use max when not random

GPU = 'cuda:0'
IS_HALF = bool(int(os.environ.get('HALF', 1)))

P_LEN = 224
SKIP_HEAD = 200
RETURN_LENGTH = 20
RETURN_LENGTH2 = 20

net_g_model: Optional[nn.Module] = None

def load_net_g_model(pth_path):
    global net_g_model
    load_start_time = time.perf_counter()

    net_g_model, cpt = get_synthesizer(pth_path=pth_path, device=GPU)
    tgt_sr = cpt["config"][-1]
    assert tgt_sr == 40000
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    assert cpt.get("f0", 1) == 1
    assert cpt.get("version", "v1") == "v2"
    if IS_HALF:
        net_g_model = net_g_model.half()
    else:
        net_g_model = net_g_model.float()

    load_done_time = time.perf_counter()
    print(f'Loaded net_g model in {(load_done_time - load_start_time) * 1000:.1f} ms')

C_feats = torch.load('data/feats.pt').to(GPU).repeat(5, 1, 1)
C_feats = C_feats.half() if IS_HALF else C_feats.float()
C_p_len = torch.full((50,), P_LEN, dtype=torch.long, device=GPU)
C_cache_pitch = torch.load('data/cache_pitch.pt').to(GPU).repeat(5, 1)
C_cache_pitchf = torch.load('data/cache_pitchf.pt').to(GPU).repeat(5, 1)
C_sid = torch.zeros(50, dtype=torch.long, device=GPU)

skip_head = torch.LongTensor([SKIP_HEAD])
return_length = torch.LongTensor([RETURN_LENGTH])
return_length2 = torch.LongTensor([RETURN_LENGTH2])

def net_g_inference():
    t0 = time.perf_counter()
    if RAND_BATCH_SIZE:
        B = random.randint(1, MAX_BATCH_SIZE)
    else:
        B = MAX_BATCH_SIZE
    feats = C_feats[:B]
    p_len = C_p_len[:B]
    cache_pitch = C_cache_pitch[:B]
    cache_pitchf = C_cache_pitchf[:B]
    sid = C_sid[:B]

    t1 = time.perf_counter()
    with torch.no_grad():
        infered_audio_batch, _, _ = net_g_model.infer(
            feats,
            p_len,
            cache_pitch,
            cache_pitchf,
            sid,
            skip_head,
            return_length,
            return_length2,
        )
    t2 = time.perf_counter()
    assert infered_audio_batch.size(0) == B
    print(f'net_g inference [B={feats.size(0)}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms')

def net_g_inference_with_profiler():
    t0 = time.perf_counter()
    if RAND_BATCH_SIZE:
        B = random.randint(1, MAX_BATCH_SIZE)
    else:
        B = MAX_BATCH_SIZE
    feats = C_feats[:B]
    p_len = C_p_len[:B]
    cache_pitch = C_cache_pitch[:B]
    cache_pitchf = C_cache_pitchf[:B]
    sid = C_sid[:B]

    t1 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("net_g_inference"):
            with torch.no_grad():
                infered_audio_batch, _, _ = net_g_model.infer(
                    feats,
                    p_len,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                    skip_head,
                    return_length,
                    return_length2,
                )
    t2 = time.perf_counter()
    print('Top 10 by self CPU time total')
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
    print('Top 10 by self CUDA time total')
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
    prof.export_chrome_trace("net_g_trace.json")
    assert infered_audio_batch.size(0) == B
    print(f'net_g inference [B={feats.size(0)}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms')

if __name__ == '__main__':
    load_net_g_model('../assets/weights/voicevox_speaker_43.pth')
    while True:
        if USE_PROFILER:
            net_g_inference_with_profiler()
        else:
            net_g_inference()
