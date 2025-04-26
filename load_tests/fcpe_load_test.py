#!/usr/bin/env python3

import os
import time
import random
from typing import Optional

import torch
from torchfcpe import spawn_bundled_infer_model
from torchfcpe.models_infer import InferCFNaiveMelPE

from torch.profiler import profile, record_function, ProfilerActivity

USE_PROFILER = bool(int(os.environ.get('USE_PROFILER', 0)))
RAND_BATCH_SIZE = bool(int(os.environ.get('RAND_BATCH_SIZE', 0)))
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 10))  # will always use max when not random

GPU = 'cuda:0'

fcpe_model: Optional[InferCFNaiveMelPE] = None

def load_fcpe_model():
    global fcpe_model
    load_start_time = time.perf_counter()
    fcpe_model = spawn_bundled_infer_model(device=GPU)
    load_done_time = time.perf_counter()
    print(f'Loaded fcpe model in {(load_done_time - load_start_time) * 1000:.1f} ms')

C_input_wav_batch = torch.load('data/input_wav_batch_cropped.pt').float().to(GPU).repeat(10, 1)
assert C_input_wav_batch.shape == torch.Size([100, 3200])

def fcpe_inference():
    t0 = time.perf_counter()
    if RAND_BATCH_SIZE:
        B = random.randint(1, MAX_BATCH_SIZE)
    else:
        B = MAX_BATCH_SIZE
    input_wav_batch = C_input_wav_batch[:B]

    t1 = time.perf_counter()
    f0_batch = fcpe_model.infer(
        input_wav_batch,
        sr=16000,
        decoder_mode="local_argmax",
        threshold=0.006,
    )
    t2 = time.perf_counter()
    assert f0_batch.size(0) == B
    print(f'fcpe inference [B={input_wav_batch.size(0)}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms')

def fcpe_inference_with_profiler():
    t0 = time.perf_counter()
    if RAND_BATCH_SIZE:
        B = random.randint(1, MAX_BATCH_SIZE)
    else:
        B = MAX_BATCH_SIZE
    input_wav_batch = C_input_wav_batch[:B]

    t1 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("fcpe_inference"):
            f0_batch = fcpe_model.infer(
                input_wav_batch.to(GPU).float(),
                sr=16000,
                decoder_mode="local_argmax",
                threshold=0.006,
            )
    t2 = time.perf_counter()
    print('Top 10 by self CPU time total')
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
    print('Top 10 by self CUDA time total')
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
    prof.export_chrome_trace("fcpe_trace.json")
    assert f0_batch.size(0) == B
    print(f'fcpe inference [B={input_wav_batch.size(0)}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms')

if __name__ == '__main__':
    load_fcpe_model()
    while True:
        if USE_PROFILER:
            fcpe_inference_with_profiler()
        else:
            fcpe_inference()
