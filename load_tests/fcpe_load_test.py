#!/usr/bin/env python3

import time
import random
from typing import Optional

import torch
from torchfcpe import spawn_bundled_infer_model
from torchfcpe.models_infer import InferCFNaiveMelPE

GPU = 'cuda:0'
IS_HALF = True

fcpe_model: Optional[InferCFNaiveMelPE] = None

def load_fcpe_model():
    global fcpe_model
    load_start_time = time.perf_counter()
    fcpe_model = spawn_bundled_infer_model(device=GPU)
    load_done_time = time.perf_counter()
    print(f'Loaded fcpe model in {(load_done_time - load_start_time) * 1000:.1f} ms')

C_input_wav_batch = torch.load('data/input_wav_batch_cropped.pt').to(GPU)
assert C_input_wav_batch.shape == torch.Size([10, 3200])

def fcpe_inference():
    t0 = time.perf_counter()
    # B = random.randint(1, 10)
    B = 10
    input_wav_batch = C_input_wav_batch[:B]

    t1 = time.perf_counter()
    f0_batch = fcpe_model.infer(
        input_wav_batch.to(GPU).float(),
        sr=16000,
        decoder_mode="local_argmax",
        threshold=0.006,
    )
    t2 = time.perf_counter()
    assert f0_batch.size(0) == B
    print(f'fcpe inference [B={input_wav_batch.size(0)}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms')

if __name__ == '__main__':
    load_fcpe_model()
    while True:
        fcpe_inference()
