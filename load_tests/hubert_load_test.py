#!/usr/bin/env python3

import time
import random
from typing import Optional

import fairseq.checkpoint_utils
from fairseq.models.hubert import HubertModel
import torch

GPU = 'cuda:0'
IS_HALF = True

hubert_model: Optional[HubertModel] = None

def load_hubert_model():
    global hubert_model
    load_start_time = time.perf_counter()

    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ["../assets/hubert/hubert_base.pt"],
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
    print(f'Loaded Hubert model in {(load_done_time - load_start_time) * 1000:.1f} ms')

C_input_wav_batch = torch.load('data/input_wav_batch.pt').to(GPU)
assert C_input_wav_batch.shape == torch.Size([10, 35840])
C_padding_mask = torch.BoolTensor(C_input_wav_batch.shape).to(GPU).fill_(False)
assert C_padding_mask.shape == torch.Size([10, 35840])

def hubert_inference():
    t0 = time.perf_counter()
    # B = random.randint(1, 10)
    B = 10
    input_wav_batch = C_input_wav_batch[:B]
    padding_mask = C_padding_mask[:B]

    t1 = time.perf_counter()
    with torch.no_grad():
        feats_batch, _ = hubert_model.extract_features(
            source=input_wav_batch,
            padding_mask=padding_mask,
            output_layer=12,
        )
    t2 = time.perf_counter()
    assert feats_batch.size(0) == B
    print(f'hubert inference [B={input_wav_batch.size(0)}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms')

if __name__ == '__main__':
    load_hubert_model()
    while True:
        hubert_inference()
