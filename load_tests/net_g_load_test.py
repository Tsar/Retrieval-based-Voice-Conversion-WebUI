#!/usr/bin/env python3

import time
from typing import Optional

import torch
import torch.nn as nn

from infer.lib.jit.get_synthesizer import get_synthesizer

GPU = 'cuda:0'
IS_HALF = True

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

C_feats = torch.load('data/feats.pt').to(GPU)
C_p_len = torch.full((10,), P_LEN, dtype=torch.long, device=GPU)
C_cache_pitch = torch.load('data/cache_pitch.pt').to(GPU)
C_cache_pitchf = torch.load('data/cache_pitchf.pt').to(GPU)
C_sid = torch.zeros(10, dtype=torch.long, device=GPU)

skip_head = torch.LongTensor([SKIP_HEAD])
return_length = torch.LongTensor([RETURN_LENGTH])
return_length2 = torch.LongTensor([RETURN_LENGTH2])

def net_g_inference():
    t0 = time.perf_counter()
    # B = random.randint(1, 10)
    B = 10
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

if __name__ == '__main__':
    load_net_g_model('../assets/weights/voicevox_speaker_43.pth')
    while True:
        net_g_inference()
