#!/usr/bin/env python3

import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
from infer.lib.jit.get_synthesizer import get_synthesizer

DATA_DIR = '../load_tests/data'

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

C_feats = torch.load(f'{DATA_DIR}/feats.pt').to(GPU)
C_p_len = torch.full((10,), P_LEN, dtype=torch.long, device=GPU)
C_cache_pitch = torch.load(f'{DATA_DIR}/cache_pitch.pt').to(GPU)
C_cache_pitchf = torch.load(f'{DATA_DIR}/cache_pitchf.pt').to(GPU)
C_sid = torch.zeros(10, dtype=torch.long, device=GPU)

C_skip_head = torch.LongTensor([SKIP_HEAD])
C_return_length = torch.LongTensor([RETURN_LENGTH])
C_return_length2 = torch.LongTensor([RETURN_LENGTH2])

class NetGWrapper(nn.Module):
    def __init__(self, orig_net_g_model: nn.Module):
        super().__init__()
        self.model = orig_net_g_model.eval()

    def forward(
        self,
        feats,
        p_len,
        pitch,
        pitchf,
        sid,
        skip_head,
        return_length,
        return_length2,
    ) -> torch.Tensor:
        return self.model.infer(
            feats,
            p_len,
            pitch,
            pitchf,
            sid,
            skip_head,
            return_length,
            return_length2,
        )[0]

def export_to_onnx(onnx_filename):
    model = NetGWrapper(orig_net_g_model=net_g_model)
    model.eval()

    torch.onnx.export(
        model,
        (
            C_feats,
            C_p_len,
            C_cache_pitch,
            C_cache_pitchf,
            C_sid,
            C_skip_head,
            C_return_length,
            C_return_length2,
        ),
        onnx_filename,
        input_names=[
            'feats',
            'p_len',
            'pitch',
            'pitchf',
            'sid',
            'skip_head',
            'return_length',
            'return_length2',
        ],
        output_names=['audio'],
        dynamic_axes={
            'feats': {0: 'batch_size', 1: 'p_len'},
            'p_len': {0: 'batch_size'},
            'pitch': {0: 'batch_size', 1: 'p_len'},
            'pitchf': {0: 'batch_size', 1: 'p_len'},
            'sid': {0: 'batch_size'},
            'audio': {0: 'batch_size', 1: 'audio_len'},
        },
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        dynamo=True,
        external_data=False,
    )

if __name__ == '__main__':
    load_net_g_model('../assets/weights/voicevox_speaker_43.pth')
    export_to_onnx('voicevox_speaker_43.onnx')
