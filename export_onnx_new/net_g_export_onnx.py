#!/usr/bin/env python3

import os
import sys
import time
from typing import Optional
import numpy as np

import torch
import torch.nn as nn

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
from infer.lib.jit.get_synthesizer import get_synthesizer

DATA_DIR = '../load_tests/data'

GPU = 'cuda:0'
IS_HALF = False

P_LEN = 224
SKIP_HEAD = 200
RETURN_LENGTH = 20

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

C_feats = torch.load(f'{DATA_DIR}/feats.pt').float().to(GPU)
C_p_len = torch.full((10,), P_LEN, dtype=torch.long, device=GPU)
C_cache_pitch = torch.load(f'{DATA_DIR}/cache_pitch.pt').to(GPU)
C_cache_pitchf = torch.load(f'{DATA_DIR}/cache_pitchf.pt').float().to(GPU)
C_sid = torch.zeros(10, dtype=torch.long, device=GPU)

C_skip_head = torch.LongTensor([SKIP_HEAD])
C_return_length = torch.LongTensor([RETURN_LENGTH])

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

def export_to_onnx(onnx_filename, return_length2: torch.LongTensor, use_scripting=False, use_dynamic_axes=True):
    model = NetGWrapper(orig_net_g_model=net_g_model)
    model.eval()

    to_export = model
    if use_scripting:
        scripted_model = torch.jit.script(model)
        to_export = scripted_model

    torch.onnx.export(
        to_export,
        (
            C_feats,
            C_p_len,
            C_cache_pitch,
            C_cache_pitchf,
            C_sid,
            C_skip_head,
            C_return_length,
            return_length2,
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
        } if use_dynamic_axes else None,
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        use_external_data_format=False,
    )

class Voice:
    def __init__(self, model_pth_path: str, pitch: int, formant_shift: float = 0.0):
        self.pitch = pitch
        self.model_pth_path = model_pth_path
        self.formant_shift = formant_shift

VOICES: dict[str, Voice] = {
    'voicevox_speaker_43': Voice(
        model_pth_path='assets/weights/voicevox_speaker_43.pth',
        pitch=8,
    ),
    'xiangling_eng': Voice(
        model_pth_path='assets/weights/xiangling_eng_30_epochs_with_pitch.pth',
        pitch=12,
        formant_shift=1.0,
    ),
    'citlali_jap': Voice(
        model_pth_path='assets/weights/citlali_jap.pth',
        pitch=6,
    ),
}

if __name__ == '__main__':
    for voice in VOICES:
        voice_props = VOICES[voice]
        factor = pow(2, voice_props.formant_shift / 12)
        ret_length2 = int(np.ceil(RETURN_LENGTH * factor))
        ret_length2_tensor = torch.LongTensor([ret_length2])

        load_net_g_model(f'../{voice_props.model_pth_path}')
        export_to_onnx(
            onnx_filename=f'{voice}_fp32.onnx',
            return_length2=ret_length2_tensor,
        )
        export_to_onnx(
            onnx_filename=f'{voice}__no_dynamic_shapes_fp32.onnx',
            return_length2=ret_length2_tensor,
            use_dynamic_axes=False,
        )
