#!/usr/bin/env python3

import time

import torch
import onnxruntime

DATA_DIR = '../load_tests/data'

GPU = 'cuda:0'
IS_HALF = True

P_LEN = 224
SKIP_HEAD = 200
RETURN_LENGTH = 20

C_feats = torch.load(f'{DATA_DIR}/feats.pt').to(GPU)
C_p_len = torch.full((10,), P_LEN, dtype=torch.long, device=GPU)
C_cache_pitch = torch.load(f'{DATA_DIR}/cache_pitch.pt').to(GPU)
C_cache_pitchf = torch.load(f'{DATA_DIR}/cache_pitchf.pt').to(GPU)
C_sid = torch.zeros(10, dtype=torch.long, device=GPU)

C_skip_head = torch.LongTensor([SKIP_HEAD])
C_return_length = torch.LongTensor([RETURN_LENGTH])

fp32_opt_suffix = '' if IS_HALF else '_fp32'
if not IS_HALF:
    C_feats = C_feats.float()
    C_cache_pitchf = C_cache_pitchf.float()

VOICES = {
    'voicevox_speaker_43',
    'xiangling_eng',
    'citlali_jap',
}

if __name__ == '__main__':
    onnxruntime.preload_dlls()
    for voice in VOICES:
        session = onnxruntime.InferenceSession(f'{voice}{fp32_opt_suffix}.onnx', providers=['CUDAExecutionProvider'])
        for _ in range(5):
            print(f'Running inference for {voice}')
            start_ts = time.perf_counter()
            outputs = session.run(
                None,
                {
                    'feats': C_feats.numpy(force=True),
                    'p_len': C_p_len.numpy(force=True),
                    'pitch': C_cache_pitch.numpy(force=True),
                    'pitchf': C_cache_pitchf.numpy(force=True),
                    'sid': C_sid.numpy(force=True),
                }
            )
            end_ts = time.perf_counter()
            print(
                f'Inference for {voice} completed in {(end_ts - start_ts) * 1000:.1f} ms; '
                f'outputs[0].shape: {outputs[0].shape}'
            )
