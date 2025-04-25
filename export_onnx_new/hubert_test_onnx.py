#!/usr/bin/env python3

import time

import torch
import onnxruntime

DATA_DIR = '../load_tests/data'

GPU = 'cuda:0'

C_input_wav_batch = torch.load(f'{DATA_DIR}/input_wav_batch.pt').to(GPU)
assert C_input_wav_batch.shape == torch.Size([10, 35840])
C_padding_mask = torch.BoolTensor(C_input_wav_batch.shape).to(GPU).fill_(False)
assert C_padding_mask.shape == torch.Size([10, 35840])

if __name__ == '__main__':
    onnxruntime.preload_dlls()
    session = onnxruntime.InferenceSession(f'hubert_extract_features.onnx', providers=['CUDAExecutionProvider'])
    for _ in range(25):
        print(f'Running hubert inference')
        start_ts = time.perf_counter()
        outputs = session.run(
            None,
            {
                'input_wav': C_input_wav_batch.numpy(force=True),
                'padding_mask': C_padding_mask.numpy(force=True),
            }
        )
        end_ts = time.perf_counter()
        print(
            f'hubert inference completed in {(end_ts - start_ts) * 1000:.1f} ms; '
            f'outputs[0].shape: {outputs[0].shape}'
        )
