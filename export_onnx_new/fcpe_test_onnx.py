#!/usr/bin/env python3

import time

import torch
from torchfcpe.models_infer import spawn_wav2mel
from torchfcpe.tools import DotDict
import onnxruntime

DATA_DIR = '../load_tests/data'

GPU = 'cuda:0'
IS_HALF = True

P_LEN = 224
SKIP_HEAD = 200
RETURN_LENGTH = 20

C_input_wav_batch = torch.load('../load_tests/data/input_wav_batch_cropped.pt').cpu()
assert C_input_wav_batch.shape == torch.Size([10, 3200])

mel_extractor_args = DotDict()
mel_extractor_args.mel = DotDict({'fmax': 8000, 'fmin': 0, 'hop_size': 160, 'n_fft': 1024, 'num_mels': 128, 'sr': 16000, 'win_size': 1024})
mel_extractor = spawn_wav2mel(mel_extractor_args, 'cpu')

C_mel = mel_extractor(C_input_wav_batch, sample_rate=16000)

if __name__ == '__main__':
    onnxruntime.preload_dlls()
    session = onnxruntime.InferenceSession(f'fcpe.onnx', providers=['CUDAExecutionProvider'])
    #for input_arg in session.get_inputs():
    #    print(f'Input: {input_arg}')
    for _ in range(25):
        print(f'Running fcpe inference')
        start_ts = time.perf_counter()
        outputs = session.run(
            None,
            {
                'mel': C_mel.numpy(force=True),
            }
        )
        end_ts = time.perf_counter()
        print(
            f'fcpe inference completed in {(end_ts - start_ts) * 1000:.1f} ms; '
            f'outputs[0].shape: {outputs[0].shape}'
        )
