#!/usr/bin/env python3

import time
from typing import Optional

import torch
import torch.nn as nn
from torchfcpe import spawn_bundled_infer_model
from torchfcpe.models_infer import InferCFNaiveMelPE

DATA_DIR = '../load_tests/data'

GPU = 'cuda:0'
IS_HALF = True

fcpe_model: Optional[InferCFNaiveMelPE] = None

def load_fcpe_model():
    global fcpe_model
    load_start_time = time.perf_counter()
    fcpe_model = spawn_bundled_infer_model(device=GPU)
    load_done_time = time.perf_counter()
    print(f'Loaded fcpe model in {(load_done_time - load_start_time) * 1000:.1f} ms')

C_input_wav_batch = torch.load(f'{DATA_DIR}/input_wav_batch_cropped.pt').to(GPU)
assert C_input_wav_batch.shape == torch.Size([10, 3200])

class FcpeWrapper(nn.Module):
    def __init__(self, orig_fcpe_model: InferCFNaiveMelPE):
        super().__init__()
        self.model = orig_fcpe_model.eval()

    def forward(self, input_wav_batch) -> torch.Tensor:
        return self.model.infer(
            input_wav_batch,
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        )

def export_to_onnx():
    model = FcpeWrapper(orig_fcpe_model=fcpe_model)
    model.eval()

    torch.onnx.export(
        model,
        (C_input_wav_batch,),
        'fcpe.onnx',
        input_names=['input_wav'],
        output_names=['f0'],
        dynamic_axes={
            'input_wav': {0: 'batch_size', 1: 'audio_len'},
            'f0': {0: 'batch_size', 1: 'f0_len'},
        },
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        dynamo=True,
        external_data=False,
    )

if __name__ == '__main__':
    load_fcpe_model()
    export_to_onnx()
