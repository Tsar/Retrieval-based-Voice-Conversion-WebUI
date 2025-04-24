#!/usr/bin/env python3

# Inspired by: https://github.com/finomen/FCPE/blob/0441b6f11719e558c6f0ee65ab98d0467e75d566/tools/fcpe_export_onnx.py

import torch
import torch.nn as nn
from torchfcpe import spawn_bundled_infer_model
from torchfcpe.models_infer import InferCFNaiveMelPE

GPU = 'cuda:0'

C_input_wav_batch = torch.load('../load_tests/data/input_wav_batch_cropped.pt')
assert C_input_wav_batch.shape == torch.Size([10, 3200])

class InferCFNaiveMelPEWrapper(nn.Module):
    def __init__(self, orig_model: InferCFNaiveMelPE):
        super().__init__()
        self.model = orig_model.float().eval()

    def forward(self, mel: torch.Tensor):
        with torch.no_grad():
            return self.model.model.infer(mel, decoder='local_argmax', threshold=0.006)

if __name__ == '__main__':
    orig_fcpe_model: InferCFNaiveMelPE = spawn_bundled_infer_model(device=GPU)
    print(f'Args: {orig_fcpe_model.args_dict}')

    C_mel = orig_fcpe_model.wav2mel(audio=C_input_wav_batch, sample_rate=16000)

    model = InferCFNaiveMelPEWrapper(orig_model=orig_fcpe_model)
    model.eval()

    torch.onnx.export(
        model,
        (C_mel,),
        'fcpe.onnx',
        input_names=['mel'],
        output_names=['pitchf'],
        dynamic_axes={
            'mel': {0: 'batch_size', 1: 'mel_len'},
            'pitchf': {0: 'batch_size', 1: 'pitchf_len'},
        },
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        use_external_data_format=False,
    )
