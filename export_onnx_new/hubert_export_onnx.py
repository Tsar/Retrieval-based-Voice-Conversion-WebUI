#!/usr/bin/env python3

import time
from typing import Optional

import fairseq.checkpoint_utils
from fairseq.models.hubert import HubertModel
import torch
import torch.nn as nn

DATA_DIR = '../load_tests/data'

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

C_input_wav_batch = torch.load(f'{DATA_DIR}/input_wav_batch.pt').to(GPU)
assert C_input_wav_batch.shape == torch.Size([10, 35840])
C_padding_mask = torch.BoolTensor(C_input_wav_batch.shape).to(GPU).fill_(False)
assert C_padding_mask.shape == torch.Size([10, 35840])

class HubertExtractFeaturesWrapper(nn.Module):
    def __init__(self, orig_hubert_model: HubertModel):
        super().__init__()
        self.model = orig_hubert_model

    def forward(self, source, padding_mask) -> torch.Tensor:  # only pass Tensors here
        return self.model.extract_features(
            source=source,
            padding_mask=padding_mask,
            output_layer=12,
        )[0]

# Leaving some useful links here:
#  https://github.com/facebookresearch/fairseq/issues/5595 - used "m = float(m)" hack from here
#  https://github.com/facebookresearch/fairseq/issues/5596 - this hack not yet used

def export_to_onnx(use_scripting=False, use_dynamic_axes=True):
    model = HubertExtractFeaturesWrapper(orig_hubert_model=hubert_model)
    model.eval()

    to_export = model
    if use_scripting:
        scripted_model = torch.jit.script(model)
        to_export = scripted_model

    torch.onnx.export(
        to_export,
        (
            C_input_wav_batch,  # source
            C_padding_mask,     # padding_mask
        ),
        'hubert_extract_features.onnx',
        input_names=['input_wav', 'padding_mask'],
        output_names=['features'],
        dynamic_axes={
            'input_wav': {0: 'batch_size', 1: 'audio_len'},
            'padding_mask': {0: 'batch_size', 1: 'audio_len'},
            'features': {0: 'batch_size', 1: 'sequence_len'},
        } if use_dynamic_axes else None,
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        use_external_data_format=False,
    )

if __name__ == '__main__':
    load_hubert_model()
    export_to_onnx()
