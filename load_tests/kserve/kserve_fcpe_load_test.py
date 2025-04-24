#!/usr/bin/env python3

import os
import time
import random
import uuid
import json

import asyncio
import aiohttp

import torch
from torchfcpe.models_infer import spawn_wav2mel
from torchfcpe.tools import DotDict

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

MODEL = 'fcpe'
URL = f'https://rvc-inference-predictor.2dchan-inference.knative.finomen.net/v2/models/{MODEL}/infer'

RAND_BATCH_SIZE = bool(int(os.environ.get('RAND_BATCH_SIZE', 0)))

C_input_wav_batch = torch.load('../data/input_wav_batch_cropped.pt').cpu()
assert C_input_wav_batch.shape == torch.Size([10, 3200])

mel_extractor_args = DotDict()
mel_extractor_args.mel = DotDict({'fmax': 8000, 'fmin': 0, 'hop_size': 160, 'n_fft': 1024, 'num_mels': 128, 'sr': 16000, 'win_size': 1024})
mel_extractor = spawn_wav2mel(mel_extractor_args, 'cpu')

async def fcpe_inference(session):
    t0 = time.perf_counter()
    if RAND_BATCH_SIZE:
        B = random.randint(1, 10)
    else:
        B = 10
    input_wav_batch: torch.Tensor = C_input_wav_batch[:B]
    mel = mel_extractor(input_wav_batch, sample_rate=16000)

    inference_header = {
        'id': str(uuid.uuid4()),
        'inputs': [
            {
                'name': 'mel',
                'shape': list(mel.shape),
                'datatype': 'FP32',
                'parameters': {'binary_data_size': mel.numel() * mel.element_size()},
            }
        ],
        'outputs': [
            {
                'name': 'pitchf',
                'parameters': {'binary_data': True},
            }
        ]
    }
    inference_header_data = json.dumps(inference_header).encode('UTF-8')
    request_body = (
        inference_header_data +
        mel.numpy().tobytes()
    )
    print(f'Request body size: {len(request_body)}, inference header size: {len(inference_header_data)}')

    t1 = time.perf_counter()
    async with session.post(URL, data=request_body, headers={'Inference-Header-Content-Length': str(len(inference_header_data))}) as resp:
        if resp.status != 200:
            print(f'Request failed with code {resp.status}: {await resp.text()}')
            return
        resp_inference_header_length = resp.headers.get('Inference-Header-Content-Length')
        result = await resp.read()
    t2 = time.perf_counter()

    assert resp_inference_header_length is not None
    resp_inference_header_length = int(resp_inference_header_length)
    resp_inference_header = json.loads(result[:resp_inference_header_length])
    assert resp_inference_header['model_name'] == MODEL
    assert len(resp_inference_header['outputs']) == 1
    output = resp_inference_header['outputs'][0]
    assert output['name'] == 'pitchf'
    assert output['datatype'] == 'FP32'
    assert output['shape'] == [B, 21, 1]
    assert 'data' not in output
    assert len(result) == resp_inference_header_length + B * 21 * 4

    t3 = time.perf_counter()
    print(f'fcpe inference [B={B}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms, post: {(t3 - t2) * 1000:.1f} ms')

async def main():
    async with aiohttp.ClientSession(headers={'Authorization': BEARER_PREFIX + AUTH_TOKEN}) as session:
        while True:
            await fcpe_inference(session)

if __name__ == '__main__':
    asyncio.run(main())
