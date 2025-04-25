import time
import json
import uuid
import aiohttp
import numpy as np
import torch

def to_api_datatype(dtype: torch.dtype):
    if dtype == torch.float32:
        return 'FP32'
    if dtype == torch.int64:
        return 'INT64'
    raise RuntimeError(f'Unsupported torch data type for API: {dtype}')

def api_datatype_to_numpy(datatype: str):
    if datatype == 'FP32':
        return np.float32
    if datatype == 'INT64':
        return np.int64
    raise RuntimeError(f'Unsupported API datatype: {datatype}')

async def run_inference_request(
    client_session: aiohttp.ClientSession,
    infer_url_prefix: str,
    model_name: str,
    input_tensors: list[tuple[str, torch.Tensor]],
    output_tensor_names: list[str],
) -> list[torch.Tensor]:
    t0 = time.perf_counter()
    inference_header = {
        'id': str(uuid.uuid4()),
        'inputs': [
            {
                'name': name,
                'shape': list(tensor.shape),
                'datatype': to_api_datatype(tensor.dtype),
                'parameters': {'binary_data_size': tensor.numel() * tensor.element_size()},
            } for name, tensor in input_tensors
        ],
        'outputs': [
            {
                'name': name,
                'parameters': {'binary_data': True},
            } for name in output_tensor_names
        ]
    }
    inference_header_data = json.dumps(inference_header).encode('UTF-8')
    request_body = inference_header_data + b''.join([tensor.numpy().tobytes() for _, tensor in input_tensors])
    print(f'Request body size: {len(request_body)}, inference header size: {len(inference_header_data)}')

    t1 = time.perf_counter()
    async with client_session.post(
        url=f'{infer_url_prefix}/v2/models/{model_name}/infer',
        data=request_body,
        headers={'Inference-Header-Content-Length': str(len(inference_header_data))},
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f'Request failed with code {resp.status}: {await resp.text()}')
        resp_inference_header_length = resp.headers.get('Inference-Header-Content-Length')
        result = await resp.read()
    t2 = time.perf_counter()

    assert resp_inference_header_length is not None
    resp_inference_header_length = int(resp_inference_header_length)
    resp_inference_header = json.loads(result[:resp_inference_header_length])
    assert resp_inference_header['model_name'] == model_name
    outputs = resp_inference_header['outputs']
    assert len(outputs) == len(output_tensor_names)
    pos = resp_inference_header_length
    output_tensors = []
    for output in outputs:
        output_data_size = output['parameters']['binary_data_size']
        array = np.frombuffer(result[pos:pos + output_data_size], dtype=api_datatype_to_numpy(output['datatype']))
        tensor = torch.from_numpy(array).view(output['shape'])
        output_tensors.append(tensor)
        pos += output_data_size

    t3 = time.perf_counter()
    print(
        f'kserve inference request [model: {model_name}, B={input_tensors[0][1].size(0)}]: '
        f'{(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms, post: {(t3 - t2) * 1000:.1f} ms'
    )
    return output_tensors
