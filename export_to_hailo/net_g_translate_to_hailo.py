#!/usr/bin/env python3

from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import Dims

# First I tried the following command:
#  hailo parser onnx --hw-arch hailo8 citlali_jap__no_dynamic_shapes.onnx
# But it failed.
# This script is to provide additional arguments.
# It also doesn't work yet...

if __name__ == '__main__':
    onnx_model_name = 'citlali_jap'
    onnx_path = '../export_onnx_new/citlali_jap__no_dynamic_shapes.onnx'

    runner = ClientRunner(hw_arch='hailo8')
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        onnx_model_name,
        start_node_names=[
            'feats',
            'p_len',
            'pitch',
            'pitchf',
            'sid',
        ],
        end_node_names=['audio'],
        net_input_format={
            'feats': [Dims.BATCH, Dims.WIDTH, Dims.CHANNELS],
            'p_len': [Dims.BATCH],
            'pitch': [Dims.BATCH, Dims.WIDTH],
            'pitchf': [Dims.BATCH, Dims.WIDTH],
            'sid': [Dims.BATCH],
        }
    )
