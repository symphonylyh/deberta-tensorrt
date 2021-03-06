#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Test ORT-TRT engine of DeBERTa model. Different precisions are supported.

Usage: 
Test model inference time:
    - python deberta_ort_inference.py --onnx=./test/deberta.onnx --test fp16

Correctness check by comparing original model and model with plugin:
    - python deberta_ort_inference.py --onnx=./test/deberta --correctness_check fp16

Notes: 
    - supported precision is fp16 (more TBD, but Microsoft current focus is FP16). For test, you can specify more than one precisions, and TensorRT engine of each precision will be built sequentially.
    - engine files are saved at `/engine_cache/[Model name]_[GPU name]_[Precision]/`. Note that TensorRT engine is specific to both GPU architecture and TensorRT version. 
    - if in --correctness_check mode, the argument for --onnx is the stem name for the model without .onnx extension.
"""

import os, argparse 
import onnxruntime as ort 
import numpy as np
import torch
from time import time

ENGINE_PATH = './engine_cache'
if not os.path.exists(ENGINE_PATH):
    os.makedirs(ENGINE_PATH)

def GPU_ABBREV(name):
    '''
    Map GPU device query name to abbreviation.
    
    ::param str name Device name from torch.cuda.get_device_name().
    ::return str GPU abbreviation.
    ''' 

    GPU_LIST = [
        'TITAN',
        'V100',
        'T4',
        'A100',
        'A10G',
        'A10'
    ] # partial list, can be extended

    for i in GPU_LIST:
        if i in name:
            return i 
    # The order of A100, A10G, A10 matters. They're put in a way to not detect substring A10 as A100

gpu_name = GPU_ABBREV(torch.cuda.get_device_name())

VALID_PRECISION = [
    'fp32',
    'fp16',
    # 'int8' # int8 needs changes from the model side, not valid at this moment
]

parser = argparse.ArgumentParser(description="Build and test TensorRT engine.")
parser.add_argument('--onnx', required=True, help='ONNX model path (or filename stem if in correctness check mode).')
parser.add_argument('--test', nargs='+', help='Test ORT-TRT engine in precision fp32/fp16. You can list multiple precisions to test all of them.') # nargs='+': varible number of args, but require at least one
parser.add_argument('--correctness_check', nargs='+', help='Correctness check for original & plugin TRT engines in precision fp32/fp16. You can list multiple precisions to check all of them.')

args = parser.parse_args()

ONNX_MODEL = args.onnx    
MODEL_STEM = os.path.splitext(args.onnx)[0].split('/')[-1]
TEST = args.test
CORRECTNESS = args.correctness_check

def test_engine():

    for precision in TEST:
        
        engine_cachepath = '/'.join([ENGINE_PATH, '_'.join([MODEL_STEM, gpu_name, precision])])     

        providers = [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': precision == 'fp16',
                'trt_int8_enable': precision == 'int8',
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': engine_cachepath
            }),
            'CUDAExecutionProvider'] # EP order indicates priority
        # EP options: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#execution-provider-options

        so = ort.SessionOptions()
        # so.log_severity_level = 0 # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2

        sess = ort.InferenceSession(ONNX_MODEL, sess_options=so, providers=providers) 

        print(f'Running inference on engine {engine_cachepath}')

        ## psuedo-random input test
        batch_size = 1
        seq_len = 2048
        vocab = 128203
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long)
        inputs = {
            'input_ids': input_ids.numpy(), 
            'attention_mask': attention_mask.numpy()
        }

        outputs = sess.run(None, inputs)

        nreps = 100
        start_time = time()
        for _ in range(nreps):
            sess.run(None, inputs)
        end_time = time()

        duration = end_time - start_time
        print(f'Average Inference time (ms) of {nreps} runs: {duration/nreps*1000:.3f}. For more accurate test, please use the onnxruntime_perf_test commands.')

def correctness_check_engines():
    
    for precision in CORRECTNESS:

        engine_cachepath1 = '/'.join([ENGINE_PATH, '_'.join([MODEL_STEM, gpu_name, precision])])
        engine_cachepath2 = '/'.join([ENGINE_PATH, '_'.join([MODEL_STEM+'_plugin', gpu_name, precision])])

        if not os.path.exists(engine_cachepath1) or not os.path.exists(engine_cachepath2):
            print('At least one of the original and/or plugin engines do not exist. Please build them first by --test')
            return 

        print(f'Running inference on original engine {engine_cachepath1} and plugin engine {engine_cachepath2}')

        so = ort.SessionOptions()

        providers1 = [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': precision == 'fp16',
                'trt_int8_enable': precision == 'int8',
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': engine_cachepath1
            }),
            'CUDAExecutionProvider'] 

        providers2 = [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': precision == 'fp16',
                'trt_int8_enable': precision == 'int8',
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': engine_cachepath2
            }),
            'CUDAExecutionProvider'] 

        sess1 = ort.InferenceSession(ONNX_MODEL+'.onnx', sess_options=so, providers=providers1)
        sess2 = ort.InferenceSession(ONNX_MODEL+'_plugin.onnx', sess_options=so, providers=providers2)

        ## psuedo-random input test
        batch_size = 1
        seq_len = 2048
        vocab = 128203
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long)
        inputs = {
            'input_ids': input_ids.numpy(), 
            'attention_mask': attention_mask.numpy()
        }

        outputs1 = sess1.run(None, inputs)
        outputs2 = sess2.run(None, inputs)

        for i in range(len(outputs1)):
            avg_abs_error = np.sum(np.abs(outputs1[i] - outputs2[i])) / outputs1[i].size
            max_abs_error = np.max(np.abs(outputs1[i] - outputs2[i]))
            print(f"Output {i}:")
            print("onnx model (original): ", outputs1[i])
            print("onnx model (plugin): ", outputs2[i])
            print(f"[Output {i} Element-wise Check] Avgerage absolute error: {avg_abs_error:e}, Maximum absolute error: {max_abs_error:e}. Below the order of 1e-3 and 1e-4 is the expected precision for FP16 (10 significance bits)" ) # machine epsilon for different precisions: https://en.wikipedia.org/wiki/Machine_epsilon

if TEST:
    test_engine()

if CORRECTNESS:
    correctness_check_engines()




