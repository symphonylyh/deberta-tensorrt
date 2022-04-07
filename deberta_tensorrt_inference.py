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
Build and test TensorRT engine of DeBERTa model. Different precisions are supported.

Usage: 
Build and test a model:
    - build: python deberta_tensorrt_inference.py --onnx=deberta.onnx --build fp16 --plugin=./plugin/build/bin/libcustomplugins.so
    - test: python deberta_tensorrt_inference.py --onnx=deberta.onnx --test fp16 --plugin=./plugin/build/bin/libcustomplugins.so
    test will measure the inference time

Correctness check by comparing original model and model with plugin:
    - [1] export ONNX model with extra output nodes: python deberta_onnx_modify.py deberta.onnx --correctness_check
    - [2] build original model: python deberta_tensorrt_inference.py --onnx=deberta_correctness_check_original.onnx --build fp16 --plugin=./plugin/build/bin/libcustomplugins.so
    - [3] build plugin model: python deberta_tensorrt_inference.py --onnx=deberta_correctness_check_plugin.onnx --build fp16 --plugin=./plugin/build/bin/libcustomplugins.so
    - [4] correctness check: python deberta_tensorrt_inference.py --onnx=deberta --correctness_check fp16 --plugin=./plugin/build/bin/libcustomplugins.so

Notes: 
    - supported precision is fp16 (more TBD, but Microsoft current focus is FP16). For both build and test, you can specify more than one precisions, and TensorRT engine of each precision will be built sequentially.
    - engine files are saved as `/path/[Model name]_[GPU name]_[Precision].engine`. Note that TensorRT engine is specific to both GPU architecture and TensorRT version. 
    - if in --correctness_check mode, the argument for --onnx is the root name for the models [root]_correctness_check_original/plugin.onnx
"""

import torch
import tensorrt as trt
import os, argparse 
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # without this, "LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
from time import time

TRT_VERSION = trt.__version__[:3] # x.x

def GPU_ABBREV(name):
    '''
    Map GPU device query name to abbreviation.
    
    ::param str name Device name from torch.cuda.get_device_name().
    ::return str GPU abbreviation.
    ''' 

    GPU_LIST = [
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
    # 'fp32',
    # 'tf32',
    'fp16'
]


parser = argparse.ArgumentParser(description="Build and test TensorRT engine.")
parser.add_argument('--onnx', required=True, help='ONNX model path (or filename stem if in correctness check mode).')
parser.add_argument('--build', nargs='+', help='Build TRT engine in precision fp32/tf32/fp16. You can list multiple precisions to build all of them.') # nargs='+': varible number of args, but require at least one
parser.add_argument('--test', nargs='+', help='Test TRT engine in precision fp32/tf32/fp16. You can list multiple precisions to test all of them.')
parser.add_argument('--correctness_check', nargs='+', help='Correctness check for original & plugin TRT engines in precision fp32/tf32/fp16. You can list multiple precisions to check all of them.')
parser.add_argument('--plugin', type=str, help='Path to plugin library.')

args = parser.parse_args()

ONNX_MODEL = args.onnx    
MODEL_NAME = os.path.splitext(args.onnx)[0]
BUILD = args.build
TEST = args.test
CORRECTNESS = args.correctness_check
PLUGIN_PATH = args.plugin

# load plugin
import ctypes 
ctypes.CDLL(PLUGIN_PATH)

if not (args.build or args.test or args.correctness_check):
    parser.error('Please specify --build and/or --test and/or --correctness_check' )

if BUILD:
    for i in BUILD:
        if i not in VALID_PRECISION:
            parser.error('Unsupported precision')
if TEST:
    for i in TEST:
        if i not in VALID_PRECISION:
            parser.error('Unsupported precision')
if CORRECTNESS:
    for i in CORRECTNESS:
        if i not in VALID_PRECISION:
            parser.error('Unsupported precision')

class TRTModel:
    '''
    Generic class to run a TRT engine by specifying engine path and giving input data.
    '''
    class HostDeviceMem(object):
        '''
        Helper class to record host-device memory pointer pairs
        '''
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def __init__(self, engine_path):
        self.engine_path = engine_path 
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # load and deserialize TRT engine
        self.engine = self.load_engine()

        # allocate input/output memory buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

        # create context
        self.context = self.engine.create_execution_context()

        # Dict of NumPy dtype -> torch dtype (when the correspondence exists). From: https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
        self.numpy_to_torch_dtype_dict = {
            bool       : torch.bool,
            np.uint8      : torch.uint8,
            np.int8       : torch.int8,
            np.int16      : torch.int16,
            np.int32      : torch.int32,
            np.int64      : torch.int64,
            np.float16    : torch.float16,
            np.float32    : torch.float32,
            np.float64    : torch.float64,
            np.complex64  : torch.complex64,
            np.complex128 : torch.complex128
        }

    def load_engine(self):
        with open(self.engine_path, 'rb') as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine
    
    def allocate_buffers(self, engine):
        '''
        Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        '''
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine: # binding is the name of input/output
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swapped to disk)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings. When cast to int, it's a linear index into the context's memory (like memory address). See https://documen.tician.de/pycuda/driver.html#pycuda.driver.DeviceAllocation
            bindings.append(int(device_mem))

            # Append to the appropriate input/output list.
            if engine.binding_is_input(binding):
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, model_inputs: list, timing=False):
        '''
        Inference step (like forward() in PyTorch).

        model_inputs: list of numpy array or list of torch.Tensor (on GPU)
        '''
        NUMPY = False
        TORCH = False
        if isinstance(model_inputs[0], np.ndarray):
            NUMPY = True
        elif torch.is_tensor(model_inputs[0]):
            TORCH = True
        else:
            assert False, 'Unsupported input data format!'
        
        # batch size consistency check
        if NUMPY:
            batch_size = np.unique(np.array([i.shape[0] for i in model_inputs]))
        elif TORCH:
            batch_size = np.unique(np.array([i.size(dim=0) for i in model_inputs]))
        assert len(batch_size) == 1, 'Input batch sizes are not consistent!'
        batch_size = batch_size[0]

        for i, model_input in enumerate(model_inputs):
            binding_name = self.engine[i] # i-th input/output name
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding_name)) # trt can only tell to numpy dtype

            # input type cast
            if NUMPY:
                model_input = model_input.astype(binding_dtype)
            elif TORCH:
                model_input = model_input.to(self.numpy_to_torch_dtype_dict[binding_dtype])

            if NUMPY:
                # fill host memory with flattened input data
                np.copyto(self.inputs[i].host, model_input.ravel())
            elif TORCH:
                if timing:
                    cuda.memcpy_dtod(self.inputs[i].device, model_input.data_ptr(), model_input.element_size() * model_input.nelement()) 
                else:
                    # for Torch GPU tensor it's easier, can just do Device to Device copy
                    cuda.memcpy_dtod_async(self.inputs[i].device, model_input.data_ptr(), model_input.element_size() * model_input.nelement(), self.stream) # dtod need size in bytes

        if NUMPY:   
            if timing:
                [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
            else:
                # input, Host to Device
                [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        
        duration = 0
        if timing:
            start_time = time()
            self.context.execute_v2(bindings=self.bindings)
            end_time = time()
            duration = end_time - start_time
        else:
            # run inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle) # v2 no need for batch_size arg

        if timing:
            [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]
        else:
            # output, Device to Host
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        if not timing:
            # synchronize to ensure completion of async calls
            self.stream.synchronize()

        
        if NUMPY: 
            return [out.host.reshape(batch_size,-1) for out in self.outputs], duration
        elif TORCH:
            return [torch.from_numpy(out.host.reshape(batch_size,-1)) for out in self.outputs], duration

def build_engine():

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    TRT_BUILDER = trt.Builder(TRT_LOGGER)

    for precision in BUILD:
        print(f'Building {precision} engine of {MODEL_NAME} model on {gpu_name} GPU...')

        ## parse ONNX model
        network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
        parse_success = onnx_parser.parse_from_file(ONNX_MODEL)
        for idx in range(onnx_parser.num_errors):
            print(onnx_parser.get_error(idx))
        if not parse_success:
            print('ONNX model parsing failed')
        
        ## build TRT engine (configuration options at: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html#ibuilderconfig)
        config = TRT_BUILDER.create_builder_config()
        
        # handle dynamic shape (min/opt/max): https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes
        profile = TRT_BUILDER.create_optimization_profile() 
        profile.set_shape("input_ids", (1,2048), (1,2048), (1,2048)) 
        profile.set_shape("attention_mask", (1,2048), (1,2048), (1,2048)) 
        config.add_optimization_profile(profile)
        
        # workspace
        if TRT_VERSION == '8.4':
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * (1 << 20)) # 4096 MiB, not in 8.2.3. 8.4.0 only. By checking print(dir(config)), I found it's max_workspace_size for 8.2.3
        elif TRT_VERSION == '8.2':
            config.max_workspace_size = 4096 * (1 << 20)

        # precision
        if precision == 'fp32':
            config.clear_flag(trt.BuilderFlag.TF32) # TF32 enabled by default, need to clear flag
        elif precision == 'tf32':
            pass
        elif precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)

        # build
        serialized_engine = TRT_BUILDER.build_serialized_network(network, config)
        # engine = TRT_BUILDER.build_engine(network, config) # just engine
        
        ## save TRT engine
        engine_filename = '_'.join([MODEL_NAME, gpu_name, precision]) + '.engine'
        with open(engine_filename, 'wb') as f:
            f.write(serialized_engine)
        print(f'Engine is saved to {engine_filename}')

def test_engine():

    for precision in TEST:
        ## load and deserialize TRT engine
        engine_filename = '_'.join([MODEL_NAME, gpu_name, precision]) + '.engine'
        print(f'Running inference on engine {engine_filename}')

        model = TRTModel(engine_filename)
        
        ## psuedo-random input test
        batch_size = 1
        seq_len = 2048
        vocab = 128203
        gpu = torch.device('cuda')
        torch.manual_seed(0) # make sure in each test the seed are the same
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long, device=gpu)
        attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=gpu)
        inputs = [input_ids, attention_mask]
        outputs, duration = model(inputs, timing=True)

        nreps = 100
        duration_total = 0
        for _ in range(nreps):
            outputs, duration = model(inputs, timing=True)
            duration_total += duration
        
        print(f'Average Inference time (ms) of {nreps} runs: {duration_total/nreps*1000:.3f}')

def correctness_check_engines():
    for precision in CORRECTNESS:
        ## load and deserialize TRT engine
        engine_filename1 = '_'.join([ONNX_MODEL+'_correctness_check_original', gpu_name, precision]) + '.engine'
        engine_filename2 = '_'.join([ONNX_MODEL+'_correctness_check_plugin', gpu_name, precision]) + '.engine'
        if not os.path.exists(engine_filename1) or not os.path.exists(engine_filename2):
            print('At least one of the original and/or plugin engines do not exist. Please build them first by --build')
            return 

        print(f'Running inference on original engine {engine_filename1} and plugin engine {engine_filename2}')

        model1 = TRTModel(engine_filename1)
        model2 = TRTModel(engine_filename2)
        
        ## psuedo-random input test
        batch_size = 1
        seq_len = 2048
        vocab = 128203
        gpu = torch.device('cuda')
        # torch.manual_seed(0) # make sure in each test the seed are the same
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long, device=gpu)
        attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=gpu)
        inputs = [input_ids, attention_mask]
        
        outputs1, _ = model1(inputs)
        outputs2, _ = model2(inputs)

        ## MSFT data test
        # TBD (load customer's test data)

        # element-wise and layer-wise output comparison
        for i in range(len(outputs1)):
            avg_abs_error = torch.sum(torch.abs(torch.sub(outputs1[i], outputs2[i]))) / torch.numel(outputs1[i])
            max_abs_error = torch.max(torch.abs(torch.sub(outputs1[i], outputs2[i])))
            print(f"[Layer {i} Element-wise Check] Avgerage absolute error: {avg_abs_error.item():e}, Maximum absolute error: {max_abs_error.item():e}. Between the order of 1e-3 and 1e-4 is the expected precision for FP16 (10 significance bits)" ) # machine epsilon for different precisions: https://en.wikipedia.org/wiki/Machine_epsilon

if BUILD:
    build_engine()

if TEST:
    test_engine()

if CORRECTNESS:
    correctness_check_engines()

