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

'''
Generate HuggingFace DeBERTa (V2) model with different configurations (e.g., sequence length, hidden size, No. of layers, No. of heads, etc.) and export in ONNX format

Usage:
    python deberta_pytorch2onnx.py [--filename xx.onnx]

Notes: 
1. HuggingFace's DeBERTaV2 implementation is at https://github.com/huggingface/transformers/tree/master/src/transformers/models/deberta_v2. A few changes need to be applied to the original implementation to remove the int8 related conversion:
    - search for `.byte()` and remove. `.byte()` is same as conversion to torch.int8. Int8 cast nodes are not currently supported in TensorRT
    - in XSoftmax.symbolic() function, change `to_i=sym_help.cast_pytorch_to_onnx["Byte"]` to `to_i=sym_help.cast_pytorch_to_onnx["Int"]` and `masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8)))` to `masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.int)))`
2. To apply the changes, use either of the following options:
    - if prefer using `from transformers import DebertaV2Model`, remember to follow editable install of HuggingFace (https://huggingface.co/docs/transformers/installation#editable-install), otherwise any changes made to the model won't apply
    - or make a copy of the `modeling_deberta_v2.py`, apply changes, and directly import from the file, `from .modeling_deberta_v2 import DebertaV2Model`
'''

import os, time, argparse 
from transformers import DebertaV2Tokenizer, DebertaV2Config # HuggingFace, editable pip install transformers
from modeling_deberta_v2 import DebertaV2ForSequenceClassification # from modified DeBERTa implementation as a single file to fix the int8 cast ops (but first install HF's transformers library)
import torch, onnxruntime as ort, numpy as np

parser = argparse.ArgumentParser(description="Generate HuggingFace DeBERTa (V2) model with different configurations and export in ONNX format. This will save the model under the same directory as 'deberta_seqxxx_hf.onnx'.")
parser.add_argument('--filename', type=str, help='Path to the save the ONNX model')

args = parser.parse_args()
onnx_filename = args.filename

assert torch.cuda.is_available(), "CUDA not available!" # if not, manully config CUDA and cuDNN installed path in ENV: "CUDA_PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/r11.4/x86_64/latest/bin CUDNN_PATH=/home/scratch.svc_compute_arch/release/cudnn/v8.2_cuda_11.3/x86_64/latest/bin PATH=${CUDA_PATH}:${CUDNN_PATH}:${PATH} python ..."

def export():
    # hyper params
    batch_size = 1
    seq_len = 2048
    max_position_embeddings = 512 if seq_len <= 512 else seq_len # maximum sequence length that this model might ever be used with. By default 512. otherwise error https://github.com/huggingface/transformers/issues/4542
    vocab = 128203
    hidden_size = 384
    layers = 12
    heads = 6
    intermediate_size = hidden_size*4 # feed forward layer dimension
    type_vocab_size = 0
    # relative attention
    relative_attention=True
    max_relative_positions = 256 # k
    pos_att_type = ["p2c", "c2p"]

    # DEBERTA V2, https://github.com/huggingface/transformers/blob/master/src/transformers/models/deberta_v2/modeling_deberta_v2.py
    deberta_model_name = f"deberta_seq{seq_len}_hf.onnx" if onnx_filename is None else onnx_filename
    deberta_config = DebertaV2Config(vocab_size=vocab, hidden_size=hidden_size, num_hidden_layers=layers, num_attention_heads=heads, intermediate_size=intermediate_size, type_vocab_size=type_vocab_size, max_position_embeddings=max_position_embeddings, relative_attention=relative_attention, max_relative_positions=max_relative_positions, pos_att_type=pos_att_type)
    deberta_model = DebertaV2ForSequenceClassification(deberta_config)
    deberta_model.cuda().eval()

    # input/output
    gpu = torch.device('cuda')
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long, device=gpu)
    attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=gpu)
    input_names = ['input_ids', 'attention_mask']   
    output_names = ['output']
    dynamic_axes={'input_ids'   : {0 : 'batch_size'}, 
                  'attention_mask'   : {0 : 'batch_size'},   
                  'output' : {0 : 'batch_size'}}
    
    # ONNX export
    torch.onnx.export(deberta_model, # model 
                     (input_ids, attention_mask), # model inputs
                     deberta_model_name,
                     export_params=True,
                     opset_version=13,
                     do_constant_folding=True,
                     input_names = input_names,
                     output_names = output_names,
                     dynamic_axes = dynamic_axes)
    
    # full precision inference
    trials = 10

    start = time.time()
    for i in range(trials):
        results = deberta_model(input_ids, attention_mask)
        # print(results)
    end = time.time()

    print("Average PyTorch FP32/TF32 time: {:.2f} ms".format((end - start)/trials*1000))
    
    # half precision inference (do this after onnx export, otherwise the export ONNX model is with FP16 weights...)
    deberta_model_fp16 = deberta_model.half()
    start = time.time()
    for i in range(trials):
        results = deberta_model_fp16(input_ids, attention_mask)
        # print(results)
    end = time.time()

    print("Average PyTorch FP16 time: {:.2f} ms".format((end - start)/trials*1000))
    
    # model size
    total_params = sum(param.numel() for param in deberta_model.parameters())
    print("Total # of params: ", total_params)

def test(onnx_path):
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1" #TRT precision: 1: TRT FP16, 0: TRT FP32
    sess_opt = ort.SessionOptions()
    execution_provider = ["TensorrtExecutionProvider", "CUDAExecutionProvider"] 
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opt, providers=execution_provider)

    np.random.seed(0)
    input_dims = [1, 2048]
    input_ids = np.random.randint(30522, size=input_dims, dtype=np.int64)
    attention_mask = np.ones(input_dims, dtype=np.int64)

    numpy_input = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    warmup = sess.run(None, numpy_input)

    print('Start inference...')
    max_iters = 100
    predict = {}
    start_time = time.time()
    for iter in range(max_iters):
        predict = sess.run(None, numpy_input)

    print("average %s seconds ---" % ((time.time() - start_time)/max_iters))

if __name__ == "__main__":
    export()
    # test(onnx_filename) # ORT can't run Deberta model yet (not until TRT 8.2.4 or 8.4 EA & with associated changes made in onnxruntime and onnx-tensorrt parser)

    