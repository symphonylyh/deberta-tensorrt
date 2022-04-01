#!/bin/bash

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

# NGC tensorrt:22.03-py3 container uses CUDA 11.6

# 1. update/install latest tensorrt by pip wheel (https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip)
python -m pip install --upgrade setuptools pip
python -m pip install nvidia-pyindex
python -m pip install --upgrade nvidia-tensorrt

# 2. install dependencies
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html # PyTorch
pip install transformers # HuggingFace transformers
pip install onnx onnxruntime onnx-graphsurgeon numpy argparse # miscellaneous
