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

## 1. build (only once)
# $ docker build --rm --no-cache -t deberta:latest -f deberta.dockerfile .
## 2. run
# $ docker run --gpus all -it --rm --memory=8g --user $(id -u):$(id -g) -v $(pwd):/deberta/ --workdir /deberta/ deberta:latest

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:22.03-py3 
# Note: this base container is TRT 8.2.3, which contains Myelin bugs that prevents general engine building of DeBERTa (Error 1) and for FP16 engine (Error 2). Upgrade it to TRT 8.4 EA (8.4.0.6) in docker_setup.sh

FROM ${BASE_IMAGE}

COPY docker_setup.sh /
RUN chmod +x /docker_setup.sh && /docker_setup.sh
RUN echo "Container setup finished!"