#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cd $ORT_PATH
./build.sh --parallel --build_shared_lib --cudnn_home ${CUDNN_HOME} --cuda_home ${CUDA_HOME} --use_tensorrt --tensorrt_home ${TRT_LIBPATH} --config Release --build_wheel --skip_tests --skip_submodule_sync

# pip install ./build/Linux/Release/dist/*.whl