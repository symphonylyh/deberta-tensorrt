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

cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=$TRT_OSSPATH/build/out
make -j$(nproc)

# for TRT to be later built into ORT, adding OSS build path to LD_LIBRARY_PATH is not working. We have to do a hard replacment of libnvinfer_plugin.so and libnvinfer_static.a (the root one, not the symlinks) to TensorRT lib
cp ./out/libnvinfer_plugin.so.8.2.4 ${TRT_LIBPATH}
cp ./out/libnvinfer_static.a ${TRT_LIBPATH}
