
- [Instructions of Using NVIDIA TensorRT Plugin in ONNX Runtime for Microsoft DeBERTa Model](#instructions-of-using-nvidia-tensorrt-plugin-in-onnx-runtime-for-microsoft-deberta-model)
  - [Background](#background)
  - [Download](#download)
  - [Step 1: PyTorch Model to ONNX graph](#step-1-pytorch-model-to-onnx-graph)
  - [Step 2: Modify ONNX graph with TensorRT plugin nodes](#step-2-modify-onnx-graph-with-tensorrt-plugin-nodes)
  - [Step 3: Build TensorRT plugin](#step-3-build-tensorrt-plugin)
  - [Step 4: Test DeBERTa model with TensorRT plugin (Python TensorRT API and `trtexec`)](#step-4-test-deberta-model-with-tensorrt-plugin-python-tensorrt-api-and-trtexec)
  - [Optional Step: Correctness Check of Modified DeBERTa Model](#optional-step-correctness-check-of-modified-deberta-model)

***

## Instructions of Using NVIDIA TensorRT Plugin in ONNX Runtime for Microsoft DeBERTa Model

### Background
Performance gap has been observed between Google's [BERT](https://arxiv.org/abs/1810.04805) design and Microsoft's [DeBERTa](https://arxiv.org/abs/2006.03654) design. The main reason of the gap is the disentangled attention design in DeBERTa triples the attention computation over BERT's regular attention. In addition to the extra matrix multiplications, the disentangled attention design also involves indirect memory accesses during the gather operations. This TensorRT plugin is designed to optimize DeBERTa's disentangled attention module.

This TensorRT plugin works for the [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/deberta_v2) of DeBERTa and includes code and scripts for (i) exporting ONNX model fro PyTorch, (ii) modifying ONNX model by inserting the plugin nodes, (iii) CUDA TensorRT implementation of the optimized disentangled attention, and (iv) measuring the correctness and performance of the optimized model.

Detailed steps are given as follows.

### Download
```bash
# this repo
git clone -b ort-trt https://github.com/symphonylyh/deberta-tensorrt.git
cd deberta-tensorrt

# TensorRT OSS
git clone -b main https://github.com/symphonylyh/TensorRT.git
cd TensorRT && git submodule update --init --recursive && cd ..

# onnxruntime
git clone -b deberta_trt_plugin https://github.com/symphonylyh/onnxruntime.git
cd onnxruntime && git submodule update --init --recursive && cd ..
```

Note: this repo has two dependencies: `TensorRT` and `onnxruntime` (adding as submodules is problematic since TensorRT repo has nested submodules too). The dependencies currently point to my forked version of TensorRT OSS and onnxruntime with all necessary changes to enable the plugin before public release. After all the following changes have been released publicly, the repos should be directed to the official repos:
* TensorRT OSS release of the disentangled attention plugin implementation
* onnx-tensorrt release of the supported plugin operator
* onnxruntime release of the supported plugin operator

<!-- To change the submodule's URL, you can either modify the .gitmodules manually, or update the upstream (using TensorRT for example below).

```bash
cd TensorRT
git remote add upstream https://github.com/NVIDIA/TensorRT.git 
git fetch upstream
git checkout upstream/main
``` -->

### Docker Setup
It is recommended to use docker for reproducing the following steps. Docker file `deberta.dockefile` configures the docker environment on top of public [NGC TensorRT container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt).

```bash
docker build --build-arg uid=$(id -u) --build-arg gid=$(id -g) -t deberta:latest -f deberta.dockerfile . # this only needs be done once on the same machine

docker run --gpus all -it --rm --memory=8g --user $(id -u):$(id -g) -v $(pwd):/workspace/ deberta:latest # run the docker (sudo password: nvidia)

# build TensorRT OSS and ONNX Runtime
./build_trt.sh
./built_ort.sh
```

Note: the docker container is installed with TensorRT version 8.2.4.2.

### Step 1: PyTorch Model to ONNX graph
```bash
python deberta_pytorch2onnx.py --filename deberta.onnx
```

This will export the DeBERTa model (`modeling_deberta_v2.py`) from HuggingFace's PyTorch implementation into ONNX format, with user given file name. Note there are several modifications from HuggingFace's implementation to remove the int8 cast operations. Details of the modification can be found in the description in `deberta_pytorch2onnx.py`.

### Step 2: Modify ONNX graph with TensorRT plugin nodes
```bash
python deberta_onnx_modify.py deberta.onnx
```
To enable TensorRT plugin for disentangled attention, the ONNX graph need to be modified by replacing a subgraph with the plugin node. This file automates the procedure and will by default save the modified graph with suffix `_plugin.onnx` in the file name. Or you can specify `--output [filename]`.

### Step 3: Build TensorRT plugin
```bash
cd ./plugin
make
cd ..
```
This will build the disentangled attention plugin as a `.so` shared library at `./plugin/build/bin/libcustomplugins.so`.

### Step 4: Test DeBERTa model with TensorRT plugin (Python TensorRT API and `trtexec`)
```bash
# build and test the original DeBERTa model (baseline)
python deberta_tensorrt_inference.py --onnx=deberta.onnx --build fp16 --plugin=./plugin/build/bin/libcustomplugins.so 
python deberta_tensorrt_inference.py --onnx=deberta.onnx --test fp16 --plugin=./plugin/build/bin/libcustomplugins.so 

# build and test the optimized DeBERTa model with plugin
python deberta_tensorrt_inference.py --onnx=deberta_plugin.onnx --build fp16 --plugin=./plugin/build/bin/libcustomplugins.so 
python deberta_tensorrt_inference.py --onnx=deberta_plugin.onnx --test fp16 --plugin=./plugin/build/bin/libcustomplugins.so 
```
This will build and test the original and optimized DeBERTa models. `--build` to build the engine from ONNX model, and `--test` to measure the optimized latency. TensorRT engine of different precisions can be built (but currently only supported fp16 for DeBERTa's use case).

Engine files are saved as `[Model name]_[GPU name]_[Precision].engine`. Note that TensorRT engines are specific to the exact GPU model they were built on, as well as the platform and the TensorRT version. On the same machine, build is needed only once and the engine will be saved for repeatedly testing.

For `trtexec` test, several example commands are provided in `deberta_trtexec.sh`.

### Optional Step: Correctness Check of Modified DeBERTa Model
```bash
# prepare the ONNX models with intermediate output nodes (this will save two new onnx models with suffix `_correctness_check_original.onnx` and `_correctness_check_plugin.onnx`)
python deberta_onnx_modify.py deberta.onnx --correctness_check

# build the ONNX models with intermediate outputs
python deberta_tensorrt_inference.py --onnx=deberta_correctness_check_original.onnx --build fp16 --plugin=./plugin/build/bin/libcustomplugins.so
python deberta_tensorrt_inference.py --onnx=deberta_correctness_check_plugin.onnx --build fp16 --plugin=./plugin/build/bin/libcustomplugins.so

# run correctness check (specify the root model name with --onnx)
python deberta_tensorrt_inference.py --onnx=deberta --correctness_check fp16 --plugin=./plugin/build/bin/libcustomplugins.so
```

Correctness check requires intermediate outputs from the model, thus it is necessary to modify the ONNX graph and add intermediate output nodes. The correctness check was added at the location of plugin outputs in each layer. The metric is average and maximum of the element-wise absolute error. Note that for FP16 precision with 10 significance bits, absolute error in the order of 1e-3 and 1e-4 is expected. Refer to [Machine Epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) for details. 



