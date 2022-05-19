
- [Instructions of Using NVIDIA TensorRT Plugin in ONNX Runtime for Microsoft DeBERTa Model](#instructions-of-using-nvidia-tensorrt-plugin-in-onnx-runtime-for-microsoft-deberta-model)
  - [Background](#background)
  - [Download](#download)
  - [Docker Setup](#docker-setup)
  - [Step 1: PyTorch Model to ONNX graph](#step-1-pytorch-model-to-onnx-graph)
  - [Step 2: Modify ONNX graph with TensorRT plugin nodes](#step-2-modify-onnx-graph-with-tensorrt-plugin-nodes)
  - [Step 3: Test DeBERTa model with ORT performance test](#step-3-test-deberta-model-with-ort-performance-test)
  - [Step 4: Test DeBERTa model with ORT Python API](#step-4-test-deberta-model-with-ort-python-api)

***

## Instructions of Using NVIDIA TensorRT Plugin in ONNX Runtime for Microsoft DeBERTa Model

### Background
Performance gap has been observed between Google's [BERT](https://arxiv.org/abs/1810.04805) design and Microsoft's [DeBERTa](https://arxiv.org/abs/2006.03654) design. The main reason of the gap is the disentangled attention design in DeBERTa triples the attention computation over BERT's regular attention. In addition to the extra matrix multiplications, the disentangled attention design also involves indirect memory accesses during the gather operations. This TensorRT plugin is designed to optimize DeBERTa's disentangled attention module.

This TensorRT plugin works for the [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/deberta_v2) of DeBERTa and includes code and scripts for (i) exporting ONNX model fro PyTorch, (ii) modifying ONNX model by inserting the plugin nodes, (iii) CUDA TensorRT implementation of the optimized disentangled attention, and (iv) measuring the correctness and performance of the optimized model.

Detailed steps are given as follows.

### Download
```bash
# this repo
git clone https://github.com/symphonylyh/deberta-tensorrt.git
cd deberta-tensorrt # make sure on master branch
git submodule update --init --recursive
```

~~Note: this repo has two submodules: `TensorRT` and `onnxruntime`. The submodules currently point to my forked version of TensorRT OSS and onnxruntime with all necessary changes to enable the plugin before public release. After all the following changes have been released publicly, the repos should be directed to the official repos:~~
~~* TensorRT OSS release of the disentangled attention plugin implementation~~
~~* onnx-tensorrt release of the supported plugin operator~~
~~* onnxruntime release of the supported plugin operator~~

~~To change the submodule's URL later, you can either modify the .gitmodules manually, or update the upstream (using TensorRT for example below).~~

<details>
  <summary>collapsed</summary>
```bash
cd TensorRT
git remote add upstream https://github.com/NVIDIA/TensorRT.git 
git fetch upstream
git checkout upstream/main
```
</details>


### Docker Setup
It is recommended to use docker for reproducing the following steps. Docker file `deberta.dockefile` configures the docker environment on top of public [NGC TensorRT container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt).

```bash
docker build --build-arg uid=$(id -u) --build-arg gid=$(id -g) -t deberta:latest -f deberta.dockerfile . # this only needs be done once on the same machine

docker run --gpus all -it --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/ deberta:latest # run the docker (sudo password: nvidia)

# build TensorRT OSS and ONNX Runtime
./build_trt.sh
./build_ort.sh # this will pip install the ORT wheel too
```

Note: the docker container is installed with TensorRT version 8.2.4.2.

### Step 1: PyTorch Model to ONNX graph
```bash
python deberta_pytorch2onnx.py --filename ./test/deberta.onnx
```

This will export the DeBERTa model (`modeling_deberta_v2.py`) from HuggingFace's PyTorch implementation into ONNX format, with user given file name. Note there are several modifications from HuggingFace's implementation to remove the int8 cast operations. Details of the modification can be found in the description in `deberta_pytorch2onnx.py`.

### Step 2: Modify ONNX graph with TensorRT plugin nodes
```bash
python deberta_onnx_modify.py ./test/deberta.onnx
```
To enable TensorRT plugin for disentangled attention, the ONNX graph need to be modified by replacing a subgraph with the plugin node. This file automates the procedure and will by default save the modified graph with suffix `_plugin.onnx` in the file name. Or you can specify `--output [filename]`.

### Step 3: Test DeBERTa model with ORT performance test
```bash
./onnxruntime/build/Linux/Release/onnxruntime_perf_test -i "trt_fp16_enable|true" -t 10 -e tensorrt ./test/deberta.onnx | tee ./test/deberta.log

./onnxruntime/build/Linux/Release/onnxruntime_perf_test -i "trt_fp16_enable|true" -t 10 -e tensorrt ./test/deberta_plugin.onnx | tee ./test/deberta_plugin.log
```

Note that the default [onnxruntime performance test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest) requires specific directory tree for the input data and models. An example data is given at `test/test_data_set_0/`.

### Step 4: Test DeBERTa model with ORT Python API
```bash
python deberta_ort_inference.py --onnx=./test/deberta.onnx --test fp16

python deberta_ort_inference.py --onnx=./test/deberta_plugin.onnx --test fp16

python deberta_ort_inference.py --onnx=./test/deberta --correctness_check fp16 # correctness check
```

The metrics for correctness check are average and maximum of the element-wise absolute error. Note that for FP16 precision with 10 significance bits, absolute error in the order of 1e-3 and 1e-4 is expected. Refer to [Machine Epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) for details. During testing, we commonly see the error is in the order of 1e-8 for FP32, 1e-4 to 1e-5 for FP16, and for INT8. Note the correctness check in the ORT run is for final prediction results. For per-layer intermediate results check which was verified before, please refer to the `trt-test` branch.



