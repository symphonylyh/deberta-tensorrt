# save TRT engine
LD_LIBRARY_PATH=./TensorRT-8.4.0.6/lib ./TensorRT-8.4.0.6/bin/trtexec --onnx=deberta_plugin.onnx --plugin=./plugin/build/bin/libcustomplugins.so --workspace=4096 --explicitBatch --optShapes=input_ids:1x2048,attention_mask:1x2048 --iterations=10 --warmUp=10 --noDataTransfers --fp16 --saveEngine=deberta_plugin_trtexec.engine

# load TRT engine
LD_LIBRARY_PATH=./TensorRT-8.4.0.6/lib ./TensorRT-8.4.0.6/bin/trtexec --plugin=./plugin/build/bin/libcustomplugins.so --workspace=4096 --explicitBatch --optShapes=input_ids:1x2048,attention_mask:1x2048 --iterations=10 --warmUp=10 --noDataTransfers --fp16 --loadEngine=deberta_plugin_trtexec.engine

# nsys profiling
LD_LIBRARY_PATH=./TensorRT-8.4.0.6/lib nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o deberta_plugin_trtexec.nsys-rep --force-overwrite true ./TensorRT-8.4.0.6/bin/trtexec --plugin=./plugin/build/bin/libcustomplugins.so --workspace=4096 --explicitBatch --optShapes=input_ids:1x2048,attention_mask:1x2048 --iterations=10 --warmUp=10 --noDataTransfers --fp16 --loadEngine=deberta_plugin_trtexec.engine