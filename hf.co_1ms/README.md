
### Quick Summary - Use vendor supplied Pytorch and you will get the same performance as Infinity (as of 10/3/2021)
Repackage OneDNN/DNNL on CPU and CUDNN for TensorRT/Tensorcore and you have Infinity without $20k/cpu/yr


Reconstruted Demos from launch Video here: https://www.youtube.com/watch?v=jiftCAhOYQA

Infinity CPU Inference Dual-core Cascade lake VM:
Seq length 16:  2.6ms
![cpu 16](images/cpu_16_2_5ms.png)
Seq length 128:  9.7ms
![gpu 128](images/cpu_9_7ms.png)

Infinity GPU Inference Quad-core Cascade lake VM + 1 T4 GPU:
Seq length 16:  1.7ms
![cpu 16](images/gpu_16_1_7ms.png)
Seq length 128:  2.6ms
![gpu 128](images/gpu_128_2_6ms.png)


The original model used in the video is available here: https://huggingface.co/philschmid/MiniLM-L6-H384-uncased-sst2

The optimized "Infinity Model" switch is basically the QNNX quantized model is available here:
	https://huggingface.co/philschmid/Infinity_cpu_MiniLM_L6_H384_uncased_sst2

# To Infinity and Beyond
For our experiments we want to start from the original model to see if we can reach the demo'ed metrics. 

Setup your Python ENV
```
python3.9 -m venv ~/1msenv
source ~/1msenv/bin/activate
pip install --upgrade pip
pip install --upgrade onnx coloredlogs packaging psutil py3nvml onnxconverter_common numpy transformers sympy wheel

# This is the compare regular PyTorch / Torchscript performance
# To install Intel's Pytorch enahcements -- you will recreate the "1ms" demos with this at 9.8ms
# uninstall with pip uninstall torch torch-ipex
pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable

# To install stock Pytorch nighty -- you will run a couple ms slower at 11ms
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

```

## build ONNX with OneDNN and CUDNN

```
./build.sh --config Release --build_wheel --parallel --use_openmp --use_dnnl --skip_tests --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda
#find . -name *.whl
./build/Linux/Release/dist/onnxruntime_gpu-1.10.0-cp39-cp39-linux_x86_64.whl
./build/Linux/Release/dist/onnxruntime_dnnl-1.10.0-cp39-cp39-linux_x86_64.whl

pip install ./build/Linux/Release/dist/onnxruntime_dnnl-1.10.0-cp39-cp39-linux_x86_64.whl ./build/Linux/Release/dist/onnxruntime_gpu-1.10.0-cp39-cp39-linux_x86_64.whl
```

## Approaching Infinity
Run the Benchmark script in this folder. Change the parameters to GPU if you are doing a gpu run

```
./hf.co_1ms/run_benchmark.sh
```

## Are we there yet? 

## CPU Benchmark Results

| Seq.Len |  1.11-dev Torchscript (FP32) | 1.11-dev Torchscript (INT8) | Intel 1.9.0 Torchscript (FP32) | Intel 1.9.0 Torchscript (Int8) | ONNX (FP32) | ONNX (Int8) |
|---------| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 16 |6.14|2.49|5.86|1.96|2.76|1.24|
| 128 |17.39|11.67|16.65|9.59|13.63|7.48|

## GPU Benchmark Results

| Seq.Len |  Nightly Torchscript (FP32) | Nightly Torchscript (INT8) | Intel Torchscript (FP32) | Intel TS (Int8) | ONNX (FP32) | ONNX (Int8) |
|---------| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 16 |      |        |||||
| 128 | |    |        | ||||


### Sample CPU run

# 'average_latency_ms': '9.59'  vs Infinity's '9.7ms'

```
./hf.co_1ms/run_benchmark.sh

Run onnxruntime on philschmid/MiniLM-L6-H384-uncased-sst2 with input shape [1, 16]
{'engine': 'onnxruntime', 'version': '1.10.0', 'device': 'cpu', 'optimizer': True, 'precision': <Precision.INT8: 'int8'>, 'io_binding': True, 'model_name': 'philschmid/MiniLM-L6-H384-uncased-sst2', 'inputs': 1, 'threads': 2, 'batch_size': 1, 'sequence_length': 16, 'datetime': '2021-10-04 03:50:45.708862', 'test_times': 100, 'latency_variance': '0.00', 'latency_90_percentile': '1.36', 'latency_95_percentile': '1.41', 'latency_99_percentile': '1.77', 'average_latency_ms': '1.24', 'QPS': '805.29'}
Run onnxruntime on philschmid/MiniLM-L6-H384-uncased-sst2 with input shape [1, 128]
{'engine': 'onnxruntime', 'version': '1.10.0', 'device': 'cpu', 'optimizer': True, 'precision': <Precision.INT8: 'int8'>, 'io_binding': True, 'model_name': 'philschmid/MiniLM-L6-H384-uncased-sst2', 'inputs': 1, 'threads': 2, 'batch_size': 1, 'sequence_length': 128, 'datetime': '2021-10-04 03:50:45.837148', 'test_times': 100, 'latency_variance': '0.00', 'latency_90_percentile': '7.62', 'latency_95_percentile': '7.70', 'latency_99_percentile': '7.75', 'average_latency_ms': '7.48', 'QPS': '133.75'}
Detail results are saved to csv file: detail.csv
Summary results are saved to csv file: result.csv
Arguments: Namespace(models=['philschmid/MiniLM-L6-H384-uncased-sst2'], model_source='pt', model_class=None, engines=['torchscript'], cache_dir='./cache_models', onnx_dir='./onnx_models', use_gpu=False, precision=<Precision.INT8: 'int8'>, verbose=False, overwrite=False, optimize_onnx=True, validate_onnx=False, fusion_csv='fusion.csv', detail_csv='detail.csv', result_csv='result.csv', input_counts=[1], test_times=100, batch_sizes=[1], sequence_lengths=[16, 128], disable_ort_io_binding=False, num_threads=[2])
Some weights of the model checkpoint at philschmid/MiniLM-L6-H384-uncased-sst2 were not used when initializing BertModel: ['classifier.bias', 'classifier.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Size of full precision Torch model(MB):86.6843729019165
Size of quantized Torch model(MB):55.913846015930176
Run PyTorch on philschmid/MiniLM-L6-H384-uncased-sst2 with input shape [1, 16]
{'engine': 'torchscript', 'version': '1.9.0+cpu', 'device': 'cpu', 'optimizer': '', 'precision': <Precision.INT8: 'int8'>, 'io_binding': '', 'model_name': 'philschmid/MiniLM-L6-H384-uncased-sst2', 'inputs': 1, 'threads': 2, 'batch_size': 1, 'sequence_length': 16, 'datetime': '2021-10-04 03:50:50.498982', 'test_times': 100, 'latency_variance': '0.00', 'latency_90_percentile': '2.02', 'latency_95_percentile': '2.04', 'latency_99_percentile': '2.14', 'average_latency_ms': '1.96', 'QPS': '509.34'}
Run PyTorch on philschmid/MiniLM-L6-H384-uncased-sst2 with input shape [1, 128]
{'engine': 'torchscript', 'version': '1.9.0+cpu', 'device': 'cpu', 'optimizer': '', 'precision': <Precision.INT8: 'int8'>, 'io_binding': '', 'model_name': 'philschmid/MiniLM-L6-H384-uncased-sst2', 'inputs': 1, 'threads': 2, 'batch_size': 1, 'sequence_length': 128, 'datetime': '2021-10-04 03:50:52.568732', 'test_times': 100, 'latency_variance': '0.00', 'latency_90_percentile': '9.79', 'latency_95_percentile': '9.84', 'latency_99_percentile': '9.94', 'average_latency_ms': '9.59', 'QPS': '104.32'}
```


