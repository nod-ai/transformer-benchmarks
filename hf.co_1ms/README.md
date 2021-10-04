Launch Video is here: https://www.youtube.com/watch?v=jiftCAhOYQA

tl;dr: Nothing special so far in Infinity. Repackage OneDNN/DNNL on CPU and CUDNN for TensorRT/Tensorcore and you have Infinity without $20k/cpu/yr

Infinity CPU Inference Dual-core Cascade lake VM:
Seq length 16:  2.6ms
![cpu 16](images/cpu_16_2_5ms.png)
Seq length 128:  9.8ms
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
pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable

# To install stock Pytorch nighty -- you will run a couple ms slower at 11ms
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

```

## build ONNX DNNL
./build.sh --config Release --build_wheel --parallel --use_openmp --use_dnnl --skip_tests

### Now install the build
pip install ./build/Linux/Release/dist/onnxruntime_dnnl-1.10.0-cp39-cp39-linux_x86_64.whl

## build ONNX CUDNN

./build.sh --config Release --build_wheel --parallel --use_openmp --use_cuda --cuda_home --cudnn_home --skip_tests



  897  pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html                                               
  899  pip install sympy
  937  python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable                                                                            
  938  pip uninstall torchvision torchaudio
