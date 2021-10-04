Launch Video is here: https://www.youtube.com/watch?v=jiftCAhOYQA



The model used is available here: https://huggingface.co/philschmid/MiniLM-L6-H384-uncased-sst2

The optimized "Infinity Model" switch is basically the QNNX quantized model is available here:
	https://huggingface.co/philschmid/Infinity_cpu_MiniLM_L6_H384_uncased_sst2


Setup your Python ENV
```
python3.9 -m venv ~/1msenv
source ~/1msenv/bin/activate
pip install --upgrade pip
pip install --upgrade onnx coloredlogs packaging psutil py3nvml onnxconverter_common numpy transformers sympy 

# To install Intel's Pytorch enahcements -- you will recreate the "1ms" demos with this at 9.8ms
pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable

# To install stock Pytorch nighty -- you will run a couple ms slower at 11ms
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

```

## build ONNX
./build.sh --config Release --build_wheel --parallel --use_openmp --use_dnnl --skip_tests


  897  pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html                                               
  899  pip install sympy
  937  python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable                                                                            
  938  pip uninstall torchvision torchaudio
