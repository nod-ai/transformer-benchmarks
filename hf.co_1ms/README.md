


The model used is available here: https://huggingface.co/philschmid/MiniLM-L6-H384-uncased-sst2

The optimized "Infinity Model" switch is basically the QNNX quantized model is available here:
	https://huggingface.co/philschmid/Infinity_cpu_MiniLM_L6_H384_uncased_sst2


## build ONNX
./build.sh --config Release --build_wheel --parallel --use_openmp --use_dnnl --skip_tests


  897  pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html                                               
  899  pip install sympy
  937  python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable                                                                            
  938  pip uninstall torchvision torchaudio
