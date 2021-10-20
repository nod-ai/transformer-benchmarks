#!/bin/bash

git clone https://github.com/powderluv/transformer-benchmarks --recursive
cd transformer-benchmarks
./run_benchmark.sh --cpu_fp32=true --create_venv=true --ort=true --torchscript=true --tensorflow=true --mlir=true --ort_optimizer=false
./run_benchmark.sh --gpu_fp32=true --create_venv=true --ort=true --torchscript=true --tensorflow=true --mlir=true --ort_optimizer=false

TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

gsutil cp *.csv gs://iree-shared-files/nod-perf/results/transformer-bench/$(TIMESTAMP)/
