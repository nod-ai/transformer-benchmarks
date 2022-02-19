#!/bin/bash -e
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Copyright (c) Nod, Inc.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This measures the performance of OnnxRuntime, PyTorch and TorchScript on transformer models.
# Please install PyTorch or Tensorflow or MLIR Runtime (see https://pytorch.org/) before running this benchmark. Like the following:
# GPU:   conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
# CPU:   conda install pytorch torchvision cpuonly -c pytorch

# When use_package=true, you need not copy other files to run benchmarks except this sh file.
# Otherwise, it will use python script (*.py) files in this directory.

ARGUMENT_LIST=(
    "gpu_fp32"
    "gpu_fp16"
    "cpu_fp32"
    "cpu_int8"
    "ort"
    "torch"
    "torchscript"
    "tensorflow"
    "mlir"
    "pip_install_pkg"
    "ort_optimizer"
    "create_venv"
    "with_nsys"
)

#setup defaults
# Devices to test (You can run either CPU or GPU, but not both: gpu need onnxruntime-gpu, and CPU need onnxruntime).
run_gpu_fp32=true
run_gpu_fp16=false
run_cpu_fp32=false
run_cpu_int8=false
# Engines to test.
run_ort=false
run_torch=false
run_torchscript=false
run_tensorflow=true
run_mlir=false

# only need once
run_create_venv=false
install_pkg=false
run_with_nsys=true

# Enable optimizer (use script instead of OnnxRuntime for graph optimization)
use_optimizer=false

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with_nsys)
            run_with_nsys=$2
            shift 2
            ;;
        --create_venv)
            run_create_venv=$2
            echo  "Removing old bench_venv.."
            rm -rf bench_venv
            echo  "Creating new bench_venv.."
            python -m venv bench_venv
            echo  "sourcing new env.."
            source bench_venv/bin/activate
            shift 2
            ;;

        --ort_optimizer)
            use_optimizer=$2
            shift 2
            ;;

        --pip_install_pkg)
            install_pkg=$2
            shift 2
            ;;

        --mlir)
            run_mlir=$2
            shift 2
            ;;

        --tensorflow)
            run_tensorflow=$2
            shift 2
            ;;

        --torchscript)
            run_torchscript=$2
            shift 2
            ;;

        --torch)
            run_torch=$2
            shift 2
            ;;

        --ort)
            run_ort=$2
            shift 2
            ;;

        --gpu_fp32)
            run_gpu_fp32=$2
            shift 2
            ;;

        --gpu_fp16)
            run_gpu_fp16=$2
            shift 2
            ;;

        --cpu_fp32)
            run_cpu_fp32=$2
            shift 2
            ;;

        --cpu_int8)
            run_cpu_int8=$2
            shift 2
            ;;

        *)
            echo "Using defaults...: "
            echo "   you can change them with --var=true or false"
            break
            ;;
    esac
done


echo "Parsed command line args as:"
echo "gpu_fp32 $run_gpu_fp32"
echo "gpu_fp16 $run_gpu_fp16"
echo "cpu_fp32 $run_cpu_fp32"
echo "cpu_int8 $run_cpu_int8"

echo "ort $run_ort"
echo "ort_optimizer $use_optimizer"
echo "torch $run_torch"
echo "torchscript $run_torchscript"
echo "tensorflow $run_tensorflow"
echo "mlir $run_mlir"
echo "create_venv $run_create_venv"
echo "run_with_nsys $run_with_nsys"
echo "pip_install_pkg $install_pkg"

echo  "Check python path.. "
which python

use_package=false
# Onnx model source (default is from pytorch, set export_onnx_from_tf=true to convert from tensorflow model)
export_onnx_from_tf=false


average_over=1000
# CPU takes longer time to run, only run 100 inferences to get average latency.
if [ "$run_cpu_fp32" = true ] || [ "$run_cpu_int8" = true ]; then
  average_over=100
fi

# Batch Sizes and Sequence Lengths
batch_sizes="1"
sequence_lengths="128"

# Number of inputs (input_ids, token_type_ids, attention_mask) for ONNX model.
# Not that different input count might lead to different performance
# Here we only test one input (input_ids) for fair comparison with PyTorch.
input_counts=1

# Pretrained transformers models can be a subset of: bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased
#models_to_test="bert-base-uncased"
#models_to_test="philschmid/MiniLM-L6-H384-uncased-sst2"
models_to_test="microsoft/MiniLM-L12-H384-uncased"
#models_to_test="gpt2"

# If you have mutliple GPUs, you can choose one GPU for test. Here is an example to use the second GPU:
# export CUDA_VISIBLE_DEVICES=1

# This script will generate a logs file with a list of commands used in tests.
echo echo "ort=$run_ort torch=$run_torch torchscript=$run_torchscript tensorflow=$run_tensorflow mlir=$run_mlir gpu_fp32=$run_gpu_fp32 gpu_fp16=$run_gpu_fp16 cpu=$run_cpu optimizer=$use_optimizer batch=$batch_sizes sequence=$sequence_length models=$models_to_test" >> benchmark.log

# Set it to false to skip testing. You can use it to dry run this script with the log file.
run_tests=true

# Directory for downloading pretrained models.
cache_dir="./cache_models"

# Directory for ONNX models
onnx_dir="./onnx_models"

# -------------------------------------------
if [ "$run_cpu_fp32" = true ] || [ "$run_cpu_int8" = true ]; then
  if [ "$run_gpu_fp32" = true ] ; then
    echo "cannot test cpu and gpu at same time"
    exit 1
  fi
  if [ "$run_gpu_fp16" = true ] ; then
    echo "cannot test cpu and gpu at same time"
    exit 1
  fi
fi


if [ "$install_pkg" = true ] ; then
  pip install --upgrade pip
  pip uninstall --yes ort-nightly ort-gpu-nightly
  pip uninstall --yes onnxruntime
  pip uninstall --yes onnxruntime-gpu
  pip uninstall --yes torch
  pip uninstall --yes iree-compiler iree-runtime iree-tools-tf iree-tools-tflite iree-tools-xla

  if [ "$run_cpu_fp32" = true ] || [ "$run_cpu_int8" = true ]; then
    pip install onnxruntime
    pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
  else
    pip install onnxruntime-gpu
    pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
    #pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/rocm4.2/torch_nightly.html
  fi
  pip install tf-nightly
  pip install --upgrade onnx coloredlogs packaging psutil py3nvml onnxconverter_common numpy transformers sympy wheel
  pip install gin-config 

  ### Installing IREE-Python
  python -m pip install iree-compiler iree-runtime iree-tools-tf iree-tools-tflite iree-tools-xla --find-links https://github.com/google/iree/releases
fi

if [ "$use_package" = true ] ; then
  echo "Use onnxruntime.transformers.benchmark"
  benchmark_script="-m onnxruntime.transformers.benchmark"
else
  benchmark_script="benchmark.py"
fi

onnx_export_options="-i $input_counts -v -b 0 --overwrite -f fusion.csv -c $cache_dir --onnx_dir $onnx_dir"
benchmark_options="-b $batch_sizes -s $sequence_lengths -t $average_over -f fusion.csv -r result.csv -d detail.csv -c $cache_dir --onnx_dir $onnx_dir"

if [ "$export_onnx_from_tf" = true ] ; then
  onnx_export_options="$onnx_export_options --model_source tf"
  benchmark_options="$benchmark_options --model_source tf"
fi

if [ "$use_optimizer" = true ] ; then
  onnx_export_options="$onnx_export_options -o"
  benchmark_options="$benchmark_options -o"
fi

# -------------------------------------------
run_one_test() {
    if [ "$run_ort" = true ] ; then
      echo python $benchmark_script -m $1 $onnx_export_options $2 $3 $4 >> benchmark.log
      echo python $benchmark_script -m $1 $benchmark_options $2 $3 $4 -i $input_counts >> benchmark.log
      if [ "$run_tests" = true ] ; then
        python $benchmark_script -m $1 $onnx_export_options $2 $3 $4
        python $benchmark_script -m $1 $benchmark_options $2 $3 $4 -i $input_counts
      fi
    fi

    if [ "$run_torch" = true ] ; then
      echo python $benchmark_script -e torch -m $1 $benchmark_options $2 $3 $4 >> benchmark.log
      if [ "$run_tests" = true ] ; then
        python $benchmark_script -e torch -m $1 $benchmark_options $2 $3 $4
      fi
    fi

    if [ "$run_torchscript" = true ] ; then
      echo python $benchmark_script -e torchscript -m $1 $benchmark_options $2 $3 $4 >> benchmark.log
      if [ "$run_tests" = true ] ; then
        python $benchmark_script -e torchscript -m $1 $benchmark_options $2 $3 $4
      fi
    fi

    if [ "$run_tensorflow" = true ] ; then
      echo python $benchmark_script -e tensorflow -m $1 $benchmark_options $2 $3 $4 >> benchmark.log
      if [ "$run_tests" = true ] ; then
        python $benchmark_script -e tensorflow -m $1 $benchmark_options $2 $3 $4
      fi
    fi

    if [ "$run_mlir" = true ] ; then
      echo python $benchmark_script -e mlir -m $1 $benchmark_options $2 $3 $4 >> benchmark.log
      if [ "$run_tests" = true ] ; then
        python $benchmark_script -e mlir -m $1 $benchmark_options $2 $3 $4
      fi
    fi
}

# -------------------------------------------
if [ "$run_gpu_fp32" = true ] ; then
  for m in $models_to_test
  do
    echo Run GPU FP32 Benchmark on model ${m}
    run_one_test "${m}" -g
  done
fi

if [ "$run_gpu_fp16" = true ] ; then
  for m in $models_to_test
  do
    echo Run GPU FP16 Benchmark on model ${m}
    run_one_test "${m}" -g -p fp16
  done
fi

if [ "$run_cpu_fp32" = true ] ; then
  for m in $models_to_test
  do
    echo Run CPU Benchmark on model ${m}
    run_one_test "${m}"
  done
fi

if [ "$run_cpu_int8" = true ] ; then
  for m in $models_to_test
  do
    echo Run CPU Benchmark on model ${m}
    run_one_test "${m}" -p int8
  done
fi

if [ "run_tests" = false ] ; then
    more $log_file
fi

# Remove duplicated lines
awk '!x[$0]++' ./result.csv > summary_result.csv
awk '!x[$0]++' ./fusion.csv > summary_fusion.csv
awk '!x[$0]++' ./detail.csv > summary_detail.csv
