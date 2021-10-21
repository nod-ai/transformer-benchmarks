#!/bin/bash -x

NO_SRC=false

TVM_TUNED_CPU=$HOME/tvm_tuned_cpu
TVM_TUNED_GPU=$HOME/tvm_tuned_gpu

while getopts “n” OPTION
do
     case $OPTION in
         n)
             echo "Not checking out src tree.. running from current checkout.."
             NO_SRC=true
             ;;
         ?)
             echo "Unsupported option.. -n for no checkout and run as developer instead of a CI"
             exit
             ;;
     esac
done

if [ "$NO_SRC" = true ]; then
  echo "Using existing checkout"
else
  echo "Checking out transformer-benchmarks..."
  git clone https://github.com/powderluv/transformer-benchmarks --recursive
  cd transformer-benchmarks
  git submodule update --init --recursive
  cd mmperf/external/iree
  git submodule update --init --recursive
  cd -
  echo "Updating submodules to origin/main...things may break.. but that is the point.."
  ./update_submodules.sh
fi

#E2E Transformer benchmarks
./run_benchmark.sh --cpu_fp32=true --create_venv=true --ort=true --torchscript=true --tensorflow=true --mlir=true --ort_optimizer=false
./run_benchmark.sh --gpu_fp32=true --create_venv=true --ort=true --torchscript=true --tensorflow=true --mlir=true --ort_optimizer=false

#Gather results
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

mkdir -p  transformer-bench-results/${TIMESTAMP}
cd transformer-bench-results
ln -s ${TIMESTAMP} latest
cd ../
cp *.csv transformer-bench-results/latest/BERT_e2e/

#mmperf tests
cd mmperf

rm -rf mmperf_env
python -m venv mmperf_env
source mmperf_env/bin/activate
pip install -r requirements.txt

#CPU tests

if [ -d ${TVM_TUNED_CPU} ]; then
  echo "Using TVM TUNED for CPU"
  cmake -GNinja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DMKL_DIR=/opt/intel/oneapi/mkl/latest/ -DUSE_TVM=ON -DUSE_MLIR=ON -DUSE_IREE=ON -DIREE_DYLIB=ON -DUSE_TVM_TUNED=ON -DTVM_LIB_DIR=${TVM_TUNED_CPU} -B build .
else
  echo "No TVM tuned libs so skipping.."
  cmake -GNinja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DMKL_DIR=/opt/intel/oneapi/mkl/latest/ -DUSE_MLIR=ON -DUSE_IREE=ON -DIREE_DYLIB=ON -B build .
fi

#build mmperf
cmake --build build

#Run all tests and generate the plots
#cmake --build build/matmul --target run_all_tests

python mmperf.py build/matmul  ../transformer-bench-results/latest/mmperf-cpu/

mv build build.cpu

#GPU tests
if [ -d ${TVM_TUNED_GPU} ] ; then
  echo "Using TVM TUNED for GPU"
  cmake -GNinja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DMKL_DIR=/opt/intel/oneapi/mkl/latest/ -DCMAKE_CUDA_COMPILER=nvcc -DUSE_TVM=ON -DUSE_MLIR=ON -DUSE_IREE=ON -DIREE_CUDA=ON -DUSE_CUBLAS=ON -DUSE_TVM_CUDA=ON -DTVM_ENABLE_CUDA=ON -DUSE_TVM_TUNED=ON -DTVM_LIB_DIR=${TVM_TUNED_GPU} -B build .
else
  echo "No TVM tuned libs so skipping.."
  cmake -GNinja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_CUDA_COMPILER=nvcc -DUSE_MLIR=ON -DUSE_IREE=ON -DIREE_CUDA=ON -DUSE_CUBLAS=ON -B build .
fi

#build mmperf
cmake --build build

#Run all tests and generate the plots
#cmake --build build/matmul --target run_all_tests

python mmperf.py build/matmul  ../transformer-bench-results/latest/mmperf-gpu/

mv build build.gpu

cd ..

echo "Copying to Google Storage.."
gsutil cp -r transformer-bench-results/* gs://iree-shared-files/nod-perf/results/transformer-bench/

if [ "$NO_SRC" = true ]; then
  echo "leaving sources and results for manual clean up"
else
  cd ..
  echo "deleting transformer-benchmarks..."
  rm -rf transformer-bench
fi
