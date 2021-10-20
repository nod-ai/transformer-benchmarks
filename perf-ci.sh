#!/bin/bash -x

NO_SRC=false

while getopts “n” OPTION
do
     case $OPTION in
         n)
             NO_SRC=true
             ;;
         ?)
             "Unsupported option.. -n for no checkout and run as developer instead of a CI"
             exit
             ;;
     esac
done

if [ "$NO_SRC" = true]; then
  echo "Using existing checkout"
else
  echo "Checking out transformer-benchmarks..."
  git clone https://github.com/powderluv/transformer-benchmarks --recursive
  cd transformer-benchmarks
fi

./run_benchmark.sh --cpu_fp32=true --create_venv=true --ort=true --torchscript=true --tensorflow=true --mlir=true --ort_optimizer=false
./run_benchmark.sh --gpu_fp32=true --create_venv=true --ort=true --torchscript=true --tensorflow=true --mlir=true --ort_optimizer=false

TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`

mkdir -p  transformer-bench-results/${TIMESTAMP}
cd transformer-bench-results
ln -s ${TIMESTAMP} latest
cd ../
cp *.csv transformer-bench-results/latest/
gsutil cp -r transformer-bench-results/* gs://iree-shared-files/nod-perf/results/transformer-bench/
rm -rf transformer-bench-results
