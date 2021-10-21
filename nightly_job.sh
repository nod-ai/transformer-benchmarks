#!/bin/bash

TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
[ -d $HOME/ci ] || mkdir $HOME/ci
log_file=$HOME/ci/nightly_log_${TIMESTAMP}.txt
exec &> >(tee -a "$log_file")

rm -rf $HOME/ci/nightly
mkdir -p $HOME/ci/nightly
cd $HOME/ci/nightly
curl -O --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/powderluv/transformer-benchmarks/main/perf-ci.sh
chmod +x $HOME/ci/nightly/perf-ci.sh
$HOME/ci/nightly/perf-ci.sh
gsutil cp $log_file gs://iree-shared-files/nod-perf/logs/
