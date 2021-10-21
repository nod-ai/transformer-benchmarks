#!/bin/bash -x

COMMIT_PUSH=false

while getopts “p” OPTION
do
     case $OPTION in
         p)
             echo "Pushing changes up.."
             COMMIT_PUSH=true
             ;;
         ?)
             echo "Unsupported option.. -p for pushing changes up after update"
             exit
             ;;
     esac
done

echo "Updating repos.."

cd mmperf && git fetch --all && git checkout origin/main

#update mmperf submodules first
git submodule update --init
#Update the submodules inside mmperf too
./update_submodules.sh

if [ "$COMMIT_PUSH" = true ]; then
  echo "Checking out transformer-benchmarks..."
  git add .
  git commit -m "Roll external deps"
  echo git push https://github.com/mmperf/mmperf
fi
