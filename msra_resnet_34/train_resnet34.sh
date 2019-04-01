#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=msra_resnet_34/resnet_34_solver.prototxt --snapshot=/home/hj14/caffe/resnet_34_msra_iter_80000.solverstate $@
