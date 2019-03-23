#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=msra_resnet_34/resnet_34_solver.prototxt $@
