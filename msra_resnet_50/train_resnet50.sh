#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=msra_resnet_50/resnet_50_solver.prototxt $@
