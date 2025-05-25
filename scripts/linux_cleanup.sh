#!/bin/bash
set -ex

docker run --rm \
    -v "$(pwd)":/nunchakukp \
    pytorch/manylinux-builder:cuda12.4 \
    bash -c "cd /nunchakukp && rm -rf *"