#!/bin/bash

arch=$(python -c 'import sys;print(sys.platform)')
sed -i '' "s|export PETSC_ARCH=.*|export PETSC_ARCH=arch-${arch}-c-opt|" ../PROTEUS.env
source ../PROTEUS.env
cd SPIDER && make clean && make -j && yes | make test
