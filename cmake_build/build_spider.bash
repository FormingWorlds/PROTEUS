#!/bin/bash

arch=$(uname -s | awk '{print tolower($0)}')
sed -i.bak "s|export PETSC_ARCH=.*|export PETSC_ARCH=arch-${arch}-c-opt|" ../PROTEUS.env
rm  ../PROTEUS.env.bak
source ../PROTEUS.env
cd SPIDER && make clean && make -j && yes '' | make test
