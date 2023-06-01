#!/bin/bash

build=$(pwd)

petsc_dir=${build}/../petsc
./configure --with-debugging=0 --with-fc=0 --with-cxx=0 --download-sundials2 --download-mpich --COPTFLAGS="-O3" --CXXOPTFLAGS="-O3" -CC=${CC}

arch=arch-$(uname -s | awk '{print tolower($0)}')-c-opt

make PETSC_DIR=${petsc_dir} PETSC_ARCH=${arch} all
make PETSC_DIR=${petsc_dir} PETSC_ARCH=${arch} check
