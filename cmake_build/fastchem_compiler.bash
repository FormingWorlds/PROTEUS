#!/bin/bash

# Substitute instances of CC and CXX in fastchem make
make_opts="../VULCAN/fastchem_vulcan/make.global_options"
sed -i '' "s|^\s*CXX .*|CXX                  = ${CXX}|;s|^\s*CC.*|CC                   = ${CXX}|;s|^\s*SHLIBLD.*|SHLIBLD              = ${CXX}|" ${make_opts}

cd ../VULCAN/fastchem_vulcan/ && make
