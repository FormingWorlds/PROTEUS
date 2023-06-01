CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
PROJECT(proteus Fortran)

include(FetchContent)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Set fortran flags, change this if you want
INCLUDE(${CMAKE_MODULE_PATH}/SetFortranFlags.cmake)

set(NetCDF_ROOT /opt/homebrew)
set (NETCDF_F90 "YES")
find_package(NetCDF REQUIRED COMPONENTS Fortran)

get_filename_component(NETCDF_LIBDIR ${NETCDF_LIBRARIES_F90} DIRECTORY)

# There is an error in CMAKE with this flag for pgf90.  Unset it
GET_FILENAME_COMPONENT(FCNAME ${CMAKE_Fortran_COMPILER} NAME)
IF(FCNAME STREQUAL "pgf90")
    UNSET(CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS)
ENDIF(FCNAME STREQUAL "pgf90")

set(SPIDER_ROOT ${CMAKE_SOURCE_DIR}/SPIDER)
set(AEOLUS_ROOT ${CMAKE_SOURCE_DIR}/AEOLUS)
set(VULCAN_ROOT ${CMAKE_SOURCE_DIR}/VULCAN)
set(PETSC_ROOT ${CMAKE_SOURCE_DIR}/petsc)

# Fetch SOCRATES
FetchContent_Declare(
  socrates
  PREFIX ${AEOLUS_ROOT}/rad_trans/socrates_code
  DOWNLOAD_DIR ${AEOLUS_ROOT}/rad_trans
  SOURCE_DIR ${AEOLUS_ROOT}/rad_trans/socrates_code
  DOWNLOAD_COMMAND
  curl -L -o ${AEOLUS_ROOT}/rad_trans/socrates_2211.tar.xz https://www.dropbox.com/sh/ixefmrbg7c94jlj/AAChibnZU9PRi8pXdVxVbdj3a/socrates_2211.tar.xz?dl=1 && tar --strip-components 1 -xvf ${AEOLUS_ROOT}/rad_trans/socrates_2211.tar.xz -C ${AEOLUS_ROOT}/rad_trans/socrates_code/ && cp -f ${AEOLUS_ROOT}/rad_trans/build_code_modified ${AEOLUS_ROOT}/rad_trans/socrates_code/build_code
)

FetchContent_MakeAvailable(socrates)
  
# Custom target for everything
add_custom_target(proteus ALL)

# PETSc
execute_process(COMMAND bash print_arch.bash
  OUTPUT_VARIABLE ARCH
OUTPUT_STRIP_TRAILING_WHITESPACE)

add_custom_target(petsc
  DEPENDS ${PETSC_ROOT}/arch-${ARCH}-c-opt/lib/libpetsc${CMAKE_SHARED_LIBRARY_SUFFIX})
# Vulcan Fastchem
add_custom_target(fastchem_vulcan
  DEPENDS ${VULCAN_ROOT}/fastchem_vulcan/fastchem)
# SPIDER
add_custom_target(spider
  DEPENDS ${SPIDER_ROOT}/spider)
# SOCRATES
INCLUDE(${CMAKE_SOURCE_DIR}/cmake_build/soc_exes.cmake)
LIST(TRANSFORM soc_exes PREPEND ${AEOLUS_ROOT}/rad_trans/socrates_code/bin/)

add_custom_target(socrates
DEPENDS ${soc_exes})

# Custom command to build petsc
add_custom_command(OUTPUT ${PETSC_ROOT}/arch-${ARCH}-c-opt/lib/libpetsc${CMAKE_SHARED_LIBRARY_SUFFIX}
  COMMAND bash ${CMAKE_SOURCE_DIR}/cmake_build/scrape_petsc_make.bash 
  WORKING_DIRECTORY ${PETSC_ROOT}
  VERBATIM)

# Custom command to build fastchem
add_custom_command(OUTPUT ${VULCAN_ROOT}/fastchem_vulcan/fastchem
  COMMAND  bash ${CMAKE_SOURCE_DIR}/cmake_build/fastchem_compiler.bash
)

# Custom command to build spider
add_custom_command(OUTPUT ${SPIDER_ROOT}/spider
  COMMAND pwd
  COMMAND bash build_spider.bash
)

#Custom command to build SOCRATES
# Need to alter Mk_cmd file so it

add_custom_command(OUTPUT ${soc_exes}
  COMMAND echo ${CMAKE_Fortran_COMPILER} ${CMAKE_Fortran_FLAGS}
  COMMAND echo "FORTCOMP = ${CMAKE_Fortran_COMPILER} -c -fcheck=all,no-recursion -fbacktrace -g -O" > make/Mk_cmd
  COMMAND echo "LINK = ${CMAKE_Fortran_COMPILER} -fcheck=all,no-recursion -fbacktrace -g -O" >> make/Mk_cmd
  COMMAND echo LIBLINK = ar rvu >> make/Mk_cmd
  COMMAND echo "INCCDF_PATH = ${NETCDF_INCLUDES}" >> make/Mk_cmd
  COMMAND echo "LIBCDF_PATH = ${NETCDF_LIBDIR}" >> make/Mk_cmd
  COMMAND echo 'LIBCDF_NAME = netcdff' >> make/Mk_cmd
  COMMAND echo 'OMPARG  = -fopenmp' >> make/Mk_cmd
  COMMAND echo 'LIBSUFFIX = a' >> make/Mk_cmd
  COMMAND echo '.SUFFIXES: $(SUFFIXES) .f90' >> make/Mk_cmd
  COMMAND bash build_code
  COMMAND type ksh >/dev/null 2>&1 ||  sed -i 's/ksh/bash/g' sbin/*
  WORKING_DIRECTORY ${AEOLUS_ROOT}/rad_trans/socrates_code)

add_dependencies(proteus fastchem_vulcan spider socrates)
add_dependencies(spider petsc)

