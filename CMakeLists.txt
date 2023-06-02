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
add_custom_target(proteus ALL
  DEPENDS
${CMAKE_SOURCE_DIR}/cmake_build/proteus_pkg.txt)


execute_process(COMMAND bash ${CMAKE_SOURCE_DIR}/cmake_build/print_arch.bash
  OUTPUT_VARIABLE ARCH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# PETSc
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
configure_file(${CMAKE_SOURCE_DIR}/cmake_build/petsc_build.bash.in ${CMAKE_SOURCE_DIR}/cmake_build/petsc_build.bash @ONLY)
add_custom_command(OUTPUT ${PETSC_ROOT}/arch-${ARCH}-c-opt/lib/libpetsc${CMAKE_SHARED_LIBRARY_SUFFIX}
  COMMAND bash ${CMAKE_SOURCE_DIR}/cmake_build/petsc_build.bash
  WORKING_DIRECTORY ${PETSC_ROOT}
  VERBATIM)

set(make_opts ${VULCAN_ROOT}/fastchem_vulcan/make.global_options)
# Custom command to build fastchem
configure_file(${CMAKE_SOURCE_DIR}/cmake_build/fastchem_build.bash.in ${CMAKE_SOURCE_DIR}/cmake_build/fastchem_build.bash @ONLY)
add_custom_command(OUTPUT ${VULCAN_ROOT}/fastchem_vulcan/fastchem
  COMMAND bash ${CMAKE_SOURCE_DIR}/cmake_build/fastchem_build.bash
  WORKING_DIRECTORY ${VULCAN_ROOT}/fastchem_vulcan
  VERBATIM
)

set(env_file ${CMAKE_SOURCE_DIR}/PROTEUS.env)
# Custom command to build spider
configure_file(${CMAKE_SOURCE_DIR}/cmake_build/build_spider.bash.in ${CMAKE_SOURCE_DIR}/cmake_build/build_spider.bash @ONLY)
add_custom_command(OUTPUT ${SPIDER_ROOT}/spider
  COMMAND bash ${CMAKE_SOURCE_DIR}/cmake_build/build_spider.bash
  WORKING_DIRECTORY ${SPIDER_ROOT}
  VERBATIM
)

# Custom command to build SOCRATES
# Need to alter Mk_cmd file so it
configure_file(${CMAKE_SOURCE_DIR}/cmake_build/socrates_build.bash.in ${CMAKE_SOURCE_DIR}/cmake_build/socrates_build.bash @ONLY)
add_custom_command(OUTPUT ${soc_exes}
  COMMAND bash ${CMAKE_SOURCE_DIR}/cmake_build/socrates_build.bash
  WORKING_DIRECTORY ${AEOLUS_ROOT}/rad_trans/socrates_code
  VERBATIM
  )


add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/cmake_build/proteus_pkg.txt
COMMAND echo "Installation complete" > ${CMAKE_SOURCE_DIR}/cmake_build/proteus_pkg.txt)
add_dependencies(proteus fastchem_vulcan spider socrates)
add_dependencies(spider petsc)

install(FILES ${CMAKE_SOURCE_DIR}/proteus_pkg.txt DESTINATION proteus)
