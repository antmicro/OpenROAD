# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2019-2025, The OpenROAD Authors

include("openroad")

project(cut)
add_library(cut)

messages(
TARGET cut
)

target_compile_options(cut PUBLIC -Wno-unused-variable)
target_compile_options(cut PUBLIC -Wno-unused-function)

target_include_directories(cut
  PUBLIC
    ../include
)

target_sources(cut
  PRIVATE
    blifParser.cpp
    blif.cpp
    abc_library_factory.cpp
    logic_extractor.cpp
    logic_cut.cpp
)

target_link_libraries(cut
  PUBLIC
    mockturtle
    rsz_lib
    ${ABC_LIBRARY}
  PRIVATE
    odb
    dbSta
    OpenSTA
    dbSta_lib
    utl_lib
 )

foreach (WARNING template-body)
  check_cxx_compiler_flag("-Wno-${WARNING}" HAS_WNO_${WARNING})
  if (HAS_WNO_${WARNING})
    target_compile_options(cut PUBLIC -Wno-${WARNING})
  endif()
endforeach()
