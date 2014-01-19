project(TestZeroCopyPython)
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0003 NEW)

find_package(SWIG REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(VTK REQUIRED)

include(${SWIG_USE_FILE})
include(${VTK_USE_FILE})

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

SET(CMAKE_SWIG_FLAGS "")

set_source_files_properties(
  vtkTestZeroCopyPython.i
  PROPERTIES
  SWIG_FLAGS "-includeall"
  CPLUSPLUS OFF
  )

swig_add_module(
  vtkTestZeroCopyPython
  python
  vtkTestZeroCopyPython.i
  vtkTestZeroCopyPython.cxx
  )

swig_link_libraries(
  vtkTestZeroCopyPython
  ${PYTHON_LIBRARIES}
  ${VTK_LIBRARIES}
  )
