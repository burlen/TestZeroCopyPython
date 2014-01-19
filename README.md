TestZeroCopyPython
==================
This is an example showing how to use zero-copy datatransfer
from a Python ndarray to vtkDataArray::SetArray w/callback.

Building
========
This project requires VTK, Python, Numpy, and SWIG.

Note: VTK needs to first be patched!

    cd /path/to/VTK
    git fetch http://review.source.kitware.com/p/VTK refs/changes/72/14072/3
    git checkout FETCH_HEAD -b array-mem-management
    cd /path/to/vtk-build
    make

Once VTK is patched the example can be built:

    mkdir /path/to/example-build
    cd /path/to/example-build
    cmake -DVTK_DIR=/path/to/vtk-build /path/to/example-source
    make

Running
=======

The example can be run form its build directory:

    PYTHONPATH=/path/to/example-build python /path/to/example-source/TestZeroCopyPython.py
