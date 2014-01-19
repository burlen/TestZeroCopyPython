TestZeroCopyPython
==================
This is an example showing how to use zero-copy datatransfer
from a Python ndarray to vtkDataArray::SetArray w/callback.

Building
========
This project requires VTK, Python, Numpy, and SWIG.

    mkdir build
    cd build
    cmake -DVTK_DIR=/path/to/vtk /path/to/source && make

Running
=======
    PYTHONPATH=/path/to/build python /path/to/source/TestZeroCopyPython.py
