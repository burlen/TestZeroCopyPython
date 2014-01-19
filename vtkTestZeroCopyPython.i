%module vtkTestZeroCopyPython

%{
#include "vtkTestZeroCopyPython.h"
/*
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_6_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
*/
%}

%init %{
/*import_array();*/
%}

%inline %{
void initialize();
void finalize();
void removeScalar(const char *name);
void render(const char *name);
%}

%native(setPoints)
PyObject *setPoints(PyObject* self, PyObject* args);

%native(addScalar)
PyObject *addScalar(PyObject* self, PyObject* args);
