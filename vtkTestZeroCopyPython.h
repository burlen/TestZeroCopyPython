#ifndef vtkTestZeroCopyPython_h
#define vtkTestZeroCopyPython_h

#ifdef __cplusplus
extern "C" {
#endif

// wrapped python functions
void initialize();
void finalize();
void removeScalar(const char *name);
void render(const char *name);

// native python functions
PyObject *addScalar(PyObject *self, PyObject *args);
PyObject *setPoints(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif
