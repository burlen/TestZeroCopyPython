#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkRendererCollection.h>
#include <vtkDataObject.h>
#include <vtkColorTransferFunction.h>
#include <vtkGlyph3D.h>
#include <vtkSphereSource.h>
#include <vtkCellArray.h>
#include <vtkObjectFactory.h>
#include <vtkDataArrayDeleteCallback.h>

namespace internal
{
vtkPolyData *Data = NULL;
vtkRenderWindow *Window = NULL;
};

// --------------------------------------------------------------------------
extern "C"
void initialize()
{
  import_array();
  internal::Data = vtkPolyData::New();
  vtkRenderer *renderer = vtkRenderer::New();
  internal::Window = vtkRenderWindow::New();
  internal::Window->AddRenderer(renderer);
  renderer->Delete();
}

// --------------------------------------------------------------------------
extern "C"
void finalize()
{
  internal::Data->Delete();
  internal::Window->Delete();
}

// --------------------------------------------------------------------------
extern "C"
void removeScalar(const char *name)
{
  internal::Data->GetPointData()->RemoveArray(name);
}

// --------------------------------------------------------------------------
extern "C"
void render(const char *name)
{
  internal::Data->GetPointData()->SetActiveScalars(name);

  vtkSphereSource *ss = vtkSphereSource::New();
  ss->SetRadius(0.125);
  ss->SetThetaResolution(16);
  ss->SetPhiResolution(16);

  vtkGlyph3D *filter = vtkGlyph3D::New();
  filter->SetInputData(internal::Data);
  filter->SetSourceConnection(ss->GetOutputPort());
  ss->Delete();
  filter->SetScaleFactor(0.125);
  filter->ScalingOff();

  vtkPolyDataMapper *mapper = vtkPolyDataMapper::New();
  mapper->SetInputConnection(filter->GetOutputPort());
  filter->Delete();

  vtkColorTransferFunction *lut = vtkColorTransferFunction::New();
  lut->SetColorSpaceToRGB();
  lut->AddRGBPoint(-1.0, 0.0, 0.0, 1.0);
  lut->AddRGBPoint( 1.0, 1.0, 0.0, 0.0);
  lut->SetColorSpaceToDiverging();
  lut->Build();
  mapper->SetLookupTable(lut);
  mapper->SetScalarModeToUsePointData();
  mapper->SetScalarVisibility(1);
  mapper->SelectColorArray(name);
  mapper->SetUseLookupTableScalarRange(1);
  mapper->SetScalarMode(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA);
  lut->Delete();

  vtkActor *actor = vtkActor::New();
  actor->SetMapper(mapper);
  mapper->Delete();

  vtkRenderer *renderer = internal::Window->GetRenderers()->GetFirstRenderer();
  renderer->AddActor(actor);
  actor->Delete();

  renderer->ResetCamera();
  internal::Window->Render();

  renderer->RemoveActor(actor);
}

// --------------------------------------------------------------------------
// memory management callback
// vtkDataArray calls this when VTK no longer needs data we gave it
class vtkPyArrayDeleteCallback : public vtkDataArrayDeleteCallback
{
public:
  static vtkPyArrayDeleteCallback *New();

  // this takes a reference to the passed
  // object.
  void SetArray(PyArrayObject *array)
    {
    this->Array = array;
    Py_INCREF(this->Array);
    fprintf(stderr, "Py_INCREF(%p)\n", this->Array);
    }

  // this releases the reference.
  virtual void Invoke()
    {
    Py_DECREF(this->Array);
    fprintf(stderr, "Py_DECREF(%p)\n", this->Array);
    }
protected:
  vtkPyArrayDeleteCallback() : Array(NULL) {}
  virtual ~vtkPyArrayDeleteCallback() {}

private:
  PyArrayObject *Array;
  vtkPyArrayDeleteCallback(const vtkPyArrayDeleteCallback&);  // Not implemented.
  void operator=(const vtkPyArrayDeleteCallback&);  // Not implemented.
};
vtkStandardNewMacro(vtkPyArrayDeleteCallback);

// --------------------------------------------------------------------------
// get a pointer to numpy data
static
bool getPointer(PyObject *obj, double *&data, int &size)
{
   PyArrayObject *nda = reinterpret_cast<PyArrayObject*>(obj);
   if (!PyArray_Check(obj)
    || (PyArray_TYPE(nda) != NPY_FLOAT64)
    || !(PyArray_IS_C_CONTIGUOUS(nda) || PyArray_IS_F_CONTIGUOUS(nda))
    || ((size = PyArray_SIZE(nda)) < 1) )
    {
    return false;
    }
  // get a pointer to the data
  data = static_cast<double*>(PyArray_DATA(nda));
  return true;
}

// --------------------------------------------------------------------------
extern "C"
PyObject *setPoints(PyObject *self, PyObject *args)
{
  (void)self;
  Py_INCREF(Py_None);

  PyObject *obj = NULL;
  if ( !PyArg_ParseTuple(args, "O", &obj) )
    {
    return Py_None;
    }

  // get a pointer to the data
  double *data;
  int size = 0;
  if (!getPointer(obj, data, size))
    {
    PyErr_SetString(PyExc_RuntimeError, "failed to get a pointer to the data");
    return Py_None;
    }

  PyArrayObject *nda = reinterpret_cast<PyArrayObject*>(obj);

  // hold a reference to the numpy object while VTK is using it.
  vtkPyArrayDeleteCallback *dcb = vtkPyArrayDeleteCallback::New();
  dcb->SetArray(nda);

  // pass it to VTK with the call back that will decrement
  // the array's ref count when VTK is done with the data.
  vtkPoints *points = vtkPoints::New();
  points->SetDataType(VTK_DOUBLE);
  vtkDoubleArray *pts = vtkDoubleArray::SafeDownCast(points->GetData());
  pts->SetArray(data, size, dcb);
  dcb->Delete();
  internal::Data->SetPoints(points);
  points->Delete();

  int nCells = size/3;
  vtkCellArray *cells = vtkCellArray::New();
  vtkIdType *pCells = cells->WritePointer(nCells, 2*nCells);
  for (int i=0; i<nCells; ++i)
    {
    int ii = 2*i;
    pCells[ii] = 1;
    pCells[ii+1] = i;
    }
  internal::Data->SetVerts(cells);
  cells->Delete();

  return Py_None;
}

// --------------------------------------------------------------------------
extern "C"
PyObject *addScalar(PyObject *self, PyObject *args)
{
  (void)self;
  Py_INCREF(Py_None);

  PyObject *obj = NULL;
  const char *name = NULL;
  if ( !PyArg_ParseTuple(args, "Os", &obj, &name) )
    {
    return Py_None;
    }

  // get a pointer to the data
  double *data;
  int size = 0;
  if (!getPointer(obj, data, size))
    {
    PyErr_SetString(PyExc_RuntimeError, "failed to get a pointer to the data");
    return Py_None;
    }

  PyArrayObject *nda = reinterpret_cast<PyArrayObject*>(obj);

  // hold a reference to the numpy object while VTK is using it.
  vtkPyArrayDeleteCallback *dcb = vtkPyArrayDeleteCallback::New();
  dcb->SetArray(nda);

  // pass it to VTK with the call back that will decrement
  // the array's ref count when VTK is done with the data.
  vtkDoubleArray *array = vtkDoubleArray::New();
  array->SetName(name);
  array->SetArray(data, size, dcb);
  dcb->Delete();
  internal::Data->GetPointData()->AddArray(array);
  array->Delete();

  return Py_None;
}
