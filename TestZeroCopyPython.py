import vtkTestZeroCopyPython as testing
import numpy as np
import sys

def Initialize(nval, nit):
  """Create a dataset with a number of scalar arrays"""
  testing.initialize()

  # add points
  pts = np.random.randn(3*nval)
  testing.setPoints(pts)

  # add scalars
  dx = 2.0/float(nit-1)
  i = 0
  while i<nit:
    s = (-1.0+i*dx) * np.ones((nval,))
    testing.addScalar(s, str(i))
    i += 1
  return

def Exercise(nit):
  """Render each array then delete it"""
  i = 0
  dx = 2.0/float(nit)
  while (i<nit):
    name = str(i)
    sys.stderr.write('render %s... '%(name))
    testing.render(name)
    sys.stderr.write('removeScalar %s... '%(name))
    testing.removeScalar(name)
    i += 1

def Finalize():
  """Free the dataset"""
  sys.stderr.write('cleaning up... ')
  testing.finalize()

if __name__ == "__main__":
  nit = 50 # number of times to render
  nval = 300 # number of points

  Initialize(nval, nit)
  Exercise(nit)
  Finalize()
