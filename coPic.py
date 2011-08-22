# -*- coding: utf-8 -*-  

from vtk import vtkUnsignedCharArray
import numpy

#--- intensity pic ---------------------------------------->>>
def toGrayscale(pd, a):
  """
  Creating grayscale image
  """
  for i in range(pd.GetNumberOfTuples()):
    a.InsertNextValue(
      (int)(.2989*pd.GetArray(0).GetValue(i*3)+.5870*pd.GetArray(0).GetValue(i*3+1)+.1140*pd.GetArray(0).GetValue(i*3+2))
    )
    
  return a

def coPic(pd, debug, c):
  """
  Insert PNGImage array into an Image
  """
  a=vtkUnsignedCharArray();
  
  if c.any():
    for i in range(c.size):
      a.InsertNextValue( (int) (c[i]) )
  else:
    a=toGrayscale(pd, a)
 
  a.SetName("PNGImage")
  pd.AddArray(a)
  
  if debug:
    print "a:",
    print a
#--- end block: "intensity pic" ---------------------------<<<