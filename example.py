#! /usr/bin/python
# -*- coding: utf-8 -*-  

from print_cow import *
from visualisation import *
from random_sampling import *
from cl_kernel import *
from coPic import *
#from ctypes import *
import sys
import getopt

debug=False;
#debug=True;
output=False;
#output=True;
visual=False;
#visual=True;

"""def main(argv):
    grammar = "kant.xml"
    try:
        opts, args = getopt.getopt(argv, "hg:d", ["help", "grammar="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt == '-d':
            global _debug
            _debug = 1
        elif opt in ("-g", "--grammar"):
            grammar = arg

    source = "".join(args)

    k = KantGenerator(grammar, source)
    print k.output()

main(sys.argv)
"""
print_cow("Let's begin")
#--- Handle input ---------------------------------------->>>
reader=vtkPNGReader()
if (len(sys.argv) > 1):
  reader.SetFileName(sys.argv[1])
  if reader.CanReadFile(sys.argv[1]):
    print "!!! The given png file is - [Valid]\n"
  else:
    s = reader.GetFileExtensions()
    print "!!! The given png file is - [Incorrect] - Maybe it isn't a *" + s + " file or just missing?\n"
else:
  print "!!! Input png file is missing!\n"
  exit();
reader.Update()
#--- end block: "Handle input" ---------------------------<<<
if debug:
  print reader.GetOutput()  
  print reader.GetOutput().GetDimensions()

#--- Get Inputsize --------------------------------------->>>
width=reader.GetOutput().GetDimensions()[0]
height=reader.GetOutput().GetDimensions()[1]
#--- end block: "Got Inputsize" --------------------------<<<

pd=vtkPointData()
pd=reader.GetOutput().GetPointData()
p=pd.GetArray(0)

if debug:
  print "p:",
  print p
#nofp=[p.GetValue(i) for i in range(pd.GetNumberOfTuples()*3)]

#--- unif.rand.sample ------------------------------------->>>
""" TODO: percentparaméter + tupel méret ne legyen égetve! """
sampled_nofp=unif_rand_sample(0.2, pd.GetNumberOfTuples())
for i in range(len(sampled_nofp)):
  pd.GetArray(0).SetValue(sampled_nofp[i], 255)
#--- end block: "unif.rand.sample" ------------------------<<<
if debug:
  print "samplesize:",
  print len(sampled_nofp)
  
#--- intensity pic ---------------------------------------->>>
coPic(pd, debug, numpy.array([]))
#--- end block: "intensity pic" ---------------------------<<<

#--- gpu kernel fv ---------------------------------------->>>
#testKernel()

example = CL()
example.loadProgram("testKernel.cl")
example.popCorn(pd.GetArray(0))
#print example.a
c=example.execute(width, height)
#example.teszt(64)
#coPic(pd, debug, c)

if debug:
  print "pd:",
  print pd
#--- end block: "gpu kernel fv" ---------------------------<<<

#--- visualisation ---------------------------------------->>>
if visual:
  visualisate(reader.GetOutputPort())
#--- end block: "visualisation" ---------------------------<<<

#--- Handle output ---------------------------------------->>>
if output:
  writer=vtkPNGWriter()
  writer.SetInputConnection(reader.GetOutputPort())
  writer.SetFileName("[Output]"+reader.GetFileName())
  writer.Write()
  print "!!! Output was written as: [Output]"+reader.GetFileName()+"\n"
#--- end block: "handle output" ---------------------------<<<