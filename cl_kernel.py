# -*- coding: utf-8 -*-  

import pyopencl as cl
import numpy, numpy.linalg as la
from time import time

#--- gpu kernel fv ---------------------------------------->>>

class CL:
  def __init__(self):
    self.ctx = cl.create_some_context()
    #plats=cl.get_platforms()
    #self.ctx = cl.Context(properties=[
              #(cl.context_properties.PLATFORM, plats[0])], devices=[
              #(cl.context_properties.PLATFORM, plats[0])], devices=None)
	       #cl.context_info.DEVICES])
    self.queue = cl.CommandQueue(self.ctx)

  def loadProgram(self, filename):
    #read in the OpenCL source file as a string
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    ############################print fstr
    #create the program
    self.program = cl.Program(self.ctx, fstr).build()

  def popCorn(self, ImageArray):
    mf = cl.mem_flags

    #initialize client side (CPU) arrays
    self.a = numpy.array(range(1024), dtype=numpy.float32)
    #self.b = numpy.array([0 for i in range(256)], dtype=numpy.float32)
    #self.a = numpy.array([ImageArray.GetValue(i) for i in range(ImageArray.GetNumberOfTuples())], dtype=numpy.float32)
    print self.a
    #print self.a.nbytes/len(self.a)
    print len(self.a)*32/4+255*32+4

    ti=time()
    #create OpenCL buffers
    self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
    #self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes/len(self.a))
    self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes)
    #self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b) #, self.a.nbytes)
    print 1,time()-ti;ti=time()
    
  def execute(self, width, height):
    #self.program.part1(self.queue, self.a.shape, None, self.a_buf, self.dest_buf, numpy.uint32(width), numpy.uint32(height))

    ti=time()
    #self.program.MMT(self.queue, (width, height), None, self.a_buf, self.dest_buf, numpy.uint32(width), numpy.uint32(height))
    #print self.a.shape[0]/2
    #self.program.PNorm2(self.queue, (self.a.shape[0]/2, ) , (4,), self.a_buf, self.dest_buf, cl.LocalMemory(4*32))
    self.program.PNorm2(self.queue, (self.a.shape[0]/2, ) , (len(self.a)/2,), self.a_buf, self.dest_buf, cl.LocalMemory(len(self.a)*32/4+255*32+4))
    #self.program.MMT(self.queue, (7, 7), None, self.a_buf, self.dest_buf, numpy.uint32(7), numpy.uint32(7))
    print 2,time()-ti;
    
    c = numpy.empty_like(self.a)
    #c=numpy.empty(1, dtype=numpy.float32)
    ti=time()
    #cl.enqueue_read_buffer(self.queue, self.dest_buf, c).wait()
    #enqueue_read_buffer turned into deprecated, the new command: 
    #(NOTE: some args' order has changed!)
    cl.enqueue_copy(self.queue, c, self.dest_buf)
    print 3,time()-ti;ti=time()
    
    #TODO Nr.2: letisztázni a cl_kernelt
    #TODO Nr.3: kell-e blokkosítani? (lokalizálni)

    #print "a", self.a
    print "c", c
    
    #f = open('results.log', 'w')
    #f.write('This file contains the pre- and postcomputed results:\n\na:\n')
    #f.write(self.a)
    #self.a.tofile(f,", ","%s")
    #f.write('\n\nc:\n')
    #f.write(c)
    #c.tofile(f,", ","%s")
    #f.close()
    
    return c

  #def teszt(self, size):
    #sumv=0
    #for i in range(size):
      #sumv+=i
    #print sumv
#--- end block: "gpu kernel fv" ---------------------------<<<