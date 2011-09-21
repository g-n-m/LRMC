# -*- coding: utf-8 -*-  

import pyopencl as cl
import numpy, numpy.linalg as la
from time import time

#--- gpu kernel fv ---------------------------------------->>>

class CL:
  def __init__(self):
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)

  def loadProgram(self, filename):
    #read in the OpenCL source file as a string
    f = open(filename, 'r')
    fstr = "".join(f.readlines())

    #create the program
    self.program = cl.Program(self.ctx, fstr).build()

  def popCorn(self, ImageArray):
    mf = cl.mem_flags

    #--- Original
    """
    #initialize client side (CPU) arrays
    self.a = numpy.array([ImageArray.GetValue(i) for i in range(ImageArray.GetNumberOfTuples())], dtype=numpy.float32)
    #"""
    
    #--- MMT Kernel
    """
    #initialize client side (CPU) arrays
    self.a = numpy.array(range(64), dtype=numpy.float32)
    
    #create OpenCL buffers
    self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes)
    #"""
    
    #--- PNorm2 Kernel
    """
    #initialize client side (CPU) arrays
    self.a = numpy.array(range(512), dtype=numpy.float32)
    
    #create OpenCL buffers
    self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes/len(self.a))
    #"""
    
    #--- PNorm2v2 Kernel
    #"""
    #initialize client side (CPU) arrays
    self.a = numpy.array(range(512), dtype=numpy.float32)
    
    #create OpenCL buffers
    self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.a.nbytes)
    #"""
    
    self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
    
    #print "max empirical loc. mem on GTX 280:", 1024*32/4+255*32+4
    #print self.a.nbytes/len(self.a)

  def execute(self, width, height):
    #--- Original
    """
    self.program.part1(self.queue, self.a.shape, None, self.a_buf, self.dest_buf, numpy.uint32(width), numpy.uint32(height))    
    #"""
    
    #--- MMT Kernel
    """
    self.program.MMT(self.queue, (8, 8), None, self.a_buf, self.dest_buf, numpy.uint32(8), numpy.uint32(8))
    self.program.MMT(self.queue, (8, 8), (4,4), self.a_buf, self.dest_buf, numpy.uint32(8), numpy.uint32(8), cl.LocalMemory(len(self.a)*32/4), cl.LocalMemory(len(self.a)*32/4))
    c = numpy.empty_like(self.a)
    #"""
    
    #--- PNorm2 Kernel    
    """
    self.program.PNorm2G(self.queue, (self.a.shape[0]/2, ) , (len(self.a)/2,), self.a_buf, self.dest_buf, cl.LocalMemory(len(self.a)*32/4+255*32+4))
    c=numpy.empty(1, dtype=numpy.float32)
    #"""
    
    #--- PNorm2v2 Kernel    
    #"""
    self.program.PNorm2v2(self.queue, (self.a.shape[0]/2, ) , (len(self.a)/2,), self.a_buf, self.dest_buf, cl.LocalMemory(len(self.a)*32/4),cl.LocalMemory(32))
    c = numpy.empty_like(self.a)
    #"""
    
    #ti=time()
      #cl.enqueue_read_buffer(self.queue, self.dest_buf, c).wait()
      #enqueue_read_buffer turned into deprecated, the new command: 
      #(NOTE: some args' order has changed!)
    cl.enqueue_copy(self.queue, c, self.dest_buf)
    #print "runtime",time()-ti;ti=time()
    
    #TODO Nr.3: blokkosítás (lokalizálni)

    #print "[c:]"+8*5*"-"; print self.a.reshape(8,8)
    print "[c:]"+8*8*"-"; print c.reshape(8,8)
    
    return c

#--- end block: "gpu kernel fv" ---------------------------<<<