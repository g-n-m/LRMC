# -*- coding: utf-8 -*-

import random

#--- unif.rand.sample ------------------------------------->>>
def unif_rand_sample(PercentOfPic, lenofp):
  """
  Egyenletes véletlen indexek előállítása mintavételezéshez
  """
  #heap=[i for i in range(lenofp)]
  heap=range(lenofp)
  nofpindex=[]
  for i in range(int(lenofp*PercentOfPic)):
    target=int(random.uniform(0,len(heap)));
    heap[target], heap[len(heap)-1]=heap[len(heap)-1], heap[target]

    """ TODO: Szépen kéne lekezelni a magasabb dimenziót! """
    target=heap.pop()
    nofpindex.append(target*3)
    nofpindex.append(target*3+1)
    nofpindex.append(target*3+2)
    
  return nofpindex
#--- end block: "unif.rand.sample" ------------------------<<<
