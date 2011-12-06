# -*- coding: utf-8 -*-

import numpy, numpy.linalg as la

#--- SMV result checker ----------------------------------->>>

def check_SMV(s, m, v, print_it):
    m_ = m.reshape(v.size, m.size/v.size)
    v_ =  v.reshape(v.size, 1)

    if print_it:
        print "[m:]"+v.size*5*"-"; print m_
        print "[v:]"+v.size*5*"-"; print v
    
        print "[R:]"+v.size*8*"-"
    
    return numpy.dot(m_,v)

#--- end block: "SMV result checker" ----------------------<<<

#--- OPG result checker ----------------------------------->>>

def check_OPG(v, w, print_it):
    if print_it:
        print "[v:]"+v.size*5*"-"; print v
        print "[w:]"+w.size*5*"-"; print w

        print "[R:]"+v.size*8*"-"
        
    return numpy.outer(v,w)

#--- end block: "OPG result checker" ----------------------<<<
