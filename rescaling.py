# -*- coding: utf-8 -*-
"""
FUNCTION: rescaling

EXPLANATION:
 The goal of this program is to scale each of the predictor variables of a 
 data set with the scaling parameters obtained after running the function
 scaling. See scaling.py.

USAGE:  rescale = rescaling(data,M,m)

INPUTS:
  data: an N by (p+1) numpy matrix, where N is the number of observations and p is
        the number of predictor variables. The last column should correspond
        to the response variable.
  M: a p numpy vector with the upper scaling parameters
  m: a p numpy vector with the lower scaling parameters

OUTPUTS:
  rescale: an N by (p+1) numpy matrix after scaling 

COMMENTS:
  This program is needed for ORRT_D1.py and ORRT_D1_lambdas.py
      
AUTHOR:
  Cristina Molero-Río (mmolero@us.es)

LAST REVISION:
  April 2021
"""

import numpy as np
def rescaling(data, M, m):
    N = len(data)
    p = len(data[0])-1           
    rescale = np.zeros((N,int(p+1))) 
    rescale = (data - m)/(M-m)        
    rescale[:,p] = data[:,p]
    return rescale