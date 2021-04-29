# -*- coding: utf-8 -*-
"""
FUNCTION: scaling

EXPLANATION:
 Given a data set, the goal of this program is to scale each of its predictor
 variables to the 0-1 interval.

USAGE:  (scale,M,m) = scaling(data)

INPUTS:
  data: an N x (p+1) numpy matrix, where N is the number of observations and p is
        the number of predictor variables. The last column should correspond
        to the response variable.

OUTPUTS:
  scale: an N by (p+1) numpy matrix after scaling 
  M: a p numpy vector with the upper scaling parameters
  m: a p numpy vector with the lower scaling parameters

COMMENTS:
  This program is needed for ORRT_D1.py and ORRT_D1_lambdas.py
      
AUTHOR:
  Cristina Molero-Río (mmolero@us.es)

LAST REVISION:
  April 2021
"""

import numpy as np
def scaling(data):
    N = len(data)
    p = len(data[0])-1
    scale = np.zeros((N,int(p+1)))

    M = np.zeros(p)
    m = np.zeros(p)
    
    m = np.min(data, axis=0)
    M = np.max(data, axis=0)

    M = np.where(m==M,m+1,M)
    
    scale = (data - m)/(M-m)
    
    scale[:,p] = data[:,p]
    
    return (scale, M, m)