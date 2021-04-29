# -*- coding: utf-8 -*-
"""
FUNCTION: predict

EXPLANATION:
 Given a data set and the parameters of an SORRT with depth D = 1, the goal of 
 this program is to provide predictions and measurements of prediction accuracy. 

USAGE:  
 (pred,error,mse,R2) = predict(data,a1opt,mu1opt,a2opt,mu2opt,a3opt,mu3opt,gamma)

INPUTS:
  data: an N by (p+1) numpy matrix, where N is the number of observations and p is
        the number of predictor variables. The last column should correspond
        to the actual values of the response variable.
  a1opt, mu1opt, a2opt, m2opt, a3opt, mu3opt, gamma: parameters of an SORRT with
        depth D = 1. They are numpy matrices and vectors.

OUTPUTS:
  pred: an N numpy vector of individual predictions
  error: an N numpy vector of errors occurred when predicting
  mse: mean squared error over data
  R2: R-squared over data
  
COMMENTS:
  This program is needed for ORRT_D1.py and ORRT_D1_lambdas.py
      
AUTHOR:
  Cristina Molero-Río (mmolero@us.es)

LAST REVISION:
  April 2021
"""

import numpy as np
def predict(data,a1opt,mu1opt,a2opt,mu2opt,a3opt,mu3opt,gamma=512):
    N = len(data)
    p = len(data[0])-1 
    p1 = np.zeros(N)
    P2 = np.zeros(N)
    P3 = np.zeros(N)
    Q2 = np.zeros(N)
    Q3 = np.zeros(N)
    pred = np.zeros(N)
    error = np.zeros(N)
    calc = -(np.dot(data[:,:p],a1opt)/p-mu1opt)*gamma
    p1 = np.zeros(len(calc))
    p1[calc<600] = 1/(1+np.exp(calc[calc<600]))
    P2 = p1
    P3 = 1-p1
    Q2 = np.dot(data[:,:p],a2opt)-mu2opt
    Q3 = np.dot(data[:,:p],a3opt)-mu3opt
    pred = P2*Q2 + P3*Q3
    error = np.square(pred-data[:,p])
    mse = np.mean(error)
    R2 = 1-mse/np.mean(np.square(data[:,p]-np.mean(data[:,p])))
    return (pred,error,mse,R2)