# -*- coding: utf-8 -*-
"""
FUNCTION: ORRT_D1

EXPLANATION:
 The goal of this program is to read a training and a test subset, solve 
 Problem (3)-(5) with lambda^L = lambda^G = 0 on the training data set after 
 scaling and make predictions on the test data set after rescaling.

USAGE:  (a1opt,mu1opt,a2opt,mu2opt,a3opt,mu3opt,gamma,
         predtrain,errortrain,msetrain,R2train,
         predtest,errortest,msetest,R2test) = ORRT_D1(train,test,S=1000,gamma=512)

INPUTS:
  train: an N x (p+1) matrix, where N is the number of observations and p is
        the number of predictor variables. The last column should correspond
        to the response variable.
  test: an n_test x (p+1) matrix, where n_test is the number of observations 
        and p is the number of predictor variables. The last column should 
        correspond to the response variable.
  S: number of random initial solutions. The default value is S = 1000.
  gamma: the parameter of the logistic CDF. The default value is gamma = 512.

OUTPUTS:
  a1opt, mu1opt, a2opt, m2opt, a3opt, mu3opt, gamma: parameters of an SORRT with depth D = 1.
  predtrain,errortrain,msetrain,R2train: outputs obtained in predict.py for the training data set.
  predtest,errortest,msetest,R2test: outputs obtained in predict.py for the test data set.
  

COMMENTS:
  This is the main program for lambda^L = lambda^G = 0.   
      
AUTHOR:
  Cristina Molero-Río (mmolero@us.es)

LAST REVISION:
  April 2021
"""

import numpy as np
from scaling import scaling
from rescaling import rescaling
from predict import predict
from scipy.optimize import minimize
from scipy.optimize import Bounds
from joblib import Parallel, delayed 
def ORRT_D1(train,test,S=1000,gamma=512):
    
    # Scale the training subset
    (train, Mtree, mtree) = scaling(train)
    N = len(train)
    p = int(len(train[0])-1)

    # Rescale the test subset
    test = rescaling(test,Mtree,mtree)    

    # Definition of the objective function
    def f(x,train,gamma,p): 
        calc = -(np.dot(train[:,:p],x[0:p])/p-x[p])*gamma
        P2 = np.zeros(len(calc))
        P2[calc<600] = 1/(1+np.exp(calc[calc<600]))
        P3 = 1-P2   
        Q2 = np.dot(train[:,:p],x[(p+1):(2*p+1)])-x[(3*p+1)]
        Q3 = np.dot(train[:,:p],x[(2*p+1):(3*p+1)])-x[(3*p+2)]
        errorp = np.square(P2*Q2 + P3*Q3 - train[:,p])
        meanerrorp = np.mean(errorp)
        if np.isnan(meanerrorp):
            exit()
        return meanerrorp 
   
    # Definition of bounds
    def lulb(p):
        lu= np.append(-np.ones(p+1),np.repeat(-50,2*p+2))
        lb= np.append(np.ones(p+1),np.repeat(50,2*p+2))
        return (lu,lb)
    d = lulb(p)
    bounds = Bounds(d[0],d[1])  

    # Definition of gradient
    def gradient(x,train,gamma,p):
        calc = -(np.dot(train[:,:p],x[0:p])/p-x[p])*gamma
        p1 = np.zeros(len(calc))
        p1[calc<600] = 1/(1+np.exp(calc[calc<600]))
        exponencial1 = np.exp(600)*np.ones(len(calc))
        exponencial1[calc<600] = np.exp(calc[calc<600])
        P2 = p1
        P3 = 1-p1
        Q2 = np.dot(train[:,:p],x[(p+1):(2*p+1)])-x[(3*p+1)]
        Q3 = np.dot(train[:,:p],x[(2*p+1):(3*p+1)])-x[(3*p+2)]
        g = P2*Q2 + P3*Q3 - train[:,p]
        der = np.zeros_like(x)
        m1 = 2*g*exponencial1*np.square(p1)*(Q2-Q3)
        der[0:p]= gamma/p*np.mean(np.transpose(train[:,:p])*m1,axis=1)
        der[p] = -gamma*np.mean(m1)
        der[(p+1):(2*p+1)] = np.mean(2*g*np.transpose(train[:,:p])*P2,axis=1)
        der[(2*p+1):(3*p+1)] = np.mean(2*g*np.transpose(train[:,:p])*P3,axis=1)
        der[(3*p+1)] = -np.mean(2*g*P2)
        der[(3*p+2)] = -np.mean(2*g*P3)
        return der 
    
    # Set the grid of S random initial solutions
    np.random.seed(1)
    a1i = np.zeros((S,p))
    mu1i = np.zeros(S)
    nn = 0
    while (nn < S):
        a1i[nn,:] = 2*np.random.random(p)-1
        mu1i[nn] = 2*np.random.random(1)-1
        vale = True
        ii =  0
        while (vale and ii < N):
            if (np.isinf(np.exp(-(np.sum(a1i[nn,:]*train[ii,:p])/p-mu1i[nn])*gamma))):
                vale = False
            else: 
                ii = ii + 1
        if vale:
            nn = nn + 1
    x0 = np.zeros((S,3*p+3))
    x0[:,0:p] = a1i
    x0[:,p] = mu1i
    x0[:,(p+1):(2*p+1)] = np.random.random((S,p))
    x0[:,(3*p+1)] = np.random.random(S)
    x0[:,(2*p+1):(3*p+1)] = np.random.random((S,p))
    x0[:,(3*p+2)] = np.random.random(S)

    # Define the function to be parallelized
    def funcion(valores):
        [f,x0nn,train,gamma,p,gradient,bounds] = valores
        try:
            res = minimize(f, x0nn, args=(train,gamma,p), method = 'SLSQP', jac=gradient,
                           options={'ftol': 1e-5, 'disp': False,'maxiter':300},bounds=bounds)
            objetivo = res.fun
            sol = res.x
        except:
            objetivo = 1e+300
            sol = np.zeros(p)
        return (objetivo,sol)    
    values = [([f,x0[nn],train,gamma,p,gradient,bounds]) for nn in range(S)]  
    
    # Solve Problem (1) for the S initial solutions
    results = Parallel(n_jobs=8)(delayed(funcion)(value) for value in values)
    
    # Obtain the best solution
    objetivo = [results[i][0] for i in range(S)]
    indexopt = np.nanargmin(objetivo)
    xopt = results[indexopt][1]
    
    # Obtain the parameters of the SORRT with depth D = 1
    a1opt = xopt[0:p]
    mu1opt = xopt[p]
    a2opt = xopt[(p+1):(2*p+1)] 
    a3opt = xopt[(2*p+1):(3*p+1)] 
    mu2opt = xopt[(3*p+1)] 
    mu3opt = xopt[(3*p+2)] 
    
    # Performance over the training and test subsets
    (predtrain,errortrain,msetrain,R2train) = predict(train,a1opt,mu1opt,a2opt,mu2opt,a3opt,mu3opt,gamma)
    (predtest,errortest,msetest,R2test) = predict(test,a1opt,mu1opt,a2opt,mu2opt,a3opt,mu3opt,gamma)
    
    return (a1opt,mu1opt,a2opt,mu2opt,a3opt,mu3opt,gamma,
            predtrain,errortrain,msetrain,R2train,
            predtest,errortest,msetest,R2test)