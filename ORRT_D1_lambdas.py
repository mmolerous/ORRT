# -*- coding: utf-8 -*-
"""
FUNCTION: ORRT_D1_lambdas

EXPLANATION:
 The goal of this program is to read a training and a test subset, solve 
 Problem (3)-(5) for a wide grid of values (lambda^L,lambda^G) on the training
 subset after scaling and make predictions on the test subset after
 rescaling. Let nlambdasL and nlambdasG the number of lambda^L and lambda^G 
 tested, respectively.

USAGE:  (a1opt,mu1opt,a2opt,mu2opt,a3opt,mu3opt,betaopt,gamma,
         predtrain,errortrain,msetrain,R2train,
         predtest,errortest,msetest,R2test,
         localsparsity,globalsparsity) = ORRT_D1_lambdas(train,test,S=1000,gamma=512)

INPUTS:
  train: an N x (p+1) numpy matrix, where N is the number of observations and p is
        the number of predictor variables. The last column should correspond
        to the response variable.
  test: an n_test x (p+1) numpy matrix, where n_test is the number of observations 
        and p is the number of predictor variables. The last column should 
        correspond to the response variable.
  S: number of random initial solutions. The default value is S = 1000.
  gamma: the parameter of the logistic CDF. The default value is gamma = 512.

OUTPUTS:
  a1opt, mu1opt, a2opt, m2opt, a3opt, mu3opt, betaopt, gamma: 
      parameters of SORRT with depth D = 1 for a grid of values (lambda^L,lambda^G).
      - a1opt, a2opt, a3opt, betaopt are p x nlambdasL x nlambdasG numpy matrices
      - mu1opt, mu2opt, mu3opt are nlambdasL x nlambdasG numpy matrices
  predtrain,errortrain,msetrain,R2train: outputs obtained in predict.py for the
      training subset.
      - predtrain and errortrain are N x nlambdasL x nlambdasG numpy matrices
      - msetrain and R2train are nlambdasL x nlambdasG numpy matrices
  predtest,errortest,msetest,R2test: outputs obtained in predict.py for the test subset.
      - predtest and errortest are n_test x nlambdasL x nlambdasG numpy matrices
      - msetest and R2test are nlambdas x nlambdas numpy matrices
  localsparsity: an nlambdasL x nlambdasG numpy matrix with the percentage of 
      predictor variables not used per node
  globalsparsity: an nlambdasL x nlambdasG numpy matrix with the percentage of 
      predictor variables not used across the whole tree

COMMENTS:
  This is the main program for a grid of values (lambda^L,lambda^G).    
      
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

def ORRT_D1_lambdas(train,test,S=1000,gamma=512):

    # Scale the training subset
    (train, Mtree, mtree) = scaling(train)
    n_train = len(train)
    p = int(len(train[0])-1)
    
    # Rescale the test subset
    test = rescaling(test,Mtree,mtree)   
    n_test = len(test)
    
    # Set the grid of values of lambda^L and lambda^G
    lambdasL = np.append(0,np.exp2(list(range(-6,6))))/(3*p)
    nlambdasL = len(lambdasL)
    lambdasG = np.append(0,np.exp2(list(range(-6,6))))/p    
    nlambdasG = len(lambdasG)
    
    # Definition of the objective function
    def f(x,train,gamma,p,lambdaL,lambdaG):
        calc = -(np.dot(train[:,:p],x[0:p]-x[(4*p+3):(5*p+3)])/p-x[p])*gamma
        P2 = np.zeros(len(calc))
        P2[calc<600] = 1/(1+np.exp(calc[calc<600]))
        P3 = 1-P2   
        Q2 = np.dot(train[:,:p],x[(p+1):(2*p+1)]-x[(5*p+3):(6*p+3)])-x[(3*p+1)]
        Q3 = np.dot(train[:,:p],x[(2*p+1):(3*p+1)]-x[(6*p+3):(7*p+3)])-x[(3*p+2)]
        errorp = np.square(P2*Q2 + P3*Q3 - train[:,p])
        meanerrorp = np.mean(errorp)
        meanerrorplasso = meanerrorp + lambdaG*np.sum(x[(3*p+3):(4*p+3)]) + lambdaL*(np.sum(x[0:p])+np.sum(x[(p+1):(3*p+1)])+np.sum(x[(4*p+3):(7*p+3)]))
        if np.isnan(meanerrorplasso):
            exit()
        return meanerrorplasso
        
    # Definition of bounds
    def lulb(p):
        lu= np.concatenate((np.zeros(p),-np.ones(1),np.repeat(0,2*p),np.repeat(-50,2),np.repeat(0,p),np.repeat(0,3*p)))
        lb= np.concatenate((np.ones(p+1),np.repeat(50,2*p+2),np.repeat(np.inf,p),np.repeat(1,p),np.repeat(50,2*p)))
        return (lu,lb)
    d = lulb(p)
    bounds = Bounds(d[0],d[1]) 
     
    # Definition of gradient
    def gradient(x,train,gamma,p,lambdaL,lambdaG):
        calc = -(np.dot(train[:,:p],x[0:p]-x[(4*p+3):(5*p+3)])/p-x[p])*gamma
        p1 = np.zeros(len(calc))
        p1[calc<600] = 1/(1+np.exp(calc[calc<600]))
        exponencial1 = np.exp(600)*np.ones(len(calc))
        exponencial1[calc<600] = np.exp(calc[calc<600])
        P2 = p1
        P3 = 1-p1
        Q2 = np.dot(train[:,:p],x[(p+1):(2*p+1)]-x[(5*p+3):(6*p+3)])-x[(3*p+1)]
        Q3 = np.dot(train[:,:p],x[(2*p+1):(3*p+1)]-x[(6*p+3):(7*p+3)])-x[(3*p+2)]
        g = P2*Q2 + P3*Q3 - train[:,p]
        der = np.zeros_like(x)
        m1 = 2*g*exponencial1*np.square(p1)*(Q2-Q3)
        der[0:p]= gamma/p*np.mean(np.transpose(train[:,:p])*m1,axis=1) + np.repeat(lambdaL,p)
        der[p] = -gamma*np.mean(m1)
        der[(p+1):(2*p+1)] = np.mean(2*g*np.transpose(train[:,:p])*P2,axis=1) + np.repeat(lambdaL,p)
        der[(2*p+1):(3*p+1)] = np.mean(2*g*np.transpose(train[:,:p])*P3,axis=1) + np.repeat(lambdaL,p)
        der[(3*p+1)] = -np.mean(2*g*P2)
        der[(3*p+2)] = -np.mean(2*g*P3)
        der[(3*p+3):(4*p+3)] = np.repeat(lambdaG,p)
        der[(4*p+3):(5*p+3)] = -gamma/p*np.mean(np.transpose(train[:,:p])*m1,axis=1) + np.repeat(lambdaL,p)
        der[(5*p+3):(6*p+3)] = -np.mean(2*g*np.transpose(train[:,:p])*P2,axis=1) + np.repeat(lambdaL,p)
        der[(6*p+3):(7*p+3)] = -np.mean(2*g*np.transpose(train[:,:p])*P3,axis=1) + np.repeat(lambdaL,p)
        return der

    # Definition of constraints and jacobian
    jacons = np.zeros((3*p,7*p+3))
    jacons[0:p,0:p] = -np.eye(p)
    jacons[0:p,(4*p+3):(5*p+3)] = -np.eye(p)
    jacons[0:p,(3*p+3):(4*p+3)] = np.eye(p)
    jacons[p:2*p,(p+1):(2*p+1)] = -np.eye(p)
    jacons[p:2*p,(5*p+3):(6*p+3)] = -np.eye(p)
    jacons[p:2*p,(3*p+3):(4*p+3)] =  np.eye(p)
    jacons[2*p:3*p,(2*p+1):(3*p+1)] = -np.eye(p)
    jacons[2*p:3*p,(6*p+3):(7*p+3)] = -np.eye(p)
    jacons[2*p:3*p,(3*p+3):(4*p+3)] = np.eye(p)    
    lambdaL = 0
    lambdaG = 0
    ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.concatenate((x[(3*p+3):(4*p+3)]-x[0:p]-x[(4*p+3):(5*p+3)],
                                               x[(3*p+3):(4*p+3)]-x[(p+1):(2*p+1)]-x[(5*p+3):(6*p+3)],
                                               x[(3*p+3):(4*p+3)]-x[(2*p+1):(3*p+1)]-x[(6*p+3):(7*p+3)])),
             'jac' : lambda x: jacons,
             'arg' : (train,gamma,p,lambdaL,lambdaG)}

    # Set the grid of S random initial solutions
    np.random.seed(1)
    x0 = np.zeros((S,7*p+3))
    x0[:,p] = 2*np.random.random(S)-1
    a1iaux = -700/gamma + x0[:,p]
    a1iaux2 = np.maximum(a1iaux,-1)
    a1iaux3 = np.transpose(np.tile(a1iaux2,(p,1)))
    a1iaux4 = (1-a1iaux3)*np.random.random((S,p))+a1iaux3
    x0[:,0:p] = np.maximum(a1iaux4,0)
    x0[:,(p+1):(2*p+1)] = np.random.random((S,p))
    x0[:,(2*p+1):(3*p+1)] = np.random.random((S,p))
    x0[:,(3*p+1)] = np.random.random(S)
    x0[:,(3*p+2)] = np.random.random(S)   
    x0[:,(3*p+3):(4*p+3)] = np.random.random((S,p))
    x0[:,(4*p+3):(5*p+3)] = np.maximum(-a1iaux4,0)
    x0[:,(5*p+3):(6*p+3)] = np.random.random((S,p))
    x0[:,(6*p+3):(7*p+3)] = np.random.random((S,p))

    # Define the function to be parallelized
    def funcion(valores):
        [f,x0nn,nn,train,gamma,p,lambdasG,lambdasL,gradient,bounds,ineq_cons] = valores
        nlambdasG = len(lambdasG)
        nlambdasL = len(lambdasL)
        objetivo = 1000000*np.ones((nlambdasL,nlambdasG))
        sol = np.zeros((nlambdasL,nlambdasG,7*p+3))
        for ll in range(nlambdasL):
            for gg in range(nlambdasG):
                try:
                    print(ll,gg,nn)
                    print('local',ll)
                    print('global',gg)                   
                    res = minimize(f, x0nn, args=(train,gamma,p,lambdasL[ll],lambdasG[gg]),
                                   method='SLSQP',jac=gradient,
                                   options={'ftol': 1e-5, 'disp': False,'maxiter':300},
                                   bounds=bounds, constraints=ineq_cons)
                    objetivo[ll,gg] = res.fun
                    sol[ll,gg,:] = res.x
                    x0nn[0:p] = res.x[0:p]
                    x0nn[p] = res.x[p]
                    x0nn[(p+1):(2*p+1)] = res.x[(p+1):(2*p+1)]
                    x0nn[(2*p+1):(3*p+1)] = res.x[(2*p+1):(3*p+1)]
                    x0nn[(3*p+1)] = res.x[(3*p+1)]
                    x0nn[(3*p+2)] = res.x[(3*p+2)]
                    x0nn[(3*p+3):(4*p+3)] = res.x[(3*p+3):(4*p+3)]
                    x0nn[(4*p+3):(5*p+3)] = res.x[(4*p+3):(5*p+3)]
                    x0nn[(5*p+3):(6*p+3)] = res.x[(5*p+3):(6*p+3)]
                    x0nn[(6*p+3):(7*p+3)] = res.x[(6*p+3):(7*p+3)]
                except:
                    pass
        return (objetivo,sol)
    values = [([f,x0[nn],nn,train,gamma,p,lambdasL,lambdasG,gradient,bounds,ineq_cons]) for nn in range(S)]

    # Solve Problem (1) for a grid of lambda^L and lambda^G
    results = Parallel(n_jobs=8)(delayed(funcion)(value) for value in values)
    objetivos = [results[i][0] for i in range(S)]
    xs = [results[i][1] for i in range(S)]
    
    # Obtain the parameters of the SORRT with depth D = 1 for the grid of
    # values of lambda^L and lambda^G, as well as the performance over
    # the training and test subsets.
    objetivopt = np.zeros((nlambdasL,nlambdasG))
    indexopt = np.zeros((nlambdasL,nlambdasG),dtype=int)
    xopt=np.zeros((nlambdasL,nlambdasG,7*p+3))
    a1opt = np.zeros((p,nlambdasL,nlambdasG))
    a2opt = np.zeros((p,nlambdasL,nlambdasG))
    a3opt = np.zeros((p,nlambdasL,nlambdasG))
    betaopt = np.zeros((p,nlambdasL,nlambdasG))
    mu1opt = np.zeros((nlambdasL,nlambdasG))
    mu2opt = np.zeros((nlambdasL,nlambdasG))
    mu3opt = np.zeros((nlambdasL,nlambdasG))
    predtrain = np.zeros((n_train,nlambdasL,nlambdasG))
    errortrain = np.zeros((n_train,nlambdasL,nlambdasG))
    msetrain = np.zeros((nlambdasL,nlambdasG)) 
    R2train = np.zeros((nlambdasL,nlambdasG))
    predtest = np.zeros((n_test,nlambdasL,nlambdasG))
    errortest = np.zeros((n_test,nlambdasL,nlambdasG))   
    msetest = np.zeros((nlambdasL,nlambdasG))    
    R2test = np.zeros((nlambdasL,nlambdasG))
    coefsnonulos = np.zeros((nlambdasL,nlambdasG))
    numberofeatures = np.zeros((nlambdasL,nlambdasG))
    localsparsity = np.zeros((nlambdasL,nlambdasG)) 
    globalsparsity = np.zeros((nlambdasL,nlambdasG))
    for ll in range(nlambdasL):
        for gg in range(nlambdasG):
            obj = [objetivos[i][ll,gg] for i in range(S)]
            objetivopt[ll,gg] = np.min(obj)
            indexopt[ll,gg] = np.nanargmin(obj)
            xopt[ll,gg,:] = xs[indexopt[ll,gg]][ll,gg]
            a1opt[:,ll,gg] = xopt[ll,gg,0:p] - xopt[ll,gg,(4*p+3):(5*p+3)]
            a2opt[:,ll,gg] = xopt[ll,gg,(p+1):(2*p+1)] - xopt[ll,gg,(5*p+3):(6*p+3)]
            a3opt[:,ll,gg] = xopt[ll,gg,(2*p+1):(3*p+1)]- xopt[ll,gg,(6*p+3):(7*p+3)]
            betaopt[:,ll,gg] = xopt[ll,gg,(3*p+3):(4*p+3)]
            mu1opt[ll,gg] = xopt[ll,gg,p]
            mu2opt[ll,gg] = xopt[ll,gg,(3*p+1)]
            mu3opt[ll,gg] = xopt[ll,gg,(3*p+2)]
            (predtrain[:,ll,gg],errortrain[:,ll,gg],msetrain[ll,gg],R2train[ll,gg]) = predict(train,a1opt[:,ll,gg],mu1opt[ll,gg],a2opt[:,ll,gg],mu2opt[ll,gg],a3opt[:,ll,gg],mu3opt[ll,gg],gamma)
            (predtest[:,ll,gg],errortest[:,ll,gg],msetest[ll,gg],R2test[ll,gg]) = predict(test,a1opt[:,ll,gg],mu1opt[ll,gg],a2opt[:,ll,gg],mu2opt[ll,gg],a3opt[:,ll,gg],mu3opt[ll,gg],gamma)
            coefsnonulos[ll,gg] = np.sum(np.absolute(np.around(a1opt[:,ll,gg],decimals=3))>=0.001,axis=0)+np.sum(np.absolute(np.around(a2opt[:,ll,gg],decimals=3))>=0.001,axis=0)+np.sum(np.absolute(np.around(a3opt[:,ll,gg],decimals=3))>=0.001,axis=0)
            numberofeatures[ll,gg] = np.sum(np.logical_or(np.absolute(np.around(a1opt[:,ll,gg],decimals=3))>=0.001,np.logical_or(np.absolute(np.around(a2opt[:,ll,gg],decimals=3))>=0.001,np.absolute(np.around(a3opt[:,ll,gg],decimals=3))>=0.001)),axis=0)
            localsparsity[ll,gg] = 100*(3*p-coefsnonulos[ll,gg])/(3*p)
            globalsparsity[ll,gg] = 100*(p-numberofeatures[ll,gg])/p
            
    return (a1opt,mu1opt,a2opt,mu2opt,a3opt,mu3opt,betaopt,gamma,
            predtrain,errortrain,msetrain,R2train,
            predtest,errortest,msetest,R2test,
            localsparsity,globalsparsity)