Repository ORRT contains the code for the approach proposed by R. Blanquero, E. Carrizosa,
C. Molero-Río and D. Romero Morales in:
https://www.researchgate.net/publication/341099512_On_Sparse_Optimal_Regression_Trees

All the files in this repository can be run in Python 3.8.

-> AUXILIARY FILES:

	* 'scaling.py': given a data set, the goal of this program is to scale 
	 each of its predictor variables to the 0-1 interval. 

	* 'rescaling.py': the goal of this program is to scale each of the predictor
	 variables of a data set with the scaling parameters obtained after running 
	 the function scaling.py

	* 'predict.py': given a data set and the parameters of an SORRT with depth D = 1,
         the goal of this program is to provide predictions and measurements of prediction
         accuracy. 
   	
-> MAIN FILES:

	* 'ORRT_D1.py': the goal of this program is to read a training and a test subset,
	 solve Problem (3)-(5) with lambda^L = lambda^G = 0 on the training data set after
	 scaling and make predictions on the test data set after rescaling.

	* 'ORRT_D1_lambdas.py': the goal of this program is to read a training and a test 
	 subset, solve Problem (3)-(5) for a wide grid of values (lambda^L,lambda^G) on the
	 training subset after scaling and make predictions on the test subset after rescaling. 

In each specific py file, a more detailed description of the program, such as the inputs and 
outputs, can be found.
