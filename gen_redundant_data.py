#!/usr/bin/python

import numpy as np

def genRedundantData(samples, features, lc, redun_col):
    
    # The output matrix (samples, features)
    X = np.empty((samples, features), float)	

    # Generation of independant variable with uniform distribution in [-1,1]
    feat = features - lc * redun_col;
    for i in range(feat):
        X[:,i] = np.random.uniform(-1,1,samples)

    # Generate the dependant variable a a linear combination of first lc features
    Y= np.zeros((samples, 1), float)	
    for j in range(lc):
            Y = Y + X[:,j]
            
    # Add redundant columns (for each columns in lc create redun_col redundant columns)
    for i in range(lc):
        for j in range(redun_col):
            X[:,feat] = X[:, i]
            feat += 1

    # Normalize X and Y
    meanX = np.mean(X, axis=0) 
    stdX = np.std(X, axis=0)
    X = (X - meanX)/stdX 		

    return Y, X


######################################################################
if __name__ == "__main__":

	# Funcion testing: generate the data and test the features
	# Columns 2 and 3 are equal to column 0 and columns 4 and 5 equal to 1
	# X1 = Y - X0
	Y,X = gen_data_redun(100,6, 2, 2)
        
	 
