
import numpy as np
import scipy.stats as stat
from scipy import stats
from sklearn import linear_model as lm

ALPHA_VALUES = [0.0, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
RHO_VALUES = [0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0]

def regBoost(regressionModel, bootstrap = True):
    # Default regression model
    model = lm.LinearRegression()
    for a,alpha in enumerate(ALPHA_VALUES):
        for r,rho in enumerate(RHO_VALUES):
            if regressionModel == 'ElasticNet':
                model = lm.ElasticNet(alpha = alpha, rho = rho)
            elif regressionModel == 'Lasso':
                model = lm.Lasso(alpha = alpha)
            elif regressionModel == 'Ridge':
                model = lm.Ridge(alpha = alpha)
                
            model.fit([[0, 0], [1, 1]], [0, 1])
            print model.coef_


    
if __name__ == '__main__':
    regBoost('Lasso')
    