
import numpy as np
#import scipy.stats as stat
#from scipy import stats
import scipy
from scikits.learn import linear_model as lm
import time
import logging
import os

from utils import genbootstrap, getrankings, read_input_files, plot_weights_iter, plot_bennet_mean_std, add_to_zip, ALPHA_VALUES, RHO_VALUES

BOOTSTRAP_NUM = 5
IMPFEAT = 100

logging.basicConfig(level=logging.DEBUG)


def regBoost(regressionModel, file_num, bootstrap = True, alpha = True, rho = False):
    
    # Default regression model
    model = lm.LinearRegression()
    
    if rho:
        rhos = RHO_VALUES
    else:
        rhos = [0]
        
    delta = open('results/' + regressionModel + '_delta.txt', 'wb')
    delta_content = ''
    
    for a,alpha in enumerate(ALPHA_VALUES):
        for r,rho in enumerate(rhos):
            
            if regressionModel == 'ElasticNet':
                model = lm.ElasticNet(alpha = alpha, rho = rho)
            elif regressionModel == 'Lasso':
                model = lm.Lasso(alpha = alpha)
            elif regressionModel == 'Ridge':
                model = lm.Ridge(alpha = alpha)
                
            input_files = read_input_files('data/', file_num)
            
            files_rcors = np.zeros([file_num, BOOTSTRAP_NUM], float)
            files_max_position = np.zeros([file_num, BOOTSTRAP_NUM], float)
            files_intersect_size = np.zeros([file_num, BOOTSTRAP_NUM], float)
                    
            intersect_size = 0
            features_selected = 0

            files_deltatime = np.zeros([file_num], float)
            start_time = time.time()
            
            for k in range(file_num):
                dx, dy = input_files.next()
                examples, features = dx.shape 

                weights = np.zeros([features], float)
            
                # To record weights over iterations
                weights_iter = np.zeros((BOOTSTRAP_NUM, features))
                
                max_position_iter = []
                rcors = []
                intersect_size_iter = []
                
                intersect = set()
                
                nx = []
                ny = []
                if bootstrap:
                    for j in range(BOOTSTRAP_NUM):
                        x, y = genbootstrap(dx, dy)
                        nx.append(np.asarray(x))  
                        ny.append(np.asarray(y))
                else:
                    nx.append(dx)  
                    ny.append(dy)
                    
                for j in range(len(nx)):
                    logging.debug('Iteration %d', (j+1))
                    x = nx[j]
                    y = ny[j]
                    model.fit(x, y)
                    
                    weights += model.coef_
                    
                    weights_iter[j,:] = weights
          
                    non_zero_features = np.sum(model.coef_ != 0)
                    features_selected += non_zero_features
       
                    max_position, computed_ranking = getrankings(weights, IMPFEAT)
                    max_position_iter.append(max_position)
                    
                    rcor = scipy.stats.spearmanr(computed_ranking, np.array(range(IMPFEAT)))[0]
                    
                    rcors.append(rcor)
                                        
                    l = np.argsort(model.coef_)[-non_zero_features:]  
                    
                    if len(intersect):
                        intersect = set(l).intersection(intersect)
                    else:
                        intersect = set(l)
                    intersect = intersect.intersection(set(range(IMPFEAT)))    
                    intersect_size_iter.append(len(intersect))
                    
                # Plot weights
                plot_weights_iter(alpha, rho, k, BOOTSTRAP_NUM, features, weights_iter)
                fname = 'results/' + regressionModel + '_W_' + str(alpha) + '_' + str(rho) + '_' + str(k) + '.txt'
                np.savetxt(fname, weights_iter)                
                add_to_zip(fname)
                os.remove(fname)
                
                files_rcors[k,:] = rcors
                files_max_position[k,:] = max_position_iter
                files_intersect_size[k,:] = intersect_size_iter
                
                end_time = time.time()
                files_deltatime[k] = end_time - start_time    
            

            
            print alpha, rho, np.sum(files_deltatime)/file_num
            delta_content += str(alpha) + ' ,' + str(rho) + ' ,' + str(np.sum(files_deltatime)/file_num) + '\n'
            
            title = 'Min-Max of Rcors'
            fname_prefix = 'minmax_rcors'
            plot_bennet_mean_std(title, fname_prefix, alpha, rho, BOOTSTRAP_NUM, files_rcors)
            
            title = 'Min-Max of MaxPos'
            fname_prefix = 'minmax_maxpos'
            plot_bennet_mean_std(title, fname_prefix, alpha, rho, BOOTSTRAP_NUM, files_max_position)
            
            title = 'Min-Max of Intersect Size'
            fname_prefix = 'minmax_intersect'
            plot_bennet_mean_std(title, fname_prefix, alpha, rho, BOOTSTRAP_NUM, files_intersect_size)
            
            average_intersect_size = intersect_size/file_num
            avg_features_selected = features_selected/(file_num*BOOTSTRAP_NUM)
            
    delta.write(delta_content)
    delta.close()
    
if __name__ == '__main__':
    regBoost('ElasticNet', 10)
    