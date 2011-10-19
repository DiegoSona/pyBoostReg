
import numpy as np
import scipy.stats as stat
import scipy
from scipy import stats
from scikits.learn import linear_model
from scikits.learn.linear_model import Lasso
import logging
import random
import os
from bm_utils import bootstrap, getrankings, read_input_files, plot_weights_no_boot, plot_no_bootstrap_model, add_to_zip, ALPHA_VALUES, RHO_VALUES
import time


def lasso(file_num = 10, impfeat = 100, bootstrap_num = 10):
   
    logging.basicConfig(level=logging.DEBUG)

    
    lasso_delta = open('results/lasso_delta.txt', 'wb')
  
    rho = 0
    delta = ''
    
    for a,alpha in enumerate(ALPHA_VALUES):
        input_files = read_input_files('data/', 1)
        
        files_rcors = np.zeros([file_num], float)
        files_max_position = np.zeros([file_num,], float)
        files_intersect_size = np.zeros([file_num], float)
                
        intersect_size = 0
        features_selected = 0
        
        rcors = []
        max_positions = []
        intersects = []
    
        files_deltatime = np.zeros([file_num], float)
        start_time = time.time()
                    
        for k in range(file_num):
            dx, dy = input_files.next()

            examples, features = dx.shape 

            lasso = Lasso(alpha=alpha)
            lasso.fit(dx, dy)
            
            intersect = set()
                     
            non_zero_features = np.sum(lasso.coef_ != 0)
            features_selected += non_zero_features
    
            max_position, computed_ranking = getrankings(lasso.coef_, impfeat)
            max_positions.append(max_position)
                
            rcor = scipy.stats.spearmanr(computed_ranking, np.array(range(impfeat)))[0]
            rcors.append(rcor)
                                    
            l = np.argsort(lasso.coef_)[-non_zero_features:]  
                
            if len(intersect):
                intersect = set(l).intersection(intersect)
            else:
                intersect = set(l)
            intersect = intersect.intersection(set(range(impfeat)))
            intersects.append(len(intersect))
                
                      
            #plot weights
            plot_weights_no_boot(alpha, rho, k, bootstrap_num, features, lasso.coef_)
            fname = 'results/weights_' + str(alpha) + '_' + str(rho) + '_' + str(k) + '.txt'
            np.savetxt(fname, lasso.coef_)                
            add_to_zip(fname)
            os.remove(fname)
        
               
        end_time = time.time()
        files_deltatime[k] = end_time - start_time
            
        print alpha, rho, np.sum(files_deltatime)/file_num
        delta += str(alpha) + ' ,' + str(rho) + ' ,' + str(np.sum(files_deltatime)/file_num) + '\n'
        
        title = 'Rcors'
        fname_prefix = 'rcors'
        plot_no_bootstrap_model(title, fname_prefix, alpha, rho, rcors)
        
        
        title = 'Max Position'
        fname_prefix = 'max_pos'
        plot_no_bootstrap_model(title, fname_prefix, alpha, rho, max_positions)
        
        title = 'Intersect Size'
        fname_prefix = 'intersect'
        plot_no_bootstrap_model(title, fname_prefix, alpha, rho, intersects)
        
    lasso_delta.write(delta)
    lasso_delta.close()
    


if __name__ == "__main__":
   lasso(file_num = 10, impfeat = 100, bootstrap_num = 10)
