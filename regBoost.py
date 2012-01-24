
import numpy as np
#import scipy.stats as stat
#from scipy import stats
import scipy
from scikits.learn import linear_model as lm
import time
import logging
import os

from utils import create_output_dirs, genbootstrap, getrankings, read_input_files, plot_weights_iter, plot_bennet_mean_std, add_to_zip, ALPHA_VALUES, RHO_VALUES
import generate_data
from gen_redundant_data import genRedundantData

logging.basicConfig(level=logging.DEBUG)
    
def regBoost(x, y, regressionModel, bootstrap_num, impFeat):
    
    intersect_size = 0
    #features_selected = 0

    start_time = time.time()

    weights = np.zeros([features], float)

    # To record weights over iterations
    weights_iter = np.zeros((bootstrap_num, features))
    
    max_position_iter = []
    rcors = []
    intersect_size_iter = []
    
    intersect = set()
    
    for j in range(bootstrap_num):
	logging.debug('Iteration %d', (j+1))

	x, y = genbootstrap(dx, dy)
	
	regressionModel.fit(x, y)
	
	weights += regressionModel.coef_
	
	weights_iter[j,:] = weights

	non_zero_features = np.sum(regressionModel.coef_ != 0)
	#print 'Non zero features: ', non_zero_features #regressionModel.coef_ != 0
	#features_selected += non_zero_features

	max_position, computed_ranking = getrankings(weights, impFeat)
	max_position_iter.append(max_position)
	
	rcor = scipy.stats.spearmanr(computed_ranking, np.array(range(impFeat)))[0]
	
	rcors.append(rcor)
			    
	l = np.argsort(regressionModel.coef_)[-non_zero_features:]
	#print 'L : ', l  
	
	if len(intersect):
	    intersect = set(l).intersection(intersect)
	else:
	    intersect = set(l)
	intersect = intersect.intersection(set(range(impFeat)))
	#print 'Intersect : ', len(intersect)    
	intersect_size_iter.append(len(intersect))
    end_time = time.time()
    deltatime = end_time - start_time  	
    
    return (weights_iter, rcors, max_position_iter, intersect_size_iter, deltatime)	
    
  
    
if __name__ == '__main__':
    
    
    # Parameters setting
    file_num = 10
    alpha = True	
    rho = True
    bootstrap = True
    samples = 100
    features = 100
    impFeat = 20
    bootstrap_num = 5
    regressionModel = 'ElasticNet'
    #regressionModel = 'Lasso'
    regressionModel = 'Ridge'
    
    
    files_rcors = np.zeros([file_num, bootstrap_num], float)
    files_max_position = np.zeros([file_num, bootstrap_num], float)
    files_intersect_size = np.zeros([file_num, bootstrap_num], float)
    files_deltatime = np.zeros([file_num], float)
    
    
    # Default regression model
    model = lm.LinearRegression()
    
    create_output_dirs()
    delta = open('results/' + regressionModel + '_delta.txt', 'wb')
    delta_content = ''
    
    if rho:
        rhos = RHO_VALUES
    else:
        rhos = [0]
    
    rc = ''
    best_alpha = 0
    best_rho = 0
    best_mean_rcor = 0
    best_mean_max_pos = features
    best_mean_max_intersect = 0
    
    if regressionModel == 'ElasticNet':
	model = lm.ElasticNet(alpha = alpha, rho = rho)
    elif regressionModel == 'Lasso':
	model = lm.Lasso(alpha = alpha)
    elif regressionModel == 'Ridge' and alpha != 0:
	model = lm.Ridge(alpha = alpha)
        
    for a,alpha in enumerate(ALPHA_VALUES):
        for r,rho in enumerate(rhos):
            for k in range(file_num):
		dy, dx = generate_data.gen_data(samples, features, impFeat)
		#dy, dx = genRedundantData(100, 6, 2, 2)
                examples, features = dx.shape
		
		(weights_iter, rcors, max_position_iter, intersect_size_iter, deltatime) = regBoost(dx, dy, model, bootstrap_num, impFeat)
	    
		files_rcors[k,:] = rcors
                if np.mean(rcors) > best_mean_rcor:
			best_mean_rcor = np.mean(rcors)
			best_alpha = alpha
			if regressionModel == 'ElasticNet' and rho != 0:
				best_rho = rho
			if best_mean_max_pos < np.mean(max_position_iter):
			    best_mean_max_pos = np.mean(max_position_iter)
			if best_mean_max_intersect > np.mean(intersect_size_iter):
			    best_mean_max_intersect = np.mean(intersect_size_iter)
			    
	    	rc += str(alpha) + ',' + str(rho) + ',' + str(np.mean(rcors)) + '\n'
		files_max_position[k,:] = max_position_iter
                files_intersect_size[k,:] = intersect_size_iter
                files_deltatime[k] = deltatime  
		
		# Plot weights
                plot_weights_iter(alpha, rho, k, bootstrap_num, features, weights_iter)
                fname = 'results/' + regressionModel + '_w_' + str(alpha) + '_' + str(rho) + '_' + str(k) + '.txt'
                np.savetxt(fname, weights_iter)                
                add_to_zip(fname)
                os.remove(fname)
                
                
	    print alpha, rho, np.sum(deltatime)/file_num
            delta_content += str(alpha) + ' ,' + str(rho) + ' ,' + str(np.sum(deltatime)/file_num) + '\n'
            
            title = 'Min-Max of Rcors'
            fname_prefix = 'minmax_rcors'
            plot_bennet_mean_std(title, fname_prefix, alpha, rho, bootstrap_num, files_rcors)
				
 
            title = 'Min-Max of MaxPos'
            fname_prefix = 'minmax_maxpos'
            plot_bennet_mean_std(title, fname_prefix, alpha, rho, bootstrap_num, files_max_position)
            
            title = 'Min-Max of Intersect Size'
            fname_prefix = 'minmax_intersect'
            plot_bennet_mean_std(title, fname_prefix, alpha, rho, bootstrap_num, files_intersect_size)
            
            #average_intersect_size = intersect_size/file_num
            #avg_features_selected = features_selected/(file_num*bootstrap_num)	 
    delta.write(delta_content)
    delta.close()
    
    print rc	
    print 'Best param : ', best_alpha, best_rho, best_mean_rcor, best_mean_max_pos, best_mean_max_intersect
