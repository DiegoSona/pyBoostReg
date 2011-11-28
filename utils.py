
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
import numpy as np
import scipy as sp
import scipy.stats as stat
import tarfile

ALPHA_VALUES = [0.0, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
RHO_VALUES = [0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0]


"""Create plots and results directories if do not exist"""
def create_output_dirs():
	import os, errno
	try:
		os.makedirs('plots')
		os.makedirs('results')
	except OSError, e:
		if e.errno != errno.EEXIST:
			raise


def genbootstrap(x, y):    
    X = []
    Y = []
    examples, fearures = x.shape
    for i in range(examples):
        j = stat.randint.rvs(0, examples - 1)
        X.append(x[j])
        Y.append(y[j])
    
    return X, Y

# get ranking of top k features 
def getrankings(arr, k):
   l = np.argsort(arr)
   l_reversed = l[::-1]
   max_position = 0
   computed_ranking = []
   for id, item in enumerate(l_reversed):
      if item < k:
         max_position = id
         computed_ranking.append(item)
   computed_ranking = np.asarray(computed_ranking)
   #max_position += 1
   return max_position, computed_ranking

# Read input files using iterators
def read_input_files(subfolder, k):
    for i in range(k):
        dx = np.loadtxt(subfolder + "x" + str(i+1) + ".txt")
        dy = np.loadtxt(subfolder + "y" + str(i+1) + ".txt")
        yield dx, dy

def array_filter(selector, list):
    return map(lambda x : x[selector], list)
    
def plot_weights_iter(alpha, rho, file_num, bootstrap_num, features, weights_iter):
    pl.legend()
    pl.xlabel('Iterations')
    pl.title('Alpha = %f, Rho = %f' % (alpha, rho))
    for f in range(features):
        pl.plot(weights_iter[:,f])
    pl.savefig('plots/' + 'weights_a' + str(alpha) + '_r' + str(rho) + '_' + str(file_num) + '.png')
    pl.cla()
    #pl.show()
    
def plot_weights_no_boot(alpha, rho, file_num, bootstrap_num, features, weights):
    pl.legend()
    pl.xlabel('Features')
    pl.title('Alpha = %f, Rho = %f' % (alpha, rho))
    pl.plot(weights)
    pl.savefig('plots/' + 'weights_a' + str(alpha) + '_r' + str(rho) + '_' + str(file_num) + '.png')
    pl.cla()
    #pl.show()    
    
def plot_bolasso_weights_iter(model_name, alpha, file_num, bootstrap_num, features, weights_iter):
    pl.legend()
    pl.xlabel('Iterations')
    pl.title('Alpha = %f' % (alpha))
    for f in range(features):
        pl.plot(weights_iter[:,f])
    pl.savefig('plots/' + model_name + '_weights_a' + str(alpha) + '_' + str(file_num) + '.png')
    pl.cla()
    
    
def plot_bennet_mean_std1(title, fname_prefix, alpha, rho, mean, std):
    pl.legend()	
    pl.xlabel('Iterations')	
    pl.title('%s - Alpha = %f, Rho = %f' % (title, alpha, rho))	
    pl.errorbar(range(len(mean)), mean,	yerr=std)	
    pl.savefig('plots/' + fname_prefix + '_a' + str(alpha) + '_r' + str(rho) + '.png')	
    pl.cla()
    
def plot_bennet_mean_std(title, fname_prefix, alpha, rho, bootstrap_num, values):
    pl.legend()	
    pl.xlabel('Iterations')	
    pl.title('%s - Alpha = %f, Rho = %f' % (title, alpha, rho))	
    deltas = []
    deltas.append(np.mean(values, axis=0) - np.min(values, axis=0))
    deltas.append(np.max(values, axis=0) - np.mean(values, axis=0))
    deltas = np.asarray(deltas)
    deltas.reshape(2, bootstrap_num)
    pl.errorbar (range( bootstrap_num), np.mean(values, axis=0), yerr=deltas)
    pl.savefig('plots/' + fname_prefix + '_a' + str(alpha) + '_r' + str(rho) + '.png')	
    pl.cla()    
    
def plot_bolasso_mean_std(title, fname_prefix, alpha, mean, std):
    pl.legend()	
    pl.xlabel('Iterations')	
    pl.title('%s - Alpha = %f' % (title, alpha))	
    pl.errorbar(range(len(mean)), mean,	yerr=std)	
    pl.savefig('plots/' + fname_prefix + '_a' + str(alpha) + '.png')	
    pl.cla()	    

def plot_no_bootstrap_model(title, fname_prefix, alpha, rho, w):
    pl.legend()	
    pl.xlabel('Files')	
    pl.title('%s - Alpha = %f' % (title, alpha))	
    pl.plot(range(10), w)	
    pl.savefig('plots/' + fname_prefix + '_a' + str(alpha) + '_r' + str(rho) + '.png')	
    pl.cla()
    
def add_to_zip(filename):
    tar = tarfile.open(filename + ".tar.gz", "w:gz")
    tar.add(filename)
    tar.close()

    
