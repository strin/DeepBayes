from dlgm import *
import numpy as np
import sys, os
import scipy.io as sio
from multiprocessing import Pool
import itertools

def universal_worker(input_pair):
    function, args = input_pair
    return function(*args)

def pool_args(function, *args):
    return zip(itertools.repeat(function), zip(*args))
    
def run(kappa, sigma, stepsize):
  mat = sio.loadmat('../data/mnist/mnistSmall.mat')
  train_data = np.array(mat['trainData'])
  train_label = np.argmax(np.array(mat['trainLabels']), axis=1)
  test_data = np.array(mat['testData'])
  test_label = np.argmax(np.array(mat['testLabels']), axis=1)

  output_path = '../result/hidden_100_stepsize_%f' % ( stepsize)
  os.system('mkdir -p ../result/%s' % output_path)
  model = DeepLatentGM([784, 50], batchsize=128, kappa=kappa, sigma=sigma, rec_hidden=50, stepsize=stepsize,\
                        num_label=10)
  model.train(train_data, train_label, 2000, test_data = test_data, test_label = test_label, output_path=output_path)

pool = Pool(10)
# pool.map(universal_worker, pool_args(run, [0] * 6, [0, 0.1, 0.01, 0.001, 0.0001, 0.00001], [0.01] * 6))
pool.map(universal_worker, pool_args(run, [0] * 1, [0.01], [0.001] * 1))
