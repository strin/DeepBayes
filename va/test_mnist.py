from va import *
import numpy as np
import sys, os
import scipy.io as sio
from multiprocessing import Pool
import itertools
import pickle, gzip

toFloat = np.vectorize(float)

def universal_worker(input_pair):
    function, args = input_pair
    return function(*args)

def pool_args(function, *args):
    return zip(itertools.repeat(function), zip(*args))
    
def run(hidden, kappa, sigma, stepsize):
  mat = sio.loadmat('../data/mnist/mnistSmall.mat')
  train_data = np.array(mat['trainData'])
  train_label = np.argmax(np.array(mat['trainLabels']), axis=1)
  test_data = np.array(mat['testData'])
  test_label = np.argmax(np.array(mat['testLabels']), axis=1)

  output_path = '../result/hidden_%d_kappa_%f_sigma_%f' % (hidden, kappa, sigma)
  os.system('mkdir -p ../result/%s' % output_path)
  model = DeepLatentGM([784, hidden, hidden], batchsize=128, kappa=kappa, sigma=sigma, rec_hidden=hidden, stepsize=stepsize,\
                        num_label=10)
  model.train(train_data, train_label, 500, test_data = test_data, test_label = test_label, output_path=output_path)

def run_full(hidden, kappa, sigma, c, stepsize):
  mat = pickle.load(gzip.open('../data/mnist/mnist.pkl.gz', 'rb'))
  train_data = np.array(list(mat[0][0]) + list(mat[1][0]))
  train_label = np.array(list(mat[0][1]) + list(mat[1][1]))
  test_data = mat[2][0]
  test_label = mat[2][1]

  output_path = '../result/hidden_%d_kappa_%f_sigma_%f_c_%f' % (hidden, kappa, sigma, c)
  os.system('mkdir -p ../result/%s' % output_path)
  model = AutoEncoder([784, hidden], num_sample=1, batchsize=512, kappa=kappa, sigma=sigma, stepsize=stepsize,\
                        num_label=10, c = c, ell=10)
  model.train(train_data, train_label, 500, test_data = test_data, test_label = test_label)

def run_tiny(hidden, kappa, sigma, c, stepsize):
  mat = sio.loadmat('../data/mnist/mnistTiny.mat')
  train_data = np.array(toFloat(mat['trainData']))   # binarize.
  train_label = np.argmax(np.array(mat['trainLabels']), axis=1)
  test_data = np.array(toFloat(mat['testData']))     # binarize.
  test_label = np.argmax(np.array(mat['testLabels']), axis=1)

  model = AutoEncoder([784, hidden], num_sample=1, batchsize=32, kappa=kappa, sigma=sigma, stepsize=stepsize,\
                        num_label=10, c = c, ell=10)
  model.train(train_data, train_label, 500, test_data = test_data, test_label = test_label)

# run_tiny(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), 0.01)
run_full(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), 0.01)

