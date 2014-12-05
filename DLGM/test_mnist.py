from dlgm import *
import numpy as np
import scipy.io as sio

mat = sio.loadmat('../data/mnist/mnistSmall.mat')
train_data = np.array(mat['trainData'])
test_data = np.array(mat['testData'])

model = DeepLatentGM([784, 50, 50], batchsize=512, kappa=0.1, sigma=0.01, rec_hidden=50, stepsize=0.01)
model.train(train_data, 2000, test_data = test_data)


