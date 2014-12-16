from dlgm import *
import numpy as np
import scipy.io as sio

mat = sio.loadmat('../data/mnist/mnistSmall.mat')
train_data = np.array(mat['trainData'])
train_label = np.argmax(np.array(mat['trainLabels']), axis=1)
test_data = np.array(mat['testData'])
test_label = np.argmax(np.array(mat['testLabels']), axis=1)

model = DeepLatentGM([784, 50, 50], batchsize=512, kappa=0.1, sigma=0.01, rec_hidden=50, stepsize=0.01, num_label=10)
model.train(train_data, train_label, 2000, test_data = test_data, test_label = test_label)


