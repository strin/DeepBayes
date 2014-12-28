import scipy.io as sio
import pickle, gzip
import numpy.random as npr
import numpy as np
import pdb
import matplotlib.pyplot as plt

toFloat = np.vectorize(float)

mat = sio.loadmat('../data/mnist/mnistTiny.mat')
train_data = np.array(mat['trainData'])   # binarize.
train_label = np.argmax(np.array(mat['trainLabels']), axis=1)
test_data = np.array(mat['testData'])     # binarize.
test_label = np.argmax(np.array(mat['testLabels']), axis=1)

batchsize = 32
num_iter = 10000
D = train_data.shape[1]
eta = 1

W = npr.randn(D, 10)
G2 = np.zeros_like(W)
data_mean = np.mean(train_data, axis=0)
train_data -= data_mean


def test_acc():
  resp = np.dot(test_data - data_mean, W)
  predict = np.argmax(resp, 1)
  return np.sum(predict == test_label) / float(len(resp))

acc = []
for it in range(num_iter):
  ind = npr.choice(range(len(train_data)), batchsize, replace=False)
  g = np.zeros_like(W)
  for (x, y) in zip(train_data[ind], train_label[ind]):
    resp = 100 + np.dot(x, W) - np.dot(x, W[:,y])
    resp[y] = 0
    yp = np.argmax(resp) 
    g[:,yp] -= x
    g[:,y] += x
  g /= float(batchsize)
  G2 += g * g
  W += eta * g / (1e-4 + np.sqrt(G2))
  acc += [test_acc()]
  print 'iter = ', it, ' , acc = ', acc[-1]

# sio.savemat('result.mat', {'W':W, 'acc':acc})
plt.plot(acc)
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.savefig('mnist.png')
plt.show()



    
    

