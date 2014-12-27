from dlgm import *
import numpy as np
import scipy.io as sio

def test_h1_v3():
  v1 = [1,0,1]
  v2 = [0,0,0]
  train_data = np.array([v1 for i in range(500)]+[v2 for i in range(500)])
  test_data = np.array([v1 for i in range(50)] + [v2 for i in range(50)])
  print 'training data', train_data
  model = DeepLatentGM([3, 4], batchsize=1, rec_hidden=1, kappa=0, stepsize=1)
  model.train(train_data, 10, test_data = train_data)
  print 'Generative Model', model.gmodel.pack()
  print 'Recognition Model', model.rmodel.pack()
  print 'Sample', model.sample(test_data)

test_h1_v3()

