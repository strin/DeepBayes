import numpy.random as npr
from va import *
import unittest

class TestDecoder(unittest.TestCase):

  def test_gen(self):
    model = Decoder([2, 4])
    xi = npr.randn(4,2)
    # print xi
    v = np.array([[1, 0], [0, 1]]).T
    param = model.pack()
    # print v
    # print model.sample(xi)
    # print model.get_lhood(v, xi)
    resp = np.dot(param['W1'], np.dot(param['G'], xi)) + param['b1']
    lhood = (v) * np.log(np.logistic(resp)) + (1-v) * np.log(1-np.logistic(resp))
    # print lhood.sum()
    assert(np.abs(lhood.sum() - model.get_lhood(v, xi)) < 1e-4)

  def test_gen_grad(self):
    model = Decoder([2, 4])
    xi = npr.randn(4,1)
    # print xi
    v = np.array([[1, 0]]).T
    gradient = model.get_grad(v, xi)
    # print gradient

  def test_gen_grad_xi(self):
    model = Decoder([2, 4])
    xi = npr.randn(4,1)
    v = np.array([[1, 0]]).T
    grad_xi = model.get_grad_xi(v, xi)
    # print grad_xi

class TestEncoder(unittest.TestCase):

  def test_reco(self):
    model = Encoder([2, 4], sigma=0.1)
    v = np.array([[1, 0]]).T
    model.sample_eps(v)




if __name__ == '__main__':
  unittest.main()
