"""
test module for dlgm.py
"""
from dlgm import *
import numpy as np
import numpy.random as npr
import unittest

class TestGenerativeModel(unittest.TestCase):

  def setUp(me):
    pass

  def test_nonlinear(me):
    arr = [1, -1, 3, 3, 5, 6]
    arch = [1,2]
    nn = GenerativeModel(arch)
    me.assertEqual(list(nn.nonlinear(arr)), [1,0,3,3,5,6])
  
  def test_generate(me):
    arch = [1,2]
    nn = GenerativeModel(arch)
    xi = [npr.randn(i) for i in arch[1:]]
    h1 = np.dot(nn.G[1].get_value(), xi[0])
    h0 = np.dot(nn.W[0].get_value(), nn.nonlinear(h1)) + nn.b[0].get_value()
    res = nn.generate(*xi)
    me.assertEqual(res[0], h0)
    assert((res[1] == h1).all())

  def test_recognition(me):
    arch = [5, 10]
    nn = RecognitionModel(arch)
    v = [1, 0, 0, 1, 1]
    z = nn.nonlinear(np.dot(nn.Wv[1].get_value(), v) + nn.bv[1].get_value())
    me.assertEqual(list(nn.get_z(v)[0]), list(z))
    mu = np.dot(nn.Wmu[1].get_value(), z) + nn.bmu[1].get_value()
    me.assertEqual(list(nn.get_mu(v)[0]), list(mu))
    d = np.exp(np.dot(nn.Wd[1].get_value(), z) + nn.bd[1].get_value())
    me.assertEqual(list(nn.get_d(v)[0]), list(d))
    u = np.dot(nn.Wu[1].get_value(), z) + nn.bu[1].get_value()
    me.assertEqual(list(nn.get_u(v)[0]), list(u))


if __name__ == "__main__":
  unittest.main() 
