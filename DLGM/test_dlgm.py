"""
test module for dlgm.py
"""
from dlgm import *
import numpy as np
import numpy.random as npr

if __name__ == "__main__":
  """
  arch = [1,2,4]
  nn = GenerativeModel(arch)
  print [npr.randn(i) for i in arch[1:]]
  print nn.sample(*[npr.randn(i) for i in arch[1:]])
  """

  arch = [2,2,4]
  nn = RecognitionModel(arch)
  print nn.sample([2,3])
  
