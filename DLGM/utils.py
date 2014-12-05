import math
import numpy as np
import numpy.linalg as npla
import numpy.random as npr

def param_op2(param, grad, op):
  for i in range(len(param)):
    param[i][:] = op(param[i], param[i])[:]

def param_op(param, op):
  for i in range(len(param)):
    param[i][:] = op(param[i])[:]

def param_add(param, grad):
  res = grad
  if param != []:
    for i in range(len(param)):
      res[i] += param[i]
  return res

def param_mul_scalar(param, scalar):
  res = param
  for i in range(len(param)):
    res[i] = param[i] * scalar
  return res

def param_neg(param):
  res = param
  for i in range(len(param)):
    res[i] = -param[i]
  return res

def randn01(*shape):
  """
  generate random vector/matrix from i.i.d. Gaussian. 
  renormalize it to unit vector/matrix.
  """
  M = npr.randn(*shape)
  return M/npla.norm(M)
