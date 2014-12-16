"""
implements the models in paper "stochastic backpropagation in DLGMs"
including
* generative model.
* recognition model.
"""
"let client and server have the same imports."
imports = ['import numpy as np', 
           'import numpy.random as npr', 
           'import theano',
           'import scipy.io as sio', 
           'import theano.sandbox.linalg as ta',
           'import theano.tensor as ts',
           'from color import *',
           'from utils import *']
for _import in imports:
  exec _import
import pdb

import time
from IPython.parallel import Client

theano.config.exception_verbosity = 'low'

ts.logistic = lambda z: 1 / (1 + ts.exp(-z)) 
np.logistic = lambda z: 1 / (1 + np.exp(-z))
toInt = np.vectorize(int)

get_value = lambda x: x.get_value() if x != None else None
get_value_all = lambda xs: [get_value(x) for x in xs]

def AdaGRAD(param, grad, G2, stepsize):
  """
  adaptive sub-gradient algorithm for tensor-shared objects.
  > input:
    param: parameters, tensor-shared objects. 
    grad: gradient, list of numpy arrays.
    G2: variance of gradient, list of numpy arrays
  """
  for (p, g, g2) in zip(param, grad, G2):
    g2[:] = (g2 + g * g)[:]
    if type(p) == theano.tensor.sharedvar.TensorSharedVariable:
      p.set_value(p.get_value() - stepsize * g / (1e-4 + np.sqrt(g2))) 
    elif type(p) == np.ndarray:
      p[:] = p[:] - stepsize * g[:] / (1e-4 + np.sqrt(g2[:]))

class GenerativeModel:
  """ generative model 
  """
  def __init__(me, arch, kappa=0.1):
    """
      create the deep latent Gaussian model.
        arch: architecture, [vis, hidden_1, hidden_2, ...]
    """
    "set options."
    me.f = ts.maximum         # nonlinear transformation.
    me.lhoodFunc = lambda v, resp: (v * ts.log(ts.logistic(resp)) + (1-v) * ts.log(1-ts.logistic(resp))).sum()

    "set properties."
    me.arch = arch
    me.num_layers = len(arch)
    me.kappa = kappa
    assert(me.num_layers > 1)

    "init layers."
    (me.G, me.W, me.b, me.xi, me.h) = tuple([[None]*(me.num_layers) for i in range(5)])
    for layer in range(me.num_layers-1, -1, -1):
      if layer < me.num_layers-1:
        me.W[layer] = theano.shared(randn01(arch[layer], arch[layer+1]), name="W%d" % layer)
        me.b[layer] = theano.shared(np.zeros(arch[layer]), name="b%d" % layer)
      me.h[layer] = 0
      if layer > 0:
        me.G[layer] = theano.shared(np.eye(arch[layer]), name="G%d" % layer)
        me.xi[layer] = ts.vector("xi%d" % layer)
        me.h[layer] += ts.dot(me.G[layer], me.xi[layer])
      if layer < me.num_layers-1:
        me.h[layer] += ts.dot(me.W[layer], me.f(0, me.h[layer+1])) + me.b[layer]

    me.param = me.G[1:] + me.W[:-1] + me.b[:-1]
    me.G2 = [np.zeros(x.get_value().shape) for x in me.param] # variance of gradient.

    "define objective."
    me.v = ts.vector("v")
    me.lhood = me.lhoodFunc(me.v, me.h[0])
    me.get_lhood = theano.function([me.v] + me.xi[1:], me.lhood) 
    me.reg = me.kappa * sum([ts.sum(p * p) for p in me.param])
    me.get_reg = theano.function([], me.reg)

    "define gradient."
    me.gradient = ts.grad(me.lhood, me.param)
    me.gradient_xi = ts.grad(me.lhood, me.xi[1:])
    # me.hessian_xi = ts.hessian(me.lhood, me.xi[1:])
    me.get_grad = theano.function([me.v] + me.xi[1:], me.gradient)
    me.get_grad_xi = theano.function([me.v] + me.xi[1:], me.gradient_xi)
    # me.get_hess_xi = theano.function([me.v] + me.xi[1:], me.hessian_xi)
    me.gradient_reg = ts.grad(me.reg, me.param)
    me.get_grad_reg = theano.function([], me.gradient_reg)

    "define utils."
    me.generate = theano.function(me.xi[1:], me.h)
    me.hidden_activation = ts.vector("hidden_activiation")
    me.hidden_rectified = me.f(0, me.hidden_activation)
    me.nonlinear = theano.function([me.hidden_activation], me.hidden_rectified)

  def sample(me, xi):
    h = me.generate(*xi)
    resp = np.logistic(h[0])
    return toInt(npr.rand(len(resp)) < resp)

  def pack(me):
    return {'G': get_value_all(me.G), \
            'W': get_value_all(me.W),
            'b': get_value_all(me.b)}
 
class RecognitionModel:
  """ recognition model (interface)
        since xi \sim \Normal(\mu, C) for each layer. 
        the recognition fits its parameters (\mu, C) discriminatively.

      a simple recognition model uses a two layer NN to fit each parameter.
      see DLGM appendix A.
  """
  def __init__(me, arch, num_hidden=10, sigma=1):
    """
      create the deep latent Gaussian recognition model.
        arch: architecture, [vis, hidden_1, hidden_2, ...]
    """
    "set options."
    me.f = ts.maximum         # nonlinear transformation.

    "set properties."
    me.arch = arch
    me.num_layers = len(arch)
    me.num_hidden = num_hidden
    me.sigma = sigma
    assert(me.num_layers > 1)

    "init layers."
    me.v = ts.vector("v")
    (me.Wv, me.Wu, me.Wd, me.Wmu, me.bv, me.bu, me.bd, me.bmu, me.z, me.d, me.u, me.mu, me.R, me.C) \
        = tuple([[None] * me.num_layers for i in range(14)])
    for layer in range(1, me.num_layers):
      me.Wv[layer] = theano.shared(randn01(num_hidden, arch[0]), name="Wv%d" % layer)
      me.Wu[layer] = theano.shared(randn01(arch[layer], num_hidden), name="Wu%d" % layer)
      me.Wd[layer] = theano.shared(randn01(arch[layer], num_hidden), name="Wd%d" % layer)
      me.Wmu[layer] = theano.shared(randn01(arch[layer], num_hidden), name="Wmu%d" % layer)
      me.bv[layer] = theano.shared(np.zeros(num_hidden), name="bv%d" % layer)
      me.bu[layer] = theano.shared(np.zeros(arch[layer]), name="bu%d" % layer)
      me.bd[layer] = theano.shared(np.zeros(arch[layer]), name="bd%d" % layer)
      me.bmu[layer] = theano.shared(np.zeros(arch[layer]), name="bmu%d" % layer)
      me.z[layer] =  me.f(0, ts.dot(me.Wv[layer], me.v) + me.bv[layer])
      me.mu[layer] = ts.dot(me.Wmu[layer], me.z[layer]) + me.bmu[layer]
      me.d[layer] = ts.exp(ts.dot(me.Wd[layer], me.z[layer]) + me.bd[layer])
      me.u[layer] = ts.dot(me.Wu[layer], me.z[layer]) + me.bu[layer]
      "me.R, me.C are auxiliary, and not used in real computation."
      me.R[layer] = ts.diag(ts.sqrt(me.d[layer])) + 0 * ts.dot(me.u[layer].T, me.u[layer])
      me.C[layer] = ts.dot(me.R[layer], me.R[layer].T) 


    """model covariance jointly
    utDneg1u = sum([ts.dot(u, u/d) for (u, d) in zip(me.u, me.d)])
    me.eta = 1/(1+utDneg1u)
    me.Rdot = theano.function([me.v] + [tensor.vector('x') for u in me.u], \
                [1/ts.sqrt(d) * x - ts.dot(1/ts.sqrt(d) * x, u) * ts.dot(u, 1/d) \
                  * (1-ts.sqrt(me.eta)) / utDneg1u \
                  for (u, d, x) in zip(me.u, me.d, me.x) \
                ]\
              )
    """
    xs = [ts.vector('x') for u in me.u] 
    me.Rdot = theano.function([me.v] + xs[1:], \
                [1/ts.sqrt(d) * x for (d, x) in zip(me.d[1:], xs[1:])] \
                )

    "utils."
    me.get_mu = theano.function([me.v], me.mu[1:])
    me.get_u = theano.function([me.v], me.u[1:])
    me.get_d = theano.function([me.v], me.d[1:])
    me.get_z = theano.function([me.v], me.z[1:])
    me.get_R = theano.function([me.v], me.R[1:])
    me.get_C = theano.function([me.v], me.C[1:])

    me.sample_eps = lambda v: [np.array([npr.normal(0, 1) for di in d]) \
                          for (mu, u, d) in zip(me.get_mu(v), me.get_u(v), me.get_d(v))]

    me.sample = lambda v, eps: param_add(me.get_mu(v), me.Rdot(v, *eps)) 

    me.hidden_activation = ts.vector("hidden_activiation")
    me.hidden_rectified = me.f(0, me.hidden_activation)
    me.nonlinear = theano.function([me.hidden_activation], me.hidden_rectified)

    "free energy."
    me.energy = 0;
    for layer in range(1, me.num_layers):
      me.energy += me.sigma * (ts.dot(me.mu[layer].T, me.mu[layer]) + ts.sum(1/me.d[layer])) \
                    + 0 * ts.dot(me.u[layer].T, me.u[layer]) \
                    +  ts.sum(ts.log(me.d[layer]))
    me.energy *= 0
    me.get_energy = theano.function([me.v], me.energy)

    "free energy gradients."
    me.param = me.Wv[1:] + me.Wu[1:] + me.Wd[1:] + me.Wmu[1:] + me.bv[1:] + me.bu[1:] + me.bd[1:]+ me.bmu[1:]
    me.G2 = [np.zeros(x.get_value().shape) for x in me.param] # variance of gradient.
    me.gradient = ts.grad(me.energy, me.param)
    me.get_grad = theano.function([me.v], me.gradient)
   
    """ stochastic gradients.
        trick. pretend our objective is inner product with the stochastic gradients.
    """
    me.grad_gm = [None] * me.num_layers
    me.eps = [None] * me.num_layers
    me.obj_mu = 0
    me.obj_R = 0
    for layer in range(1, me.num_layers):
      me.grad_gm[layer] = ts.vector('grad_gm_%d' % layer)
      me.eps[layer] = ts.vector('eps_%d' % layer)
      me.obj_mu += ts.sum(me.mu[layer] * me.grad_gm[layer])
      me.obj_R += .5 * ts.dot(me.grad_gm[layer] * me.eps[layer], 1/ts.sqrt(me.d[layer])) + 0 * ts.dot(me.u[layer].T,
          me.u[layer])
    me.stoc_grad = ts.grad(me.obj_mu + me.obj_R, me.param)
    me.get_stoc_grad = theano.function([me.v] + me.grad_gm[1:] + me.eps[1:], me.stoc_grad)
    
  def pack(me):
    return {'Wv': get_value_all(me.Wv), 
            'Wu': get_value_all(me.Wu),
            'Wd': get_value_all(me.Wd),
            'Wmu': get_value_all(me.Wmu),
            'bv': get_value_all(me.bv),
            'bu': get_value_all(me.bu),
            'bd': get_value_all(me.bd),
            'bmu': get_value_all(me.bmu)}
        

"parallel"
try:
  rc = Client()
  num_threads = len(rc)
  rc[:].use_dill()
  for _import in imports:
    rc[:].execute(_import)
  view = rc.load_balanced_view()
  view.block = True
  mapf = view.map
except:
  "cannot connect to parallel server."
  num_threads = 1
  mapf = map

def test(ti, v):
  return v

class DeepLatentGM(object):
  """
    train/test DLGM on datasets.
  """
  def __init__(me, arch, batchsize = 1, num_sample = 1, kappa = 1, sigma = 1, rec_hidden = 100, 
                    stepsize=0.1, num_label=2, ell=100, c = 1, v = 1):
    me.num_threads = num_threads
    printBlue('> Thread Pool (%d)' % me.num_threads)
    me.kappa = kappa
    me.batchsize = batchsize
    me.stepsize = stepsize
    me.num_sample = num_sample
    printBlue('> Compiling neural network')
    me.gmodel = GenerativeModel(arch, kappa=kappa)
    me.rmodel = RecognitionModel(arch, num_hidden=rec_hidden, sigma=sigma)

    me.ell = ell
    me.num_label = num_label
    me.W = np.zeros((sum(arch[1:]), me.num_label))
    me.W_G2 = np.zeros_like(me.W)
    me.c = c
    me.v = 1

  def __concat__(me, xi):
    latent = []
    for x in xi:
      latent += list(x)
    latent = np.array(latent)
    return latent

  def process(me, ti, V, Y = None):
    """
      process one single data point.
        > return: (grad of generative model, grad of recognition model)
        > input
          ti: thread id.
          v: data point.
    """
    rmodel = me.rmodel
    gmodel = me.gmodel

    grad_gs = []
    grad_rs = []
    grad_w = np.zeros_like(me.W)

    for (vi, v) in enumerate(V):
      grad_g = []
      grad_r = []
      for si in range(me.num_sample):
        "first sample stochastic variables."

        eps = rmodel.sample_eps(v)
        xi = rmodel.sample(v, eps)

        "compute gradient of generative model."
        gg = gmodel.get_grad(v, *xi)
        gg = param_neg(gg)
        grad_g = param_add(grad_g, gg)

        "compute gradient of regularizer in generative model."
        gg_reg = gmodel.get_grad_reg()
        gg_reg = param_mul_scalar(gg_reg, me.kappa)
        grad_g = param_add(grad_g, gg_reg)

        "compute free-energy gradient of recognition model."
        gr = rmodel.get_grad(v)
        grad_r = param_add(grad_r, gr) 

        "compute stochastic gradient of recognition model."
        gg_xi = gmodel.get_grad_xi(v, *xi)
        gg_xi = param_neg(gg_xi)

        # compute hessian: strategy 1. (numerically unstable)
        # hh_xi = gmodel.get_hess_xi(v, *xi)
        # hh_xi = param_neg(hh_xi)
        # hh_xi = param_mul_scalar(hh_xi, .5)
        "add supervision"
        if Y != None:
          latent = me.__concat__(xi)
          y = Y[vi]
          
          resp = me.ell + np.dot(latent, me.W) - np.dot(latent, me.W[:,y])
          resp[y] = 0
          yp = np.argmax(resp) 
          grad_w[:,yp] += latent
          grad_w[:,y] -= latent

          ind = 0
          for ni in range(len(gg_xi)):
            for nj in range(len(gg_xi[ni])):
              gg_xi[ni][nj] + me.c * (me.W[ind, yp] - me.W[ind, y])
              ind += 1

        gr_stoc = rmodel.get_stoc_grad(v, *(gg_xi + eps))
        grad_r = param_add(grad_r, gr_stoc)


      grad_g = param_mul_scalar(grad_g, 1.0/me.num_sample)
      grad_r = param_mul_scalar(grad_r, 1.0/me.num_sample)
      grad_w /= me.num_sample

      grad_gs = param_add(grad_gs, grad_g)
      grad_rs = param_add(grad_rs, grad_r)
      
    return (grad_gs, grad_rs, grad_w) 

  def neg_lhood(me, data):
    nlh = 0
    for v in data:
      eps = me.rmodel.sample_eps(v)
      xi = me.rmodel.sample(v, eps)
      nlh -= me.gmodel.get_lhood(v, *xi)
    return nlh

  def test(me, data, label):
    predict = []
    acc = 0
    for (v, lb) in zip(data, label):
      eps = me.rmodel.sample_eps(v)
      xi = me.rmodel.sample(v, eps)
      latent = me.__concat__(xi)
      resp = np.dot(latent, me.W)
      yp = np.argmax(resp)
      predict += [yp]
      if yp == lb:
        acc += 1
    acc /= float(len(v))
    return (predict, acc)

  def sample(me, data):
    recon = []
    for v in data:
      eps = me.rmodel.sample_eps(v)
      recon.append(me.gmodel.sample(me.rmodel.sample(v, eps)))
    return recon
      
  def train(me, data, label, num_iter, test_data = None, test_label = None):
    """
      start the training algorithm.
        > input
          data: N x D data matrix, each row is a data of dimension D.
    """
    printBlue('> Start training neural nets')

    data = np.array(data)
    for it in range(num_iter):
      allind = set(range(data.shape[0]))
      while len(allind) >= me.batchsize:
        "extract mini-batch" 
        ind = npr.choice(list(allind), me.batchsize, replace=False)
        allind -= set(ind)
        V = data[ind, :]
        Y = label[ind]

        "compute gradients"
        # result = mapf(me.process, range(me.num_threads), [V])
        result = mapf(me.process, [0], [V], [Y])

        grad_r = []
        grad_g = []
        grad_w = np.zeros_like(me.W)

        for (ti, res) in enumerate(result):
          grad_g = param_add(grad_g, res[0])
          grad_r = param_add(grad_r, res[1])
          grad_w += res[2]
        
        grad_g = param_mul_scalar(grad_g, 1.0/len(V));
        grad_r = param_mul_scalar(grad_r, 1.0/len(V));
        grad_w /= len(V)


        "aggregate gradients"
        AdaGRAD(me.gmodel.param, grad_g, me.gmodel.G2, me.stepsize)
        AdaGRAD(me.rmodel.param, grad_r, me.rmodel.G2, me.stepsize)
        AdaGRAD([me.W], [grad_w], [me.W_G2], [me.stepsize])

      "evaluate"
      if test_data != None:
        [predict, acc] = me.test(test_data, test_label)
        print 'epoch = ', it, '-lhood', me.neg_lhood(test_data), '-lhood(train)', me.neg_lhood(data), 'test acc', acc
        # print '\tGenerative Model', me.gmodel.pack()
        # print '\tRecognition Model', me.rmodel.pack()
        recon = me.sample(test_data)
        recon_train = me.sample(data)
        sio.savemat('recon.mat', {'recon': recon, 'data':test_data, 'recon_train':recon_train})


    printBlue('> Training complete')

if __name__ == "__main__":
  model = DeepLatentGM([2,4]) 
  model.train(npr.randn(1024,2), 16)
  print 'Generative Model', model.gmodel.pack()
  print 'Recognition Model', model.rmodel.pack()


    



     
    




   



