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
           'import sys, os',
           'import scipy.io as sio', 
           'import theano.sandbox.linalg as ta',
           'from theano.tensor.shared_randomstreams import RandomStreams as trng',
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
toStr = np.vectorize(str)

get_value = lambda x: x.get_value() if x != None else None
get_value_all = lambda xs: [get_value(x) for x in xs]

nonlinear_f = lambda x : ts.log(1+ts.exp(x))  # smooth ReLU.
nonlinear_s = "smooth ReLU"
if os.environ.has_key('nonlinear'):
  nonlinear_s = os.environ['nonlinear']
  if nonlinear_s == "ReLU":
    f = lambda x : ts.maximum(0, x) # ReLU.
  if nonlinear_s == "tanh":
    f = lambda x : ts.tanh(x)

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
      p.set_value(p.get_value() - stepsize * g / (1e-8 + np.sqrt(g2))) 
    elif type(p) == np.ndarray:
      p[:] = p[:] - stepsize * g[:] / (1e-8 + np.sqrt(g2[:]))

class Decoder:
  """ generative model 
  """
  def __init__(me, arch, kappa=0.1):
    """
      create the variational decoder.
        arch: architecture, [vis, hidden]
    """
    "set options."
    me.f = ts.maximum         # nonlinear transformation.

    "set properties."
    me.arch = arch
    me.num_layers = len(arch)
    me.kappa = kappa
    assert(me.num_layers > 1)

    "init layers."
    me.G = theano.shared(np.eye(arch[1]), name="G")
    me.xi = ts.matrix("xi")
    me.h = ts.dot(me.G, me.xi)

    def one_layer_logistic(h):
      """
      one layer network for decoding
      """
      W1 = theano.shared(randn01(arch[0], arch[1]), name="W1")
      b1 = theano.shared(np.zeros((arch[0], 1)), name="b1", broadcastable=(False,True))
      resp = ts.dot(W1, h) + b1
      return ( resp,
               [W1, b1], 
               lambda v : (v * ts.log(ts.logistic(resp)) + (1-v) * ts.log(1-ts.logistic(resp))).sum()
             )

    def two_layer_logistic(h):
      """
      two-layer network for decoding
      """
      hidden = 100
      W1 = theano.shared(randn01(hidden, arch[1]), name="W1")
      b1 = theano.shared(np.zeros((hidden, 1)), name="b1", broadcastable=(False,True))
      W2 = theano.shared(randn01(arch[0], hidden), name="W2")
      b2 = theano.shared(np.zeros((arch[0], 1)), name="b2", broadcastable=(False,True))
      u = nonlinear_f(ts.dot(W1, h) + b1)
      resp = ts.dot(W2, u) + b2
      return ( resp,
               [W1, b1, W2, b2], 
               lambda v : (v * ts.log(ts.logistic(resp)) + (1-v) * ts.log(1-ts.logistic(resp))).sum()
             )
    
    (me.resp, me.param, me.lhoodFunc) = one_layer_logistic(me.h)
    #(me.resp, me.param, me.lhoodFunc) = two_layer_logistic(me.h)

    me.param += [me.G]
    me.G2 = [np.zeros(x.get_value().shape) for x in me.param] # variance of gradient.

    "define objective."
    me.v = ts.matrix("v")
    me.lhood = me.lhoodFunc(me.v)
    me.get_lhood = theano.function([me.v, me.xi], me.lhood) 
    me.reg = me.kappa * sum([ts.sum(p * p) for p in me.param])
    me.get_reg = theano.function([], me.reg)

    "define gradient."
    me.gradient = ts.grad(me.lhood, me.param)
    me.gradient_xi = ts.grad(me.lhood, me.xi)
    # me.hessian_xi = ts.hessian(me.lhood, me.xi[1:])
    me.get_grad = theano.function([me.v, me.xi], me.gradient)
    me.get_grad_xi = theano.function([me.v, me.xi], me.gradient_xi)
    # me.get_hess_xi = theano.function([me.v] + me.xi[1:], me.hessian_xi)
    me.gradient_reg = ts.grad(me.reg, me.param)
    me.get_grad_reg = theano.function([], me.gradient_reg)

    "define utils."
    me.generate = theano.function([me.xi], me.resp)

  def sample(me, xi):
    resp = me.activate(xi)
    return toInt(npr.rand(*resp.shape) < resp)
  
  def reconstruct(me, xi):
    resp = me.activate(xi)
    return toInt(np.ones(resp.shape) * 0.5 < resp)
  
  def activate(me, xi):
    resp = me.generate(xi)
    return np.logistic(resp)


  def pack(me):
    param = dict()
    for p in me.param:
      param.update({str(p): p.get_value()})
    return param
 
class Encoder:
  """ recognition model (encoder)
        since xi \sim \Normal(\mu, C) for each layer. 
        the recognition fits its parameters (\mu, C) discriminatively.

      a simple recognition model uses a two layer NN to fit each parameter.
      see DLGM appendix A.
  """
  def __init__(me, arch, sigma=1):
    """
      create the deep latent Gaussian recognition model.
        arch: architecture, [vis, hidden_1, hidden_2, ...]
    """
    "set options."
    me.f = ts.maximum         # nonlinear transformation.

    "set properties."
    me.arch = arch
    me.num_layers = len(arch)
    me.sigma = sigma
    assert(me.num_layers > 1)

    "init layers."
    me.v = ts.matrix("v")

    def two_layer_recognition(v):
      num_hidden = me.arch[1]
      Wv = theano.shared(randn01(num_hidden, arch[0]), name="Wv")
      bv = theano.shared(np.zeros((num_hidden, 1)), name="bv", broadcastable=(False,True))
      Wmu = theano.shared(randn01(arch[1], num_hidden), name="Wmu")
      bmu = theano.shared(np.zeros((arch[1], 1)), name="bmu", broadcastable=(False, True))
      Wd = theano.shared(randn01(arch[1], num_hidden), name="Wd")
      bd = theano.shared(np.zeros((arch[1], 1)), name="bd", broadcastable=(False, True))
      z =  nonlinear_f(ts.dot(Wv, v) + bv)
      d = ts.exp(ts.dot(Wd, z) + bd)
      mu = ts.dot(Wmu, z) + bmu
      xs = ts.matrix('x')
      return (mu, 
              d,
              [Wv, bv, Wmu, bmu, Wd, bd], 
              theano.function([v, xs], mu + 1/ts.sqrt(d) * xs)
             )

    me.sample_eps = lambda V: npr.normal(0, 1, (arch[1], V.shape[1]))
    (me.mu, me.d, me.param, me.sample) = two_layer_recognition(me.v)
    me.G2 = [np.zeros(x.get_value().shape) for x in me.param] # variance of gradient.
    me.get_mu = theano.function([me.v], me.mu)
    me.get_d = theano.function([me.v], me.d)
     
    "free energy and gradients."
    me.energy = 0;
    for layer in range(1, me.num_layers):
      me.energy += .5 * me.sigma * ((me.mu * me.mu).sum() + ts.sum(1/me.d) + ts.sum(ts.log(me.d)))
    me.get_energy = theano.function([me.v], me.energy)
    me.gradient = ts.grad(me.energy, me.param)
    me.get_grad = theano.function([me.v], me.gradient)
   
    """ stochastic gradients.
        trick. pretend our objective is inner product with the stochastic gradients.
    """
    me.grad_gm = ts.matrix('grad_gm')
    me.eps = ts.matrix('eps')
    me.obj_mu = -ts.sum(me.mu * me.grad_gm)
    me.obj_R = -.5 * ts.sum(me.grad_gm * me.eps * 1/ts.sqrt(me.d))
    me.stoc_grad = ts.grad(me.obj_mu + me.obj_R, me.param)
    me.get_stoc_grad = theano.function([me.v] + [me.grad_gm] + [me.eps], me.stoc_grad)
    
  def pack(me):
    param = dict()
    for p in me.param:
      param.update({str(p): p.get_value()})
    return param
        

"parallel, if server is reachable; otherwise, use map."
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

class AutoEncoder(object):
  """
    train/test DLGM on datasets.
  """
  def __init__(me, arch, batchsize = 1, num_sample = 1, kappa = 1, sigma = 1, 
                    stepsize=0.1, num_label=2, ell=10, c = 1, v = 1):

    if os.environ.has_key('hidden'):
      arch[1] = int(os.environ['hidden'])
    me.num_threads = num_threads
    printBlue('> Thread Pool (%d)' % me.num_threads)
    me.arch = arch
    me.kappa = kappa
    me.sigma = sigma
    me.batchsize = batchsize
    me.stepsize = stepsize
    me.num_sample = num_sample

    me.ell = ell
    me.c = c
    me.num_label = num_label

    if os.environ.has_key('ell'):
      me.ell = float(os.environ['ell'])
    if os.environ.has_key('c'):
      me.c = float(os.environ['c'])
    if os.environ.has_key('kappa'):
      me.kappa = float(os.environ['kappa'])
    if os.environ.has_key('sigma'):
      me.sigma = float(os.environ['sigma'])
    if os.environ.has_key('stepsize'):
      me.stepsize = float(os.environ['stepsize'])
    me.stepsize_w = me.stepsize
    if os.environ.has_key('stepsize_w'):
      me.stepsize_w = float(os.environ['stepsize_w'])
    if os.environ.has_key('output'):
      me.output_path = os.environ['output']
    else:
      me.output_path = 'default'
    print 'ell = ', me.ell, 'c = ', me.c, 'sigma = ', me.sigma, 'kappa = ', me.kappa, \
          'stepsize = ', me.stepsize, 'arch = ', me.arch
    print 'nonlinear_f = ', nonlinear_s

    printBlue('> Compiling neural network')
    me.W = np.zeros((sum(me.arch[1:])+1, me.num_label))
    me.W_G2 = np.zeros_like(me.W)
    me.gmodel = Decoder(me.arch, kappa=me.kappa)
    me.rmodel = Encoder(me.arch, sigma=me.sigma)


  def __concat__(me, xi):
    latent = [1]
    latent += list(xi)
    latent = np.array(latent)
    return latent

  def process(me, ti, V, Y = []):
    """
      process one single data point.
        > return: (grad of generative model, grad of recognition model)
        > input
          ti: thread id.
          v: data point.
    """
    rmodel = me.rmodel
    gmodel = me.gmodel

    grad_g = []
    grad_r = []
    grad_w = np.zeros_like(me.W)

    for si in range(me.num_sample):
      "first sample stochastic variables."
      "eps is randomness for recognition model, xi is randomness for generative model"
      eps = rmodel.sample_eps(V)
      xi = rmodel.sample(V, eps)

      "compute gradient of generative model."
      gg = gmodel.get_grad(V, xi)
      gg = param_neg(gg)
      grad_g = param_add(grad_g, gg)

      "compute gradient of regularizer in generative model."
      gg_reg = gmodel.get_grad_reg()
      grad_g = param_add(grad_g, gg_reg)

      "compute free-energy gradient of recognition model."
      gr = rmodel.get_grad(V)
      grad_r = param_add(grad_r, gr) 

      "compute stochastic gradient of recognition model."
      gg_xi = gmodel.get_grad_xi(V, xi)

      "add supervision"
      if Y != []:
        # latents = rmodel.get_mu(V)
        latents = xi
        for (ni, (y, latent)) in enumerate(zip(Y, latents.T)):
          latent = me.__concat__(latent)
          resp = me.ell + np.dot(latent, me.W) - np.dot(latent, me.W[:,y])
          resp[y] = 0
          yp = np.argmax(resp) 
          grad_w[:,yp] += latent
          grad_w[:,y] -= latent

          gg_xi[:, ni] -= me.c * (me.W[1:, yp] - me.W[1:, y])

      gr_stoc = rmodel.get_stoc_grad(V, gg_xi, eps)
      grad_r = param_add(grad_r, gr_stoc)

      grad_g = param_mul_scalar(grad_g, 1.0/me.num_sample)
      grad_r = param_mul_scalar(grad_r, 1.0/me.num_sample)
      grad_w /= me.num_sample

    return (grad_g, grad_r, grad_w) 

  def neg_lhood(me, data):
    eps = me.rmodel.sample_eps(data.T)
    xi = me.rmodel.sample(data.T, eps)
    nlh = -me.gmodel.get_lhood(data.T, xi)
    return nlh

  def test(me, data, label):
    predict = []
    acc = 0
    # eps = me.rmodel.sample_eps(data.T)
    # xi = me.rmodel.sample(data.T, eps).T        # use posterior mean to make predictions.
    xi = me.rmodel.get_mu(data.T).T
    for (v, lb, x) in zip(data, label, xi):
      # eps = me.rmodel.sample_eps(v)
      # xi = me.rmodel.sample(v, eps)
      latent = me.__concat__(x)
      resp = np.dot(latent, me.W)
      yp = np.argmax(resp)
      predict += [yp]
      if yp == lb:
        acc += 1
    acc /= float(len(data))
    return (predict, acc)

  def reconstruct(me, data):
    eps = me.rmodel.sample_eps(data.T)
    xi = me.rmodel.sample(data.T, eps)
    recon = me.gmodel.activate(xi).T
    return (recon, xi)
      
  def train(me, data, label, num_iter, test_data = [], test_label = []):
    """
      start the training algorithm.
        > input
          data: N x D data matrix, each row is a data of dimension D.
    """
    printBlue('> Start training neural nets')

    os.system('mkdir -p ../result/%s' % me.output_path)

    data = np.array(data)
    lhood = []
    test_lhood = []
    recon_err = []
    test_recon_err = []
    train_recon_err = []
    accuracy = []

    for it in range(num_iter):
      allind = set(range(data.shape[0]))
      while len(allind) >= me.batchsize:
        "extract mini-batch" 
        ind = npr.choice(list(allind), me.batchsize, replace=False)
        allind -= set(ind)
        V = data[ind, :].T
        Y = label[ind]

        "compute gradients"
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
        AdaGRAD([me.W], [grad_w], [me.W_G2], me.stepsize_w)

      "evaluate"
      if test_data != [] and (it+1) % 10 == 0:
        [predict, acc] = me.test(test_data, test_label)
        accuracy += [acc]
        # print '\tGenerative Model', me.gmodel.pack()
        # print '\tRecognition Model', me.rmodel.pack()
        (recon, xi) = me.reconstruct(test_data)
        recon_err += [np.abs(recon - test_data).sum() / float(test_data.shape[0]) / float(test_data.shape[1])]

        test_lhood += [me.neg_lhood(test_data)]
        lhood += [me.neg_lhood(data)]

        (recon_train, xi_train) = me.reconstruct(data)
        train_recon_err += [np.abs(recon_train - data).sum() / float(data.shape[0]) / float(data.shape[1])]

        print 'epoch = ', it, '-lhood', test_lhood[-1], '-lhood(train)', lhood[-1], 'test recon err', \
            recon_err[-1], 'train recon err', train_recon_err[-1], 'test acc', acc

        sio.savemat('../result/%s/recon.mat' % me.output_path, {'recon': recon, 'xi': xi, 'xi_train':xi_train, 'data':test_data, 
                    'recon_train':recon_train, 'lhood':lhood, 'test_lhood':test_lhood, 'recon_err':recon_err, 
                    'train_recon_err':train_recon_err, 'test_acc':accuracy})

    with open('../result/%s/log.txt' % me.output_path, "a") as output:
      output.write('\n')
      output.write(' '.join(toStr(['ell = ', me.ell, 'c = ', me.c, 'sigma = ', me.sigma, 'kappa = ', me.kappa, \
                    'stepsize = ', me.stepsize, 'arch = ', me.arch[0], me.arch[1]]))+'\n')
      output.write(' '.join(toStr(['nonlinear_f = ', nonlinear_s]))+'\n')
      output.write(' '.join(toStr(['epoch = ', it, '-lhood', test_lhood[-1], '-lhood(train)', lhood[-1],  
                    'test recon err', recon_err[-1], 'test acc', acc]))+'\n')
      output.flush()
      output.close()



    printBlue('> Training complete')

if __name__ == "__main__":
  model = DeepLatentGM([2,4]) 
  model.train(npr.randn(1024,2), 16)
  print 'Generative Model', model.gmodel.pack()
  print 'Recognition Model', model.rmodel.pack()
