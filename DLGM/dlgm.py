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
           'from theano import config',
           'import sys, os',
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
toStr = np.vectorize(str)
strConcat = lambda ls : '  '.join(toStr(ls))

get_value = lambda x: x.get_value() if x != None else None
get_value_all = lambda xs: [get_value(x) for x in xs if x != None]

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
    me.lhoodFunc = lambda v, resp: ts.sum(v * ts.log(ts.logistic(resp)) + (1-v) * ts.log(1-ts.logistic(resp)))

    "set properties."
    me.arch = arch
    me.num_layers = len(arch)
    me.kappa = kappa
    assert(me.num_layers > 1)

    "init layers."
    (me.G, me.W, me.b, me.xi, me.h) = tuple([[None]*(me.num_layers) for i in range(5)])
    for layer in range(me.num_layers-1, -1, -1):
      if layer < me.num_layers-1:
        me.W[layer] = theano.shared(np.asarray(randn01(arch[layer], arch[layer+1]), config.floatX), name="W%d" % layer)
        me.b[layer] = theano.shared(np.asarray(np.zeros((arch[layer], 1)), config.floatX), name="b%d" % layer, broadcastable=(False,True))
      me.h[layer] = 0
      if layer > 0:
        me.G[layer] = theano.shared(np.asarray(np.eye(arch[layer]), config.floatX), name="G%d" % layer)
        me.xi[layer] = ts.matrix("xi%d" % layer)
        me.h[layer] += ts.dot(me.G[layer], me.xi[layer])
      if layer < me.num_layers-1:
        me.h[layer] += ts.dot(me.W[layer], me.f(0, me.h[layer+1])) + me.b[layer]

    me.param = me.G[1:] + me.W[:-1] + me.b[:-1]
    me.G2 = [np.asarray(np.zeros(x.get_value().shape), config.floatX) for x in me.param] # variance of gradient.

    "define objective."
    me.v = ts.matrix("v")
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
    me.generate = theano.function(me.xi[1:], ts.logistic(me.h[0]))
    me.hidden_activation = ts.vector("hidden_activiation")
    me.hidden_rectified = me.f(0, me.hidden_activation)
    me.nonlinear = theano.function([me.hidden_activation], me.hidden_rectified)

  def sample(me, xi):
    resp = me.generate(*xi)
    return toInt(npr.rand(*resp.shape) < resp)

  def activate(me, xi):
    return me.generate(*xi)

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
    me.v = ts.matrix("v") # N x K matrix, N is the sample size, K is the dimension.
    (me.Wv, me.Wu, me.Wd, me.Wmu, me.bv, me.bu, me.bd, me.bmu, me.z, me.d, me.u, me.mu, me.R, me.C) \
        = tuple([[None] * me.num_layers for i in range(14)])
    for layer in range(1, me.num_layers):
      me.Wv[layer] = theano.shared(np.asarray(randn01(num_hidden, arch[0]), config.floatX), name="Wv%d" % layer)
      me.Wu[layer] = theano.shared(np.asarray(randn01(arch[layer], num_hidden), config.floatX), name="Wu%d" % layer)
      me.Wd[layer] = theano.shared(np.asarray(randn01(arch[layer], num_hidden), config.floatX), name="Wd%d" % layer)
      me.Wmu[layer] = theano.shared(np.asarray(randn01(arch[layer], num_hidden), config.floatX), name="Wmu%d" % layer)
      me.bv[layer] = theano.shared(np.asarray(np.zeros((num_hidden, 1)), config.floatX), name="bv%d" % layer, broadcastable=(False, True))
      me.bu[layer] = theano.shared(np.asarray(np.zeros((arch[layer], 1)), config.floatX), name="bu%d" % layer, broadcastable=(False, True))
      me.bd[layer] = theano.shared(np.asarray(np.zeros((arch[layer], 1)), config.floatX), name="bd%d" % layer, broadcastable=(False, True))
      me.bmu[layer] = theano.shared(np.asarray(np.zeros((arch[layer], 1)), config.floatX), name="bmu%d" % layer, broadcastable=(False, True))
      me.z[layer] =  me.f(0, ts.dot(me.Wv[layer], me.v) + me.bv[layer])
      me.mu[layer] = ts.dot(me.Wmu[layer], me.z[layer]) + me.bmu[layer]
      me.d[layer] = ts.exp(ts.dot(me.Wd[layer], me.z[layer]) + me.bd[layer])
      me.u[layer] = ts.dot(me.Wu[layer], me.z[layer]) + me.bu[layer]


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
    eps_s = [ts.matrix('x') for u in me.u] 
    me.Rdot = theano.function([me.v] + eps_s[1:], \
                [1/ts.sqrt(d) * x for (d, x) in zip(me.d[1:], eps_s[1:])] \
                )

    "utils."
    me.get_mu = theano.function([me.v], me.mu[1:])
    me.get_u = theano.function([me.v], me.u[1:])
    me.get_d = theano.function([me.v], me.d[1:])
    me.get_z = theano.function([me.v], me.z[1:])

    me.sample_eps = lambda v: [np.asarray(npr.randn(ac, v.shape[1]), config.floatX) \
                                for ac in arch[1:]]

    me.sample = lambda v, eps: param_add(me.get_mu(v), me.Rdot(v, *eps)) 

    me.hidden_activation = ts.vector("hidden_activiation")
    me.hidden_rectified = me.f(0, me.hidden_activation)
    me.nonlinear = theano.function([me.hidden_activation], me.hidden_rectified)

    "free energy."
    me.energy = 0;
    for layer in range(1, me.num_layers):
      me.energy += me.sigma * (ts.sum(me.mu[layer] * me.mu[layer]) + ts.sum(1/me.d[layer])) \
                    + 0 * ts.sum(me.u[layer] * me.u[layer]) \
                    + ts.sum(ts.log(me.d[layer]))
    me.energy *= 0
    me.get_energy = theano.function([me.v], me.energy)

    "free energy gradients."
    me.param = me.Wv[1:] + me.Wu[1:] + me.Wd[1:] + me.Wmu[1:] + me.bv[1:] + me.bu[1:] + me.bd[1:]+ me.bmu[1:]
    me.G2 = [np.asarray(np.zeros(x.get_value().shape), config.floatX) for x in me.param] # variance of gradient.
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
      me.grad_gm[layer] = ts.matrix('grad_gm_%d' % layer)
      me.eps[layer] = ts.matrix('eps_%d' % layer)
      me.obj_mu += ts.sum(me.mu[layer] * me.grad_gm[layer])
      me.obj_R += .5 * ts.sum(me.grad_gm[layer] * me.eps[layer] / ts.sqrt(me.d[layer])) + 0 * ts.sum(me.u[layer] * \
          me.u[layer])
      # me.obj_R += .5 * (ts.outer(me.grad_gm[layer], me.eps[layer]) * 1/ts.sqrt(me.d[layer])).sum() + 0 * ts.dot(me.u[layer].T,
      #     me.u[layer])
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
        

class DeepLatentGM(object):
  """
    train/test DLGM on datasets.
  """
  def __init__(me, arch, batchsize = 1, num_sample = 1, kappa = 1, sigma = 1, rec_hidden = 100, 
                    stepsize=0.1, num_label=2, ell=100, c = 1, v = 1):
    if os.environ.has_key('hidden'):
      hidden = int(os.environ['hidden'])
      rec_hidden = hidden
      for i in range(1, len(arch)):
        arch[i] = hidden 

    me.arch = arch
    me.kappa = kappa
    me.sigma = sigma
    me.batchsize = batchsize
    me.stepsize = stepsize
    me.num_sample = num_sample

    me.ell = ell
    me.c = c
    me.num_label = num_label
    me.v = 1

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

    printRed(strConcat(['ell = ', me.ell, 'c = ', me.c, 'sigma = ', me.sigma, 'kappa = ', me.kappa, 
                    'stepsize = ', me.stepsize, 'arch = ', strConcat(me.arch)]))
    printBlue('> Compiling neural network')
    me.gmodel = GenerativeModel(me.arch, kappa=me.kappa)
    me.rmodel = RecognitionModel(me.arch, num_hidden=rec_hidden, sigma=me.sigma)

    me.W = np.zeros((sum(arch[1:])+1, me.num_label))
    me.W_G2 = np.zeros_like(me.W)

  def __concat__(me, xi):
    latent = [1]
    for x in xi:
      latent += list(x)
    latent = np.array(latent)
    return latent

  def process(me, V, Y):
    """
      process one single data point.
        > return: (grad of generative model, grad of recognition model)
        > input
          ti: thread id.
          v: data point.
    """
    rmodel = me.rmodel
    gmodel = me.gmodel

    V = np.array(V)
    if len(V.shape) < 2:
      V = np.array([V])

    grad_g = []
    grad_r = []
    grad_w = np.zeros_like(me.W)

    for si in range(me.num_sample):
      "first sample stochastic variables."
      eps = rmodel.sample_eps(V.T)
      xi = rmodel.sample(V.T, eps)

      # pdb.set_trace()
      ta = time.clock()
      "compute gradient of generative model."
      gg = gmodel.get_grad(V.T, *xi)
      gg = param_neg(gg)
      grad_g = param_add(grad_g, gg)

      "compute gradient of regularizer in generative model."
      gg_reg = gmodel.get_grad_reg()
      gg_reg = param_mul_scalar(gg_reg, me.kappa)
      grad_g = param_add(grad_g, gg_reg)

      "compute free-energy gradient of recognition model."
      gr = rmodel.get_grad(V.T)
      grad_r = param_add(grad_r, gr) 

      "compute stochastic gradient of recognition model."
      gg_xi = gmodel.get_grad_xi(V.T, *xi)
      gg_xi = param_neg(gg_xi)

      "add supervision"
      code = rmodel.get_mu(V.T)
      for vi in range(V.shape[0]):
        latent = me.__concat__([c[:, vi] for c in code])
        y = Y[vi]
        
        resp = me.ell + np.dot(latent, me.W) - np.dot(latent, me.W[:,y])
        resp[y] = 0
        yp = np.argmax(resp) 
        grad_w[:,yp] += latent
        grad_w[:,y] -= latent

        # ind = 1   # skip bias.
        # for ni in range(len(gg_xi)):
        #   for nj in range(len(gg_xi[ni])):
        #     gg_xi[ni][nj] += me.c * (me.W[ind, yp] - me.W[ind, y])
        #     ind += 1

      gr_stoc = rmodel.get_stoc_grad(V.T, *(gg_xi + eps))
      grad_r = param_add(grad_r, gr_stoc)

    grad_g = param_mul_scalar(grad_g, 1.0/me.num_sample)
    grad_r = param_mul_scalar(grad_r, 1.0/me.num_sample)
    grad_w /= me.num_sample

    return (grad_g, grad_r, grad_w) 

  def neg_lhood(me, data):
    nlh = 0
    V = np.array(data);
    eps = me.rmodel.sample_eps(V.T)
    xi = me.rmodel.sample(V.T, eps)
    nlh -= me.gmodel.get_lhood(V.T, *xi)
    return nlh

  def test(me, data, label):
    predict = []
    acc = 0
    xi = me.rmodel.get_mu(data.T)
    for (li, lb) in enumerate(label):
      latent = me.__concat__([x[:,li] for x in xi])
      resp = np.dot(latent, me.W)
      yp = np.argmax(resp)
      predict += [yp]
      if yp == lb:
        acc += 1
    acc /= float(len(label))
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

    data = np.array(data).astype(np.float32)
    if test_data != []:
      test_data = np.array(test_data).astype(np.float32)
    label = label.astype(np.float32)
    if test_label != []:
      test_label = test_label.astype(np.float32)

    lhood = []
    test_lhood = []
    recon_err = []
    train_recon_err = []
    accuracy = []

    LAG = 10
    ta = time.time()
    for it in range(num_iter):
      allind = set(range(data.shape[0]))
      while len(allind) >= me.batchsize:
        "extract mini-batch" 
        ind = npr.choice(list(allind), me.batchsize, replace=False)
        allind -= set(ind)
        V = data[ind, :]
        Y = label[ind]

        "compute gradients"

        (grad_g, grad_r, grad_w) = me.process(V, Y)
        
        grad_g = param_mul_scalar(grad_g, 1.0/len(V));
        grad_r = param_mul_scalar(grad_r, 1.0/len(V));
        grad_w /= len(V)

        "aggregate gradients"
        AdaGRAD(me.gmodel.param, grad_g, me.gmodel.G2, me.stepsize)
        AdaGRAD(me.rmodel.param, grad_r, me.rmodel.G2, me.stepsize)
        AdaGRAD([me.W], [grad_w], [me.W_G2], me.stepsize)

      "evaluate"
      if test_data != [] and (it+1) % LAG == 0:
        tb = time.time()
        [predict, acc] = me.test(test_data, test_label)
        accuracy += [acc]
        # print '\tGenerative Model', me.gmodel.pack()
        # print '\tRecognition Model', me.rmodel.pack()
        (recon, xis) = me.reconstruct(test_data)
        recon_err += [np.abs(recon - test_data).sum() / float(test_data.shape[0]) / float(test_data.shape[1])]

        test_lhood += [me.neg_lhood(test_data)]
        lhood += [me.neg_lhood(data)]

        (recon_train, xis_train) = me.reconstruct(data)
        train_recon_err += [np.abs(recon_train - data).sum() / float(data.shape[0]) / float(data.shape[1])]

        time_elapsed = (tb-ta) / float(LAG)

        print 'epoch = ', it, 'time elapsed = ', time_elapsed, '-lhood', test_lhood[-1], '-lhood(train)', lhood[-1], 'test recon err', \
            recon_err[-1], 'train recon err', train_recon_err[-1], 'test acc', acc

        result = {'recon': recon, 'xi': xis, 'xi_train':xis_train, 'data':test_data, 
                    'recon_train':recon_train, 'lhood':lhood, 'test_lhood':test_lhood, 'recon_err':recon_err, 
                    'train_recon_err':train_recon_err, 'test_acc':accuracy, 'time_elapsed':time_elapsed}
        result.update(me.rmodel.pack())
        result.update(me.gmodel.pack())
        sio.savemat('../result/%s/recon.mat' % me.output_path, result)


    with open('../result/%s/log.txt' % me.output_path, "a") as output:
        print >>output, 'epoch = ', it, 'time elapsed = ', time_elapsed, '-lhood', test_lhood[-1], '-lhood(train)', lhood[-1], 'test recon err', \
            recon_err[-1], 'train recon err', train_recon_err[-1], 'test acc', acc
        output.flush()
        output.close()
      
    printBlue('> Training complete')

if __name__ == "__main__":
  model = DeepLatentGM([2,4]) 
  model.train(npr.randn(1024,2), 16)
  print 'Generative Model', model.gmodel.pack()
  print 'Recognition Model', model.rmodel.pack()
