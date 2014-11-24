"""
implements the models in paper "stochastic backpropagation in DLGMs"
including
* generative model.
* recognition model.
"""
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as ts

theano.config.exception_verbosity = 'high'

ts.logistic = lambda z: 1 / (1 + ts.exp(-z)) 

class GenerativeModel:
  """ generative model 
  """
  def __init__(me, arch):
    """
      create the deep latent Gaussian model.
        arch: architecture, [vis, hidden_1, hidden_2, ...]
    """
    # set options.
    me.f = ts.maximum         # nonlinear transformation.
    me.lhoodFunc = lambda v, resp: (v * ts.log(ts.logistic(resp)) + (1-v) * ts.log(1-ts.logistic(resp))).sum()

    # set properties.
    me.arch = arch
    me.num_layers = len(arch)
    assert(me.num_layers > 1)

    # init layers.
    (me.G, me.W, me.b, me.xi, me.h) = tuple([[None]*(me.num_layers) for i in range(5)])
    for layer in range(me.num_layers-1, -1, -1):
      if layer < me.num_layers-1:
        me.W[layer] = theano.shared(npr.randn(arch[layer], arch[layer+1]), name="W%d" % layer)
        me.b[layer] = theano.shared(np.zeros(arch[layer]), name="b%d" % layer)
      me.h[layer] = 0
      if layer > 0:
        me.G[layer] = theano.shared(np.eye(arch[layer]), name="G%d" % layer)
        me.xi[layer] = ts.vector("xi%d" % layer)
        me.h[layer] += ts.dot(me.G[layer], me.xi[layer])
      if layer < me.num_layers-1:
        me.h[layer] += ts.dot(me.W[layer], me.f(0, me.h[layer+1])) + me.b[layer]

    # define objective.
    me.v = ts.vector("v")
    me.lhood = me.lhoodFunc(me.v, me.h[0])
    
    # define utils.
    me.generate = theano.function(me.xi[1:], me.h)
    me.hidden_activation = ts.vector("hidden_activiation")
    me.hidden_rectified = me.f(0, me.hidden_activation)
    me.nonlinear = theano.function([me.hidden_activation], me.hidden_rectified)
 

class RecognitionModel:
  """ recognition model (interface)
        since xi \sim \Normal(\mu, C) for each layer. 
        the recognition fits its parameters (\mu, C) discriminatively.

      a simple recognition model uses a two layer NN to fit each parameter.
      see DLGM appendix A.
  """
  def __init__(me, arch, num_hidden=10):
    """
      create the deep latent Gaussian recognition model.
        arch: architecture, [vis, hidden_1, hidden_2, ...]
    """
    # set options.
    me.f = ts.maximum         # nonlinear transformation.

    # set properties.
    me.arch = arch
    me.num_layers = len(arch)
    me.num_hidden = num_hidden
    assert(me.num_layers > 1)

    # init layers.
    me.v = ts.vector("v")
    (me.Wv, me.Wu, me.Wd, me.Wmu, me.bv, me.bu, me.bd, me.bmu, me.z, me.d, me.u, me.mu) \
        = tuple([[None] * me.num_layers for i in range(12)])
    for layer in range(1, me.num_layers):
      me.Wv[layer] = theano.shared(npr.randn(num_hidden, arch[0]), name="Wv%d" % layer)
      me.Wu[layer] = theano.shared(npr.randn(arch[layer], num_hidden), name="Wu%d" % layer)
      me.Wd[layer] = theano.shared(npr.randn(arch[layer], num_hidden), name="Wd%d" % layer)
      me.Wmu[layer] = theano.shared(npr.randn(arch[layer], num_hidden), name="Wmu%d" % layer)
      me.bv[layer] = theano.shared(np.zeros(num_hidden), name="bv%d" % layer)
      me.bu[layer] = theano.shared(np.zeros(arch[layer]), name="bu%d" % layer)
      me.bd[layer] = theano.shared(np.zeros(arch[layer]), name="bd%d" % layer)
      me.bmu[layer] = theano.shared(np.zeros(arch[layer]), name="bmu%d" % layer)
      me.z[layer] =  me.f(0, ts.dot(me.Wv[layer], me.v) + me.bv[layer])
      me.mu[layer] = ts.dot(me.Wmu[layer], me.z[layer]) + me.bmu[layer]
      me.d[layer] = ts.exp(ts.dot(me.Wd[layer], me.z[layer]) + me.bd[layer])
      me.u[layer] = ts.dot(me.Wu[layer], me.z[layer]) + me.bu[layer]

    me.get_mu = theano.function([me.v], me.mu[1:])
    me.get_u = theano.function([me.v], me.u[1:])
    me.get_d = theano.function([me.v], me.d[1:])
    me.get_z = theano.function([me.v], me.z[1:])


    # utils.

    me.sample = lambda v: [npr.multivariate_normal(mu, np.diag(d) + np.matrix(u).T * np.matrix(u)) \
                            for (mu, u, d) in zip(me.get_mu(v), me.get_u(v), me.get_d(v))]

    me.hidden_activation = ts.vector("hidden_activiation")
    me.hidden_rectified = me.f(0, me.hidden_activation)
    me.nonlinear = theano.function([me.hidden_activation], me.hidden_rectified)

      




