"""
implements the models in paper "stochastic backpropagation in DLGMs"
including
* generative model.
* recognition model.
"""
# let client and server have the same imports.
imports = ['import numpy as np', 
           'import numpy.random as npr', 
           'import theano',
           'import theano.tensor as ts']
for _import in imports:
  exec _import

from IPython.parallel import Client

theano.config.exception_verbosity = 'high'

ts.logistic = lambda z: 1 / (1 + ts.exp(-z)) 

def param_add(param, grad):
  for i in range(len(param)):
    param[i] += grad[i]

def param_mul_scalar(param, scalar):
  for i in range(len(param)):
    param[i] *= scalar

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
    me.get_lhood = theano.function([me.v] + me.xi[1:], me.lhood) 

    # define gradient.
    me.gradient = ts.grad(me.lhood, me.G[1:] + me.W[:-1] + me.b[:-1])
    me.get_grad = theano.function([me.v] + me.xi[1:], me.gradient)

    # define utils.
    me.generate = theano.function(me.xi[1:], me.h)
    me.hidden_activation = ts.vector("hidden_activiation")
    me.hidden_rectified = me.f(0, me.hidden_activation)
    me.nonlinear = theano.function([me.hidden_activation], me.hidden_rectified)

  def pack(me):
    return [x.get_value() for x in [me.G, me.W, me.b]]
  
  def unpack(me, param):
    me.G.set_value(param[0])
    me.W.set_value(param[1])
    me.b.set_value(param[2])

  def zero(me):
    return [0] * 3
    
 
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

  def pack(me):
    return [x.get_value() for x in [me.Wv, me.Wu, me.Wd, me.Wmu, me.bv, me.bu, me.bd, me.bmu]]
  
  def unpack(me, param):
    me.Wv.set_value(param[0])
    me.Wu.set_value(param[1])
    me.Wd.set_value(parma[2])
    me.Wmu.set_value(param[3])
    me.bv.set_value(param[4])
    me.bu.set_value(param[5])
    me.bd.set_value(param[6])
    me.bmu.set_value(param[7])

  def zero(me):
    return [0] * 8
        
class DeepLatentGM:
  """
    train/test DLGM on datasets.
  """
  def __init__(me, arch, batchsize=1, num_sample=1):
    try:
      # parallel
      me.rc = Client()
      me.num_threads = len(me.rc)
      for _import in imports:
        me.rc[:].execute(_import)
      me.view = me.rc.load_balanced_view()
      me.view.block = True
      me.map = me.view.map
    except:
      # cannot connect to parallel server.
      me.num_threads = 1
      me.map = map
    me.batchsize = batchsize
    me.num_sample = num_sample
    me.gmodel = list()
    me.rmodel = list()
    for ni in range(me.num_threads):
      print 'compiling neural network', ni
      me.gmodel += [GenerativeModel(arch)]
      me.rmodel += [RecognitionModel(arch)]


  def process(me, ti, v):
    """
      process one single data point.
        > return: (grad of generative model, grad of recognition model)
        > input
          ti: thread id.
          v: data point.
    """
    rmodel = me.rmodel[ti]
    gmodel = me.gmodel[ti]
    grad_r = rmodel.zero()
    grad_g = gmodel.zero()
    for si in range(me.num_sample):
      # first sample stochastic variables.
      xi = rmodel.sample(v)
      gg = gmodel.get_grad(v, *xi)
      param_add(grad_g, gg)
    param_mul_scalar(grad_g, 1.0/me.num_sample)
    return (grad_g, grad_r) 
      
  def train(me, data, num_iter):
    """
      start the training algorithm.
        > input
          data: N x D data matrix, each row is a data of dimension D.
    """
    data = np.array(data)
    for it in range(num_iter):
      ind = npr.choice(range(data.shape[0]), me.batchsize)
      V = data[ind, :]
      result = me.map(me.process, range(me.num_threads), list(V))
      for (ti, res) in enumerate(result):
        rmodel = me.rmodel[ti]
        gmodel = me.gmodel[ti]
        grad_r = rmodel.zero()
        grad_g = gmodel.zero()
        param_add(grad_g, res[0])
        param_add(grad_r, res[1])
       



if __name__ == "__main__":
  model = DeepLatentGM([2,4,8]) 
  model.train([[1,1]], 1)


    



     
    




   



