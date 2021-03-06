"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import time

try:
  import PIL.Image as Image
except ImportError:
  import Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data

import pdb


# start-snippet-1
class RBM(object):
  """Restricted Boltzmann Machine (RBM)  """
  def __init__(
    self,
    input=None,
    label=None,
    n_visible=784,
    n_hidden=500,
    W=None,
    hbias=None,
    vbias=None,
    numpy_rng=None,
    theano_rng=None,
    c = 1,
    ell = 100, 
    n_class = 10,
  ):
    """
    RBM constructor. Defines the parameters of the model along with
    basic operations for inferring hidden from visible (and vice-versa),
    as well as for performing CD updates.

    :param input: None for standalone RBMs or symbolic variable if RBM is
    part of a larger graph.

    :param n_visible: number of visible units

    :param n_hidden: number of hidden units

    :param W: None for standalone RBMs or symbolic variable pointing to a
    shared weight matrix in case RBM is part of a DBN network; in a DBN,
    the weights are shared between RBMs and layers of a MLP

    :param hbias: None for standalone RBMs or symbolic variable pointing
    to a shared hidden units bias vector in case RBM is part of a
    different network

    :param vbias: None for standalone RBMs or a symbolic variable
    pointing to a shared visible units bias
    """

    self.n_visible = n_visible
    self.n_hidden = n_hidden

    if numpy_rng is None:
      # create a number generator
      numpy_rng = numpy.random.RandomState(1234)

    if theano_rng is None:
      theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    if W is None:
      # W is initialized with `initial_W` which is uniformely
      # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
      # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
      # converted using asarray to dtype theano.config.floatX so
      # that the code is runable on GPU
      initial_W = numpy.asarray(
        numpy_rng.uniform(
          low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
          high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
          size=(n_visible, n_hidden)
        ),
        dtype=theano.config.floatX
      )
      # theano shared variables for weights and biases
      W = theano.shared(value=initial_W, name='W', borrow=True)

    if hbias is None:
      # create shared variable for hidden units bias
      hbias = theano.shared(
        value=numpy.zeros(
          n_hidden,
          dtype=theano.config.floatX
        ),
        name='hbias',
        borrow=True
      )

    if vbias is None:
      # create shared variable for visible units bias
      vbias = theano.shared(
        value=numpy.zeros(
          n_visible,
          dtype=theano.config.floatX
        ),
        name='vbias',
        borrow=True
      )

    # initialize input layer for standalone RBM or layer0 of DBN
    self.input = input
    if not input:
      self.input = T.matrix('input')
    self.label = label
    if not label:
      self.label = T.matrix('label')

    self.W = W
    self.hbias = hbias
    self.vbias = vbias
    self.theano_rng = theano_rng
    # **** WARNING: It is not a good idea to put things in this list
    # other than shared variables created in this function.

    # initialize parameters for supervised learning. 
    self.c = c
    self.ell = 16
    self.weights =  theano.shared(
                      value=numpy.zeros(
                        (n_visible, n_class),
                        dtype=theano.config.floatX
                      ),
                      name='weights',
                      borrow=True
                    )
    # parameter grouping.
    self.params = [self.weights]
    self.G2 = [
                theano.shared(value=numpy.zeros((n_hidden, n_class)), borrow=True)
              ]
    # end-snippet-1

  def free_energy(self, v_sample):
    ''' Function to compute the free energy '''
    wx_b = T.dot(v_sample, self.W) + self.hbias
    vbias_term = T.dot(v_sample, self.vbias)
    hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
    return -hidden_term - vbias_term

  def loss(self, vis, y):
    ell = T.cast(self.ell, dtype=theano.config.floatX)
    true_resp = (T.dot(vis, self.weights) * y).sum(axis=1, keepdims=True)
    T.addbroadcast(true_resp, 1)
    return (self.ell * (1-y) + T.dot(vis, self.weights) - true_resp).max(axis=1).sum()

  def classify(self, vis):
    pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(vis)
    predict = T.dot(ph_mean, self.weights)
    return predict

  def propup(self, vis):
    '''This function propagates the visible units activation upwards to
    the hidden units

    Note that we return also the pre-sigmoid activation of the
    layer. As it will turn out later, due to how Theano deals with
    optimizations, this symbolic variable will be needed to write
    down a more stable computational graph (see details in the
    reconstruction cost function)

    '''
    pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
    return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

  def sample_h_given_v(self, v0_sample):
    ''' This function infers state of hidden units given visible units '''
    # compute the activation of the hidden units given a sample of
    # the visibles
    pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
    # get a sample of the hiddens given their activation
    # Note that theano_rng.binomial returns a symbolic sample of dtype
    # int64 by default. If we want to keep our computations in floatX
    # for the GPU we need to specify to return the dtype floatX
    h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                       n=1, p=h1_mean,
                       dtype=theano.config.floatX)
    return [pre_sigmoid_h1, h1_mean, h1_sample]

  def propdown(self, hid):
    '''This function propagates the hidden units activation downwards to
    the visible units

    Note that we return also the pre_sigmoid_activation of the
    layer. As it will turn out later, due to how Theano deals with
    optimizations, this symbolic variable will be needed to write
    down a more stable computational graph (see details in the
    reconstruction cost function)

    '''
    pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
    return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

  def sample_v_given_h(self, h0_sample):
    ''' This function infers state of visible units given hidden units '''
    # compute the activation of the visible given the hidden sample
    pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
    # get a sample of the visible given their activation
    # Note that theano_rng.binomial returns a symbolic sample of dtype
    # int64 by default. If we want to keep our computations in floatX
    # for the GPU we need to specify to return the dtype floatX
    v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                       n=1, p=v1_mean,
                       dtype=theano.config.floatX)
    return [pre_sigmoid_v1, v1_mean, v1_sample]

  def gibbs_hvh(self, h0_sample):
    ''' This function implements one step of Gibbs sampling,
      starting from the hidden state'''
    pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
    pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
    return [pre_sigmoid_v1, v1_mean, v1_sample,
        pre_sigmoid_h1, h1_mean, h1_sample]

  def gibbs_vhv(self, v0_sample):
    ''' This function implements one step of Gibbs sampling,
      starting from the visible state'''
    pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
    pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
    return [pre_sigmoid_h1, h1_mean, h1_sample,
        pre_sigmoid_v1, v1_mean, v1_sample]

  # start-snippet-2
  def get_cost_updates(self, lr=0.1, persistent=None, k=1, update_method='adagrad'):
    """This functions implements one step of CD-k or PCD-k

    :param lr: learning rate used to train the RBM

    :param persistent: None for CD. For PCD, shared variable
      containing old state of Gibbs chain. This must be a shared
      variable of size (batch size, number of hidden units).

    :param k: number of Gibbs steps to do in CD-k/PCD-k

    Returns a proxy for the cost and the updates dictionary. The
    dictionary contains the update rules for weights and biases but
    also an update of the shared variable used to store the persistent
    chain, if one is used.

    """
    cost = self.c * self.loss(self.input, self.label)
    # We must not compute the gradient through the gibbs sampling
    gparams = T.grad(cost, self.params)
    # end-snippet-3 start-snippet-4
    if update_method == 'sgd':
      # constructs the update dictionary
      for gparam, param in zip(gparams, self.params):
        # make sure that the learning rate is of the right dtype
        updates[param] = param - gparam * T.cast(
          lr,
          dtype=theano.config.floatX
          )
    elif update_method == 'adagrad':
      for gparam, param, g2 in zip(gparams, self.params, self.G2):
        # make sure that the learning rate is of the right dtype
        updates[g2] = g2 + gparam * gparam
        updates[param] = param - gparam * T.cast(lr,     \
                            dtype=theano.config.floatX)  \
                            / (1e-4 + T.sqrt(g2 + gparam * gparam))

    monitoring_cost = 0
    train_err = 0
    return monitoring_cost, train_err, updates
    # end-snippet-4

  def get_error(self, predict, label):
    return T.neq(T.argmax(predict, axis=1), 
                T.argmax(label, axis=1)).sum() / T.cast(label.shape[0], dtype=theano.config.floatX)


  def get_pseudo_likelihood_cost(self, updates):
    """Stochastic approximation to the pseudo-likelihood"""

    # index of bit i in expression p(x_i | x_{\i})
    bit_i_idx = theano.shared(value=0, name='bit_i_idx')

    # binarize the input image by rounding to nearest integer
    xi = T.round(self.input)

    # calculate free energy for the given bit configuration
    fe_xi = self.free_energy(xi)

    # flip bit x_i of matrix xi and preserve all other bits x_{\i}
    # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
    # the result to xi_flip, instead of working in place on xi.
    xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

    # calculate free energy with bit flipped
    fe_xi_flip = self.free_energy(xi_flip)

    # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
    cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                              fe_xi)))

    # increment bit_i_idx % number as part of updates
    updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

    return cost

  def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
    """Approximation to the reconstruction error

    Note that this function requires the pre-sigmoid activation as
    input.  To understand why this is so you need to understand a
    bit about how Theano works. Whenever you compile a Theano
    function, the computational graph that you pass as input gets
    optimized for speed and stability.  This is done by changing
    several parts of the subgraphs with others.  One such
    optimization expresses terms of the form log(sigmoid(x)) in
    terms of softplus.  We need this optimization for the
    cross-entropy since sigmoid of numbers larger than 30. (or
    even less then that) turn to 1. and numbers smaller than
    -30. turn to 0 which in terms will force theano to compute
    log(0) and therefore we will get either -inf or NaN as
    cost. If the value is expressed in terms of softplus we do not
    get this undesirable behaviour. This optimization usually
    works fine, but here we have a special case. The sigmoid is
    applied inside the scan op, while the log is
    outside. Therefore Theano will only see log(scan(..)) instead
    of log(sigmoid(..)) and will not apply the wanted
    optimization. We can not go and replace the sigmoid in scan
    with something else also, because this only needs to be done
    on the last step. Therefore the easiest and more efficient way
    is to get also the pre-sigmoid activation as an output of
    scan, and apply both the log and sigmoid outside scan such
    that Theano can catch and optimize the expression.

    """

    cross_entropy = T.mean(
      T.sum(
        self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
        (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
        axis=1
      )
    )

    return cross_entropy


def test_rbm(learning_rate=1, training_epochs=200,
       dataset='../data/mnist/mnist.pkl.gz', batch_size=32,
       n_chains=20, n_samples=10, output_folder='rbm_plots',
       n_hidden=200, n_example=-1):
  """
  Demonstrate how to train and afterwards sample from it using Theano.

  This is demonstrated on MNIST.

  :param learning_rate: learning rate used for training the RBM

  :param training_epochs: number of epochs used for training

  :param dataset: path the the pickled dataset

  :param batch_size: size of a batch used to train the RBM

  :param n_chains: number of parallel Gibbs chains to be used for sampling

  :param n_samples: number of samples to plot for each chain

  """
  datasets = load_data(dataset, n_example)

  train_set_x, train_set_y = datasets[0]
  test_set_x, test_set_y = datasets[2]

  def convert_to_ind(y, borrow=True):
    y = y.get_value()
    label = numpy.unique(y)
    newy = numpy.zeros((len(y), len(label)))
    for i in range(len(y)):
        newy[i, y[i]] = 1
    sharedy = theano.shared(numpy.asarray(newy,
                                          dtype=theano.config.floatX),
                            borrow=borrow)
    return sharedy

  train_set_y_ind = convert_to_ind(train_set_y)
  test_set_y_ind = convert_to_ind(test_set_y)


  # compute number of minibatches for training, validation and testing
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

  # allocate symbolic variables for the data
  index = T.lscalar()  # index to a [mini]batch
  x = T.matrix('x')  # the data is presented as rasterized images
  y = T.matrix('y')  # the label is a N x C matrix, each row only true class is 1.

  rng = numpy.random.RandomState(123)
  theano_rng = RandomStreams(rng.randint(2 ** 30))

  # initialize storage for the persistent chain (state = hidden
  # layer of chain)
  persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                         dtype=theano.config.floatX),
                   borrow=True)

  # construct the RBM class
  rbm = RBM(input=x, label=y, n_visible=28 * 28,
        n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

  # get the cost and the gradient corresponding to one step of CD-15
  cost, train_err, updates = rbm.get_cost_updates(lr=learning_rate,
                     persistent=persistent_chain, k=15)

  #################################
  #   Training the RBM      #
  #################################
  if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
  os.chdir(output_folder)

  # start-snippet-5
  # it is ok for a theano function to have no output
  # the purpose of train_rbm is solely to update the RBM parameters
  train_rbm = theano.function(
    [index],
    [cost, train_err],
    updates=updates,
    givens={
      x: train_set_x[index * batch_size: (index + 1) * batch_size], 
      y: train_set_y_ind[index * batch_size : (index + 1) * batch_size]
    },
    name='train_rbm'
  )

  tx = T.matrix('tx')  # the data is presented as rasterized images
  ty = T.matrix('ty')  # the label is a N x C matrix, each row only true class is 1.
  predict = rbm.classify(tx)
  test_err = rbm.get_error(predict, ty)
  test_rbm = theano.function(
    [], 
    [predict, test_err],
    givens = {
      tx: test_set_x,
      ty: test_set_y_ind
    },
    name = 'test_rbm'
  )

  plotting_time = 0.
  start_time = time.clock()

  # go through training epochs
  test_err_list = []
  for epoch in xrange(training_epochs):

    # go through the training set
    mean_cost = []
    mean_train_err = []
    for batch_index in xrange(n_train_batches):
      [cost, train_err] = train_rbm(batch_index)
      mean_cost += [cost]
      mean_train_err += [train_err]

    # Test on test set.
    [predict, test_err] = test_rbm()
    test_err_list += [test_err]
    print 'Training epoch %d, cost = %f, test err = %f' % (epoch, numpy.mean(mean_cost), test_err)

  end_time = time.clock()

  pretraining_time = (end_time - start_time) - plotting_time

  print ('Training took %f minutes' % (pretraining_time / 60.))
  # end-snippet-5 start-snippet-6
  #################################
  #   Sampling from the RBM   #
  #################################
  # find out the number of test samples
  number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

  # pick random test examples, with which to initialize the persistent chain
  test_idx = rng.randint(number_of_test_samples - n_chains)
  persistent_vis_chain = theano.shared(
    numpy.asarray(
      test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
      dtype=theano.config.floatX
    )
  )
  # end-snippet-6 start-snippet-7
  plot_every = 1000
  # define one step of Gibbs sampling (mf = mean-field) define a
  # function that does `plot_every` steps before returning the
  # sample for plotting
  (
    [
      presig_hids,
      hid_mfs,
      hid_samples,
      presig_vis,
      vis_mfs,
      vis_samples
    ],
    updates
  ) = theano.scan(
    rbm.gibbs_vhv,
    outputs_info=[None, None, None, None, None, persistent_vis_chain],
    n_steps=plot_every
  )

  # add to updates the shared variable that takes care of our persistent
  # chain :.
  updates.update({persistent_vis_chain: vis_samples[-1]})
  # construct the function that implements our persistent chain.
  # we generate the "mean field" activations for plotting and the actual
  # samples for reinitializing the state of our persistent chain
  sample_fn = theano.function(
    [],
    [
      vis_mfs[-1],
      vis_samples[-1]
    ],
    updates=updates,
    name='sample_fn'
  )

  # create a space to store the image for plotting ( we need to leave
  # room for the tile_spacing as well)
  image_data = numpy.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1),
    dtype='uint8'
  )
  for idx in xrange(n_samples):
    # generate `plot_every` intermediate samples that we discard,
    # because successive samples in the chain are too correlated
    vis_mf, vis_sample = sample_fn()
    print ' ... plotting sample ', idx
    image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
      X=vis_mf,
      img_shape=(28, 28),
      tile_shape=(1, n_chains),
      tile_spacing=(1, 1)
    )

  # construct image
  image = Image.fromarray(image_data)
  image.save('samples.png')
  # end-snippet-7
  os.chdir('../')

if __name__ == '__main__':
  test_rbm(n_example=-1)
  #test_rbm(n_example=100)
