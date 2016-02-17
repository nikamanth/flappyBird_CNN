import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def conv_batch_relu_pool_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param):
  """
  Convenience layer that performs a convolution, batchNorm, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - gamma, beta: batch norm weights
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  b, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(b)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache


def conv_batch_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-batchNorm-relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  db = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(db, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta


def affine_batch_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine-batchNorm-ReLU layer

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta: batch norm weights

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  b, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_batch_relu_backward(dout, cache):
  """
  Backward pass for the affine-batch-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  db = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward(db, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta


class CustomConvNet(object):
  """
  A four-layer convolutional network with the following architecture:
  
  It is a [conv-batchNorm-relu-pool]x2 - [affine-batchNorm-relu] - [affine] - [softmax] architecture
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    self.params['W1'] = weight_scale * np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(num_filters,num_filters,filter_size,filter_size)
    self.params['b2'] = np.zeros(num_filters)
    self.params['W3'] = weight_scale * np.random.randn(input_dim[1]/4*input_dim[2]/4*num_filters,hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['W4'] = weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params['b4'] = np.zeros(num_classes)

    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    self.params['gamma3'] = np.ones(hidden_dim)
    self.params['beta3'] = np.zeros(hidden_dim)

    self.bn_params = [{'mode': 'train'} for i in xrange(3)]


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the four-layer convolutional network.
    
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2'] 
    gamma3, beta3 = self.params['gamma3'], self.params['beta3'] 

    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    out,cache_L1 = conv_batch_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, self.bn_params[0])
    out,cache_L2 = conv_batch_relu_pool_forward(out, W2, b2, gamma2, beta2, conv_param, pool_param, self.bn_params[1])
    out,cache_L3 = affine_batch_relu_forward(out, W3, b3, gamma3, beta3, self.bn_params[2])
    scores,cache_L4 = affine_forward(out,W4,b4)

    if y is None:
      return scores
    
    loss, grads = 0, {}

    loss,dscores = softmax_loss(scores,y)
    loss += 0.5 * self.reg * sum([np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)]) for i in range(4)])

    dout,grads['W4'],grads['b4'] = affine_backward(dscores, cache_L4)
    grads['W4'] += self.reg * self.params['W4']
    dout, grads['W3'], grads['b3'],grads['gamma3'], grads['beta3'] = affine_batch_relu_backward(dout, cache_L3)
    grads['W3'] += self.reg * self.params['W3']
    dout, grads['W2'],grads['b2'],grads['gamma2'],grads['beta2'] = conv_batch_relu_pool_backward(dout, cache_L2)
    grads['W2'] += self.reg * self.params['W2']
    dX, grads['W1'],grads['b1'],grads['gamma1'],grads['beta1'] = conv_batch_relu_pool_backward(dout, cache_L1)
    grads['W1'] += self.reg * self.params['W1']
  
    return loss, grads
  
  
pass
