# -*- coding: utf-8 -*- 
"""
a bunch of lasagne code implementing gumbel softmax
https://arxiv.org/abs/1611.01144
"""
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
from lasagne.layers import Layer

class GumbelSoftmax:
    """
    A gumbel-softmax nonlinearity with gumbel(0,1) noize
    In short, it's a quasi-one-hot nonlinearity that "samples" from softmax 
    categorical distribution.
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    Code mostly follows http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    
    Softmax normalizes over the LAST axis (works exactly as T.nnet.softmax for 2d).
    
    :param t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
    :param eps: a small number used for numerical stability
    :returns: a callable that can (and should) be used as a nonlinearity
    
    """
    def __init__(self,
                 t=0.1,
                 eps=1e-20):
        assert t != 0
        self.temperature=t
        self.eps=eps
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
         
    def __call__(self,logits):
        """computes a gumbel softmax sample"""
                
        #sample from Gumbel(0, 1)
        uniform = self._srng.uniform(logits.shape,low=0,high=1)
        gumbel = -T.log(-T.log(uniform + self.eps) + self.eps)
        
        #draw a sample from the Gumbel-Softmax distribution
        return T.nnet.softmax((logits + gumbel) / self.temperature)

def onehot_argmax(logits):
    """computes a hard one-hot vector encoding maximum"""
    return T.extra_ops.to_one_hot(T.argmax(logits,-1),logits.shape[-1])

class GumbelSoftmaxLayer(Layer):
    """
    lasagne.layers.GumbelSoftmaxLayer(incoming,**kwargs)
    A layer that just applies a GumbelSoftmax nonlinearity.
    In short, it's a quasi-one-hot nonlinearity that "samples" from softmax 
    categorical distribution.
    
    If you provide "hard_max=True" in lasagne.layers.get_output
    it will instead compute one-hot of aт argmax.
    
    Softmax normalizes over the LAST axis (works exactly as T.nnet.softmax for 2d).
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    Code mostly follows http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    
    
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic (e.g. shared)
    eps: a small number used for numerical stability

    """
    def __init__(self, incoming, t=0.1, eps=1e-20, **kwargs):
        super(GumbelSoftmaxLayer, self).__init__(incoming, **kwargs)
        self.gumbel_softmax = GumbelSoftmax(t=t,eps=eps)

    def get_output_for(self, input, hard_max=False, **kwargs):
        if hard_max:
            return onehot_argmax(input)
        else:
            return self.gumbel_softmax(input)
