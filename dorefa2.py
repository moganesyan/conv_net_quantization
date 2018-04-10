#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from tensorpack.utils.argtools import graph_memoized
import functools
from tensorflow.python.ops import variable_scope


_origin_get_variable = tf.get_variable
_object_stack = []


def _new_get_variable(*args, **kwargs):
    v = _origin_get_variable(*args, **kwargs)
    if len(_object_stack) != 0:
        return _object_stack[-1]._fn(v)
    else:
        return v


class TFVariableReplaceHelper(object):

    def __init__(self, fn):
        self._old_get_variable = None
        self._fn = fn

    def __enter__(self):
        global _object_stack
        _object_stack.append(self)
        self._old_get_variable = tf.get_variable
        tf.get_variable = _new_get_variable
        variable_scope.get_variable = _new_get_variable

    def __exit__(self, *args):
        global _object_stack
        _object_stack.pop()
        tf.get_variable = self._old_get_variable
        variable_scope.get_variable = self._old_get_variable

def replace_variable(fn):
    return TFVariableReplaceHelper(fn)


@graph_memoized
def get_dorefa(bitW, bitA, bitG,phase_train):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def quantize(x, k):
        #n = float(2**k - 1)
        n=tf.cast(2**k-1,tf.float32)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n
        
#    def round_bit(x, bit):
#        if bit == 32:
#            return x
#        g = tf.get_default_graph()
#        k = tf.cast(2**bit - 1,tf.float32)
#        with g.gradient_override_map({'Round': 'Identity'}):
#            return tf.round(x * k) / k

    def fw(x):
#        if bitW == 32:
#            return x
#        G=tf.get_default_graph()
#        if bitW == 1:   # BWN
#            with G.gradient_override_map({"Sign": "Identity"}):
#                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
#                return tf.sign(x)
#        x = tf.tanh(x)
#        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
#        return 2 * round_bit(x, bitW) - 1
        def trn(x):
           # x=tf.Print(x,[phase_train],message="training..")
        
            def func1(x):
                with G.gradient_override_map({"Sign": "Identity"}):
                    E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
                    #x=tf.Print(x,[bitW],message="The bitw 1 is: ") 
                    return tf.sign(x / E) * E
            def func2(x):
                #x=tf.Print(x,[bitW],message="The bitw >1 is: ") 
                x = tf.tanh(x)
                x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
                return 2 * quantize(x, bitW) - 1             
            
            x_f=tf.cond(bitW>tf.constant(1,dtype=tf.float32),lambda:func2(x),lambda:func1(x))
            return x_f
        
        def tst(x):
            x=tf.Print(x,[phase_train],message="testing..")
            return x
        x_f=tf.cond(phase_train,lambda: trn(x), lambda: tst(x))

#        if bitW == 32:
#            return x
#        if bitW == 1:   # BWN
#            with G.gradient_override_map({"Sign": "Identity"}):
#                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
#                return tf.sign(x / E) * E
        #x=tf.Print(x,[x],message="executing here")
#        x = tf.tanh(x)
#        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
#        return 2 * quantize(x, bitW) - 1    
        return x_f
        
        

    def fa(x):
#        if bitA == 32:
#            return x
        return quantize(x, bitA)
#        x = tf.tanh(x)
#        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
#        return 2 * quantize(x, bitA) - 1 
    

    @tf.RegisterGradient("FGGrad")
    def grad_fg(op, x):
        rank = x.get_shape().ndims
        assert rank is not None
        maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
        x = x / maxx
        #n = float(2**bitG - 1)
        n=tf.cast(2**bitG-1,tf.float32)
        x = x * 0.5 + 0.5 + tf.random_uniform(
            tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = quantize(x, bitG) - 0.5
        return x * maxx * 2

    def fg(x):
#        if bitG == 32:
#            #lol=tf.Print(x,[x],message="executing1")
#            return x
        with G.gradient_override_map({"Identity": "FGGrad"}):
            #lol=tf.Print(x,[x],message="after fg",summarize=20)

            return tf.identity(x)
        
#        
#    def round_bit(x, bit):
#        if bit == 32:
#            return x
#        g = tf.get_default_graph()
#        k = tf.cast(2**bit - 1,tf.float32)
#        with g.gradient_override_map({'Round': 'Identity'}):
#            return tf.round(x * k) / k
#    
#    
#    _grad_defined = False
#    if not _grad_defined:
#        @tf.RegisterGradient("IdentityMaxMinGrad")
#        def _identigy_max_min_grad(op, grad):
#            return grad, None
#    
#    
#    def quantize_w(x, bit):
#            if bit == 32:
#                return x
#            G=tf.get_default_graph()
#            if bit == 1:   # BWN
#                with G.gradient_override_map({"Sign": "Identity"}):
#                    E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
#                    return tf.sign(x)
#            x = tf.tanh(x)
#            x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
#            return 2 * round_bit(x, bit) - 1       
    
    return fw, fa, fg




#def round_bit(x, bit):
#    if bit == 32:
#        return x
#    g = tf.get_default_graph()
#    k = tf.cast(2**bit - 1,tf.float32)
#    with g.gradient_override_map({'Round': 'Identity'}):
#        return tf.round(x * k) / k
#
#
##_grad_defined = False
##if not _grad_defined:
##    @tf.RegisterGradient("IdentityMaxMinGrad")
##    def _identigy_max_min_grad(op, grad):
##        return grad, None
#
#
#def quantize_w(x, bit):
#        if bit == 32:
#            return x
#        G=tf.get_default_graph()
#        if bit == 1:   # BWN
#            with G.gradient_override_map({"Sign": "Identity"}):
#                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
#                return tf.sign(x)
#        x = tf.tanh(x)
#        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
#        return 2 * round_bit(x, bit) - 1

                              

#round_bit_1bit = functools.partial(round_bit, bit=1)
#round_bit_2bit = functools.partial(round_bit, bit=2)
#round_bit_3bit = functools.partial(round_bit, bit=3)
#round_bit_4bit = functools.partial(round_bit, bit=4)
#
#quantize_w_1bit = functools.partial(quantize_w, bit=1)
#quantize_w_2bit = functools.partial(quantize_w, bit=2)
#quantize_w_3bit = functools.partial(quantize_w, bit=3)
#quantize_w_4bit = functools.partial(quantize_w, bit=4)




#    def quantize(x, k):
#        #n = float(2**k - 1)
#        n=tf.cast(2**k-1,tf.float32)
#        with G.gradient_override_map({"Round": "Identity"}):
#            return tf.round(x * n) / n
#
#    def fw(x):
#        if bitW == 32:
#            return x
#        if bitW == 1:   # BWN
#            with G.gradient_override_map({"Sign": "Identity"}):
#                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
#                return tf.sign(x / E) * E
#        x = tf.tanh(x)
#        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
#        return 2 * quantize(x, bitW) - 1