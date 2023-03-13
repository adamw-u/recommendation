# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import (Zeros, glorot_normal,
                                                  glorot_uniform)
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.layers import utils

class FM(Layer):
    def __init__(self, units, k, **kwargs):
        self.units = units
        self.k = k
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w0 = self.add_weight(name = 'W0', 
                                 shape=(self.units,),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.w = self.add_weight(name = 'W', 
                                 shape=(input_dim, self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.v = self.add_weight(name='V',
                                 shape=(input_dim, self.k),
                                 initializer='glorot_uniform',
                                 trainable=True)

        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        linear_terms = tf.add(tf.matmul(x, self.w), self.w0) #(None, units)
        #tf.matmul(x, self.w) 刚好就是(wi*xi)的累加
        pair_interactions = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(tf.matmul(x, self.v), 2),              #(None, 10) 
                tf.matmul(tf.pow(x, 2), tf.pow(self.v, 2))    #(None, 10)
            ),                                                              
            1, keepdims=True)                                 #(None, 1) 
        #print (pair_interactions.shape, linear_terms.shape)
        output = tf.add(linear_terms, pair_interactions)  
        return output
    def compute_output_shape(self, input_shape):
        return (None,self.units)