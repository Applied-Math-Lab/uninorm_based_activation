from typing import Any, Mapping, List, Tuple, Optional, Union
import tensorflow as tf
import numpy as np
import random
import sys
import math


# Base class for neural net layers
class Layer(object):
    def __init__(self) -> None:
        self.params: List[tf.Tensor]

    # Repeats a tensor as necessary and crops it to fit the specified output size
    @staticmethod
    def resize(tensor: tf.Tensor, newsize: int) -> tf.Tensor:
        if newsize < tensor.shape[1]:
            return tensor[ : , 0 : newsize]
        elif newsize > tensor.shape[1]:
            multiples = (newsize + int(tensor.shape[1]) - 1) // tensor.shape[1]
            tiled = tf.tile(tensor, [1, multiples])
            if newsize < tiled.shape[1]:
                return tiled[ : , 0 : newsize]
            else:
                return tiled
        else:
            return tensor

    # Gather all the variable values into an object for serialization
    def marshall(self) -> Mapping[str, Any]:
        return { 'params': [ p.numpy().tolist() for p in self.params ] }

    # Load the variables from a deserialized object
    def unmarshall(self, ob: Mapping[str, Any]) -> None:
        params = ob['params']
        if len(params) != len(self.params):
            raise ValueError('Mismatching number of params')
        for i in range(len(params)):
            self.params[i].assign(np.array(params[i]))

    # Returns the number of weights in this layer
    def weightCount(self) -> int:
        wc = 0
        for p in self.params:
            s = 1
            for d in p.shape:
                s *= d
            wc += s
        return wc


# A classic linear (a.k.a. "fully-connected", a.k.a. "dense") layer
class LayerLinear(Layer):
    def __init__(self, inputsize: int, outputsize: int) -> None:
        self.weights = tf.Variable(tf.random.normal([inputsize, outputsize], stddev = max(0.03, 1.0 / inputsize), dtype = tf.float64))
        self.bias = tf.Variable(tf.random.normal([outputsize], stddev = max(0.03, 1.0 / inputsize), dtype = tf.float64))
        self.params = [ self.weights, self.bias ]


    def act(self, x: tf.Tensor) -> tf.Tensor:
        return tf.add(tf.matmul(x, self.weights), self.bias)


# A fuzzy logic layer. Expects twice as many inputs as outputs.
class LayerFuzzyLogic(Layer):
    def __init__(self, outputsize: int) -> None:
        self.alpha = tf.Variable(tf.random.normal([outputsize], stddev = 0.03, dtype = tf.float64))
        self.params = [ self.alpha ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
        # This implements equation 2 from the paper "A parameterized activation function for learning fuzzy logic operations in deep neural networks"
        xx = x[:,:x.shape[1]//2]
        yy = x[:,x.shape[1]//2:]
        veg = (1/50)*np.log((1+np.exp(50*(xx+yy+self.alpha)))/((1+np.exp(50*(xx+yy+self.alpha-1)))))
        return 2*(1 / (80) * tf.math.log((1 + tf.math.exp((80) * ((xx+1)/2 + (yy+1)/2 -  (self.alpha+1)/2))) / (1 + tf.math.exp((80) * ((xx+1)/2 +  (yy+1)/2 - (self.alpha+1)/2 - 1)))))-1


    # Prevents weights from going outside the range [-1,1]
    def clip_weights(self) -> None:
        self.alpha.assign(tf.clip_by_value(self.alpha, -1, 1))


# Makes outputs for each pairing of inputs. Also pairs each input with -1 and 1.
class LayerAllPairings(Layer):
    def __init__(self, inputsize: int) -> None:
        a: List[int] = []
        b: List[int] = []
        for i in range(inputsize):
            for j in range(i + 1, inputsize):
                a.append(i)
                b.append(j)
            a.append(i)
            b.append(inputsize)
            a.append(i)
            b.append(inputsize + 1)
        self.pairings = a + b

    def act(self, x: tf.Tensor) -> tf.Tensor:
        bias = tf.tile(tf.constant([[-1.,1.]], tf.float64), [x.shape[0], 1])
        full_in = tf.concat([x, bias], axis=1)
        return tf.gather(full_in, self.pairings, axis=1)


# A classic linear (a.k.a. "fully-connected", a.k.a. "dense") layer
class LayerFeatureSelector(Layer):
    def __init__(self, inputsize: int, outputsize: int) -> None:
        self.weights = tf.Variable(tf.tile(tf.constant([[1./outputsize]], dtype = tf.float64), [inputsize, outputsize]))
        self.params = [ self.weights ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, self.weights)

    # Prevents weights from going outside the range [-1,1], and regularize them too.
    def clip_weights(self) -> None:
        self.weights.assign(tf.clip_by_value(self.weights - 0.00001 * tf.sign(self.weights), -1, 1))


# Reduces the size of an image by 2 in both dimensions by taking the maximum value
class LayerMaxPooling2d(Layer):
    def __init__(self) -> None:
        self.params = []

    def act(self, x: tf.Tensor) -> tf.Tensor:
        h = x.shape[1]
        w = x.shape[2]
        c = x.shape[3]
        cols = tf.reshape(x, (-1, h, int(w) // 2, 2, c))
        halfwidth = tf.math.maximum(cols[:,:,:,0,:], cols[:,:,:,1,:])
        rows = tf.reshape(halfwidth, (-1, int(h) // 2, 2, int(w) // 2, c))
        halfheight = tf.math.maximum(rows[:,:,0,:,:], rows[:,:,1,:,:])
        return tf.reshape(halfheight, (-1, int(h) // 2, int(w) // 2, c))


# A convolutional layer
class LayerConv(Layer):
    # filter_shape should take the form: (height, width, channels_incoming, channels_outgoing)
    def __init__(self, filter_shape: Tuple[int, ...]):
        spatial_size = 1
        for i in range(0, len(filter_shape) - 2):
            spatial_size *= filter_shape[i]
        self.weights = tf.Variable(tf.random_normal(filter_shape, stddev = 1.0 / spatial_size, dtype = tf.float64))
        self.params = [ self.weights ]

    def act(self, x: tf.Tensor) -> tf.Tensor:
        self.activation = tf.nn.convolution(x, self.weights, "SAME")


# Computes pair-wise products to reduce a vector size by 2
class LayerProductPooling(Layer):
    def __init__(self) -> None:
        self.params = []

    def act(self, x: tf.Tensor) -> tf.Tensor:
        half_size = int(x.shape[1]) // 2
        if int(x.shape[1]) != half_size * 2:
            raise ValueError("Expected an even number of input values")
        two_halves = tf.reshape(x, [-1, 2, half_size])
        return tf.multiply(two_halves[:, 0], two_halves[:, 1])
