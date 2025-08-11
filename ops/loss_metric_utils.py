import pickle
import numpy as np
import scipy
import os
import time
from datetime import datetime
from collections import defaultdict as def_dict
import tensorflow as tf
import logging

logger = logging.getLogger("logger")

# weights methods

def nan_mask(x):
    return tf.logical_not(tf.math.is_nan(tf.cast(x, tf.float64)))

def zero_mask(x):
    return tf.equal(x, tf.zeros_like(x))

def not_zero_mask(x):
    return tf.not_equal(x, tf.zeros_like(x))

def one_mask(x):
    return tf.equal(x, tf.ones_like(x))

def not_one_mask(x):
    return tf.not_equal(x, tf.ones_like(x))

def nan_zero_mask(x):
    return tf.logical_and(nan_mask(x), zero_mask(x))

def nan_not_zero_mask(x):
    return tf.logical_and(nan_mask(x), not_zero_mask(x))

def nan_one_mask(x):
    return tf.logical_and(nan_mask(x), one_mask(x))

def nan_not_one_mask(x):
    return tf.logical_and(nan_mask(x), not_one_mask(x))

def subtract_exp(x):
    x_1, x_2 = tf.split(x, axis=-1, num_or_size_splits=2)
    return tf.math.exp(x_1-x_2)

def first_exp(x):
    x_1, x_2 = tf.split(x, axis=-1, num_or_size_splits=2)
    return tf.math.exp(x_1)

def second_exp(x):
    x_1, x_2 = tf.split(x, axis=-1, num_or_size_splits=2)
    return tf.math.exp(x_2)

# target methods
def remove_nan(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def remove_nan_ones(x):
    return tf.where(nan_one_mask(x), x, tf.ones_like(x))

def tar_ce(x):
    return remove_nan(x + 1)

def tar_hinge(x):
    return remove_nan(x)

# pred methods
def identity(x):
    return x

def split_identity_0(x):
    x = tf.split(x, num_or_size_splits=2, axis=-1)
    return x[0]

def split_identity_1(x):
    x = tf.split(x, num_or_size_splits=2, axis=-1)
    return x[1]

# def identity_ce(*x):
#     return x[0]
#
# def identity_hinge(*x):
#     return x[1]
#
# def argmax_ce(*x):
#     return tf.argmax(x[0], axis=-1)
#
# def argmax_hinge(*x):
#     return tf.where(x[1] > 0.0, tf.ones_like(x[1]), -tf.ones_like(x[1]))
#
# def predict_big_change(*y):
#     x = y[0]
#     return x * tf.concat([tf.ones([x.shape[0],x.shape[1],1]),
#                           tf.zeros([x.shape[0],x.shape[1],1]),
#                           tf.ones([x.shape[0],x.shape[1],1])], axis=-1)