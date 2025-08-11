import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class DVContinuousLoss(tf.keras.losses.Loss):
    def __init__(self, base=np.e, name='dv_loss'):
        super(DVContinuousLoss, self).__init__(name=name, reduction='none')
        self.base = tf.constant(base, dtype=tf.float64)

    def call(self, t, t_, **kwargs):
        mean = K.mean(t) - K.log(K.mean(K.exp(t_)))
        return -mean * tf.math.log(self.base)

