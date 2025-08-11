import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
import logging
import numpy as np

logger = logging.getLogger("logger")


class DVContinuous(Metric):
    def __init__(self, base=np.e, name='dv_metric', **kwargs):
        super(DVContinuous, self).__init__(name=name, **kwargs)
        self.t = self.add_weight(name='t', initializer='zeros')
        self.exp_t_bar = self.add_weight(name='exp_t_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')
        self.base = tf.constant(base, dtype=tf.float64)

    def update_state(self, t, t_, **kwargs):

        self.t.assign(self.t + tf.reduce_sum(t))
        self.exp_t_bar.assign(self.exp_t_bar + tf.reduce_sum(K.exp(t_)))

        N = tf.cast(tf.reduce_prod(tf.shape(t)[:-1]), tf.float64)
        self.global_counter.assign(self.global_counter + N)
        N_ = tf.cast(tf.reduce_prod(tf.shape(t_)), tf.float64) / tf.cast(tf.shape(t)[-1], tf.float64)
        self.global_counter_ref.assign(self.global_counter_ref + N_)

    def result(self):
        if self.global_counter == 0.0:
            return self.global_counter
        else:
            loss = self.t / self.global_counter - K.log(self.exp_t_bar / self.global_counter_ref)
            return loss / tf.math.log(self.base)


class DI(Metric):
    def __init__(self, base=np.e, name='dv_metric', **kwargs):
        super(DI, self).__init__(name=name, **kwargs)
        self.ty = self.add_weight(name='ty', initializer='zeros')
        self.exp_ty_bar = self.add_weight(name='exp_ty_bar', initializer='zeros')
        self.txy = self.add_weight(name='txy', initializer='zeros')
        self.exp_txy_bar = self.add_weight(name='exp_txy_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')
        self.base = tf.constant(base, dtype=tf.float64)

    def update_state(self, t, t_, **kwargs):
        t_list = tf.split(t, num_or_size_splits=2, axis=-1)
        txy, ty = t_list
        t_list_ = tf.split(t_, num_or_size_splits=2, axis=-1)
        txy_, ty_ = t_list_
        self.ty.assign(self.ty + tf.reduce_sum(ty))
        self.exp_ty_bar.assign(self.exp_ty_bar + tf.reduce_sum(K.exp(ty_)))

        self.txy.assign(self.txy + tf.reduce_sum(txy))
        self.exp_txy_bar.assign(self.exp_txy_bar + tf.reduce_sum(K.exp(txy_)))

        N = tf.cast(tf.reduce_prod(tf.shape(ty)[:-1]), tf.float64)
        self.global_counter.assign(self.global_counter + N)
        N_ = tf.cast(tf.reduce_prod(tf.shape(ty_)), tf.float64) / tf.cast(tf.shape(ty)[-1], tf.float64)
        self.global_counter_ref.assign(self.global_counter_ref + N_)

    def result(self):
        if self.global_counter == 0.0:
            return self.global_counter
        else:
            loss = self.txy / self.global_counter - K.log(self.exp_txy_bar / self.global_counter_ref) - \
                   (self.ty / self.global_counter - K.log(self.exp_ty_bar / self.global_counter_ref))
            return loss * tf.math.log(self.base)
