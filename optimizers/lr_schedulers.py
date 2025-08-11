import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule, ExponentialDecay
import logging

logger = logging.getLogger("logger")


class HalfCycleLrScheduler(LearningRateSchedule):

    def __init__(self, lr, max_steps, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.lr = lr

    def __call__(self, step):
        lr = self.lr * (1 + tf.math.sin(tf.constant(np.pi, dtype=tf.float64) * step / (self.max_steps * 0.75)))
        return lr

    def get_config(self):
        pass


class ConstantScheduler(LearningRateSchedule):

    def __init__(self, lr, **kwargs):
        super(ConstantScheduler, self).__init__(**kwargs)
        self.lr = lr

    def __call__(self, step):
        return self.lr

    def get_config(self):
        pass


