import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Activation

tf.keras.backend.set_floatx('float32')
dtype = tf.keras.backend.floatx()

class ContrastiveGaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, mean=0.0, std=1.0, **kwargs):
        super(ContrastiveGaussianNoiseLayer, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def call_fn(self, inputs, duplicates):
        output_shape = tf.shape(tf.stack([tf.zeros_like(inputs)
                                          for _ in range(duplicates)], axis=2))
        return tf.random.normal(shape=output_shape, dtype=dtype)

    def __call__(self, inputs, duplicates, training=None, mask=None, *args, **kwargs):
        contrastive_samples = self.call_fn(inputs, duplicates) * self.std + self.mean

        return contrastive_samples

    def reset_states(self):
        pass


class ContrastiveDiscreteUniformNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, minval=0, maxval=1, **kwargs):
        super(ContrastiveDiscreteUniformNoiseLayer, self).__init__(**kwargs)
        self.minval = minval
        self.maxval = maxval

    def call_fn(self, inputs, duplicates):
        output_shape = tf.shape(tf.stack([tf.zeros_like(inputs)
                                          for _ in range(duplicates)], axis=2))
        return tf.cast(tf.random.uniform(shape=output_shape, dtype=tf.int32, minval=self.minval, maxval=self.maxval+1), dtype)

    def __call__(self, inputs, duplicates, training=None, mask=None, *args, **kwargs):
        contrastive_samples = self.call_fn(inputs, duplicates)

        return contrastive_samples

    def reset_states(self):
        pass


class ResnetIdentityBlock(Layer):
    def __init__(self, hidden_size=50, activation='elu', max_norm=2.0):
        super(ResnetIdentityBlock, self).__init__(name='')

        self.activation = Activation(activation)
        self.dense1 = Dense(hidden_size, activation=None, use_bias=True)
        self.bn1 = tf.keras.layers.LayerNormalization()

        self.dense2 = Dense(hidden_size, activation=None, use_bias=True)
        self.bn2 = tf.keras.layers.LayerNormalization()

        self.dense3 = Dense(hidden_size, activation=None, use_bias=True)
        self.bn3 = tf.keras.layers.LayerNormalization()

        self.dense4 = Dense(hidden_size, activation=None, use_bias=True)
        self.bn4 = tf.keras.layers.LayerNormalization()

    def call(self, input_tensor, training=False):
        x = self.dense1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)

        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = self.activation(x)

        x = self.dense4(x)
        x = self.bn4(x, training=training)

        x += input_tensor
        return self.activation(x)


# class SplitEvenOdd(Layer):
#     def __init__(self, axis, name='SplitEvenOdd'):
#         super(SplitEvenOdd, self).__init__(name='SplitEvenOdd')
#         self.axis = axis
#
#     def call(self, inputs):
#         # N = tf.shape(inputs)[1]
#         # odd_indices = tf.tile(
#         #     tf.expand_dims(tf.expand_dims(tf.range(start=0, limit=N, delta=2, dtype=tf.int64), 0), -1),
#         #     [inputs.shape[0], 1, 1])
#         # even_indices = tf.tile(
#         #     tf.expand_dims(tf.expand_dims(tf.range(start=1, limit=N, delta=2, dtype=tf.int64), 0), -1),
#         #     [inputs.shape[0], 1, 1])
#         #
#         # y_odd = tf.gather_nd(inputs, odd_indices, batch_dims=1)
#         # y_even = tf.gather_nd(inputs, even_indices, batch_dims=1)
#         shape = tf.shape(inputs)
#         y_ = tf.reshape(inputs,
#                         tf.concat(([shape[0], shape[1] // 2, 2], shape[2:]), 0))
#         start = tf.zeros(len(shape)+1, tf.int32)
#         y_odd = tf.squeeze(tf.slice(y_,
#                                     start,
#                                     tf.concat(([shape[0], shape[1] // 2, 1], shape[2:]), 0)), axis=2)
#         start = tf.tensor_scatter_nd_update(start, [[2]], [1])
#         y_even = tf.squeeze(tf.slice(y_,
#                                      start,
#                                      tf.concat(([shape[0], shape[1] // 2, 1], shape[2:]), 0)), axis=2)
#         return y_odd, y_even

# class Interleave(Layer):
#     def __init__(self, name='interleave'):
#         super(Interleave, self).__init__(name='interleave')
#
#     def call(self, inputs):
#         # N = tf.shape(inputs)[1]
#         # lower_indices = tf.range(start=0, limit=N // 2)
#         # upper_indices = tf.range(start=N // 2, limit=N)
#         # indices = tf.stack([lower_indices, upper_indices], axis=1)
#         # indices = tf.reshape(indices, [-1, 1])
#         # indices = tf.tile(indices[tf.newaxis, :, :], [inputs.shape[0], 1, 1])
#         # y = tf.gather_nd(inputs, indices, batch_dims=1)
#         shape = tf.shape(inputs)
#
#         y_ = tf.reshape(inputs, tf.concat(([shape[0], 2, shape[1] // 2], shape[2:]), 0))
#         y = tf.reshape(tf.transpose(y_, [0, 2, 1, 3]), tf.shape(inputs))
#         return y
#
#
# class F2(Layer):
#     def __init__(self, name='F2'):
#         super(F2, self).__init__()
#
#     @staticmethod
#     def mod(x, y):
#         x, y = tf.equal(x, 1), tf.equal(y, 1)
#         z = tf.cast(tf.math.logical_xor(x, y), dtype)
#         return z
#
#     def call(self, u1hardprev, u2hardprev):
#         u1hardprev, u2hardprev = tf.cast(u1hardprev, dtype), tf.cast(u2hardprev, dtype)
#         return tf.concat([F2.mod(u1hardprev, u2hardprev), u2hardprev], axis=1)
#
#

class SplitEvenOdd(Layer):
    def __init__(self, axis, name='SplitEvenOdd'):
        super(SplitEvenOdd, self).__init__(name=name)
        self.axis = axis

    def call(self, inputs):
        shape = tf.shape(inputs)
        y_ = tf.reshape(inputs,
                        tf.concat((shape[0:self.axis], [shape[self.axis] // 2, 2], shape[self.axis+1:]), 0))
        start = tf.zeros(len(shape)+1, tf.int32)
        y_odd = tf.squeeze(tf.slice(y_,
                                    start,
                                    tf.concat((shape[0:self.axis], [shape[self.axis] // 2, 1], shape[self.axis+1:]), 0)
                                    ), axis=self.axis+1)
        start = tf.tensor_scatter_nd_update(start, [[self.axis+1]], [1])
        y_even = tf.squeeze(tf.slice(y_,
                                     start,
                                     tf.concat((shape[0:self.axis], [shape[self.axis] // 2, 1], shape[self.axis+1:]), 0)), axis=self.axis+1)
        return y_odd, y_even


class Interleave(Layer):
    def __init__(self, axis=1, name='interleave'):
        super(Interleave, self).__init__(name='interleave')
        self.axis = axis

    def call(self, inputs):
        # N = tf.shape(inputs)[1]
        # lower_indices = tf.range(start=0, limit=N // 2)
        # upper_indices = tf.range(start=N // 2, limit=N)
        # indices = tf.stack([lower_indices, upper_indices], axis=1)
        # indices = tf.reshape(indices, [-1, 1])
        # indices = tf.tile(indices[tf.newaxis, :, :], [inputs.shape[0], 1, 1])
        # y = tf.gather_nd(inputs, indices, batch_dims=1)
        shape = tf.shape(inputs)
        N = shape[self.axis]
        perm = tf.reshape(tf.transpose(tf.reshape(tf.range(N), [2, -1])), [-1])
        out = tf.gather(inputs, perm, axis=self.axis)
        # perm = tf.concat((shape[0:self.axis], [2, shape[self.axis] // 2], shape[self.axis + 1:]), 0)
        #
        # y_ = tf.reshape(inputs, perm)
        #
        # # Indices of elements you want to switch
        # i, j = self.axis, self.axis + 1
        #
        # # Create a tensor with switched elements
        # indices = tf.range(perm.shape[0])
        # updated_indices = tf.tensor_scatter_nd_update(indices, [[i], [j]], [j, i])
        # y = tf.reshape(tf.transpose(y_, updated_indices), tf.shape(inputs))
        return out


class F2(Layer):
    def __init__(self, axis=1, name='F2'):
        super(F2, self).__init__()
        self.axis = axis

    def call(self, inputs):
        u1, u2 = inputs
        return tf.concat([tf.math.floormod(u1 + u2, 2), u2], axis=self.axis)

