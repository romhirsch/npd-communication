from abc import abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.python.keras import initializers

tf.keras.backend.set_floatx('float32')
dtype = tf.keras.backend.floatx()


class Channel(Model):
    def __init__(self, *args, **kwargs):
        super(Channel, self).__init__()
        self.save_name = f"channel"
        self.cardinality_x = self.cardinality_s = 2
        self.cardinality_y = 1

    def get_config(self):
        pass

    def kernel(self, *args):
        return

    def call(self, inputs, training=None, *args, **kwargs):
        y = self.kernel(inputs)
        return y

    def llr(self, y):
        return y

    @staticmethod
    def sample_noise(batch_size, p):
        p_t = p * tf.ones(shape=[batch_size, 1], dtype=p.dtype)
        p_bar_t = 1 - p_t
        logits = tf.math.log(tf.concat([p_bar_t, p_t], axis=-1))
        noise = tf.cast(tf.random.categorical(logits=logits, num_samples=1), dtype)
        return noise

    def sample_channel_outputs(self, x):
        y = self.kernel(x)
        return y


class ChannelMemory(Channel):
    def __init__(self, batch_size, dim=1, **kwargs):
        super(ChannelMemory, self).__init__()
        self.state = self.add_weight(name="state", shape=(batch_size,  dim), dtype=dtype, trainable=False,
                                     initializer=initializers.constant(0.0))
        self.cardinality_x = self.cardinality_s = 2

    def get_config(self):
        pass

    def __call__(self, inputs, training=None, **kwargs):
        x = tf.cast(inputs, dtype)
        y = self.kernel(x)
        return y

    def reset_state(self):
        self.state.assign(tf.zeros_like(self.state))

    def sample_channel_outputs(self, x):
        x_list = tf.split(x, num_or_size_splits=x.shape[1], axis=1)
        y = tf.TensorArray(dtype=tf.float32, size=int(x.shape[1]))
        for i, xx in enumerate(x_list):
            y = y.write(i, self.kernel(tf.squeeze(xx, axis=1)))
        y = tf.transpose(y.stack(), perm=[1, 0, 2])
        return y


class AWGN(Channel):
    """
    """

    def __init__(self, var=1.0, mean=0.0):
        super(AWGN, self).__init__()
        self.var = tf.constant(var, dtype=dtype)
        self.mean = tf.constant(mean, dtype=dtype)

        self.save_name = f"snr-{1. / var}-awgn"

    def kernel(self, x):
        return x * 2. - 1. + tf.sqrt(self.var) * tf.random.normal(tf.shape(x), dtype=dtype)

    def llr(self, y):
        return 2.0 * y / self.var


class BSC(Channel):
    """
    """

    def __init__(self, p=0.1):
        super(BSC, self).__init__(name='bsc')
        self.p = tf.constant(p, dtype=dtype)

    def kernel(self, x):
        logits = tf.tile(tf.math.log([[1 - self.p, self.p]]), [tf.shape(x)[0], 1])
        noise = tf.random.categorical(logits, tf.shape(x)[1])
        y = tf.math.floormod(tf.cast(noise, dtype) + x, 2)
        return y

    def llr(self, y):
        return tf.math.log((1 - self.p) / self.p) * (2 * y - 1.)


class Z(Channel):
    """
    """
    def __init__(self, p=0.5):
        super(Z, self).__init__(name='z')
        self.p = tf.constant(p, dtype=dtype)

    def kernel(self, x):
        logits = tf.tile(tf.math.log([[1 - self.p, self.p]]), [tf.shape(x)[0], 1])
        noise = tf.random.categorical(logits, tf.shape(x)[1])
        y = tf.where(tf.equal(x, 0), x, tf.math.floormod(x + noise, 2))
        return y

    def llr(self, y):
        return tf.math.log(1/self.p) * (1 - y) - 10 * y


class BEC(Channel):
    """
    """
    def __init__(self, p0=0.4, p1=0.8159):
        super(BEC, self).__init__(name='bec')
        self.p0 = tf.constant(p0, dtype=dtype)
        self.p1 = tf.constant(p1, dtype=dtype)

    def kernel(self, x):
        logits0 = tf.tile(tf.math.log([[1 - self.p0, self.p0]]), [tf.shape(x)[0], 1])
        noise0 = tf.random.categorical(logits0, tf.shape(x)[1])
        logits1 = tf.tile(tf.math.log([[1 - self.p1, self.p1]]), [tf.shape(x)[0], 1])
        noise1 = tf.random.categorical(logits1, tf.shape(x)[1])
        erasure_symbol = tf.constant(0.5, dtype=dtype)
        y = tf.where(tf.equal(x, 0),
                     tf.where(tf.equal(noise0, 0), x, erasure_symbol),
                     tf.where(tf.equal(noise1, 0), x, erasure_symbol))
        return y

    def llr(self, y):
        erasure_symbol = tf.constant(0.5, dtype=dtype)
        inf_substitute = tf.constant(10, dtype=dtype)
        llr = tf.where(tf.equal(y, erasure_symbol),
                       tf.math.log(self.p0/self.p1),
                       tf.where(tf.equal(y, 0),  inf_substitute, -inf_substitute)
                       )
        return llr


class Ising(ChannelMemory):
    """
    """
    def __init__(self, batch_size, dim=1, p=0.1):
        super(Ising, self).__init__(batch_size, dim, name='ising')
        self.p = tf.constant(p, dtype=dtype)
        self.cardinality_s = 2
        self.state = self.add_weight(name="state", shape=(batch_size,  dim), dtype=dtype, trainable=False,
                                     initializer=initializers.constant(0.0))
        self.save_name = f"ising"
        def f_s(x, s, y, s_prime):
            if x == s_prime:
                p_s_prime = 1.0
            else:
                p_s_prime = 0.0
            return p_s_prime

        def f_y(x, s, y):
            if x == s:
                p_y = (x == y) * 1
            elif x == y:
                p_y = 0.5
            elif s == y:
                p_y = 0.5
            else:
                p_y = 0.
            return p_y

        self.cardinality = cardinality = 2

        P_out = np.array([[[f_y(x, s, y) for y in range(cardinality)]
                           for s in range(cardinality)]
                          for x in range(cardinality)])

        P_state = np.array([[[[f_s(x, s, y, s_prime) for y in range(cardinality)]
                              for s_prime in range(cardinality)]
                             for s in range(cardinality)]
                            for x in range(cardinality)])

        self.joint = (P_out[:, :,  tf.newaxis, :] * P_state)
        alphabet = tf.constant([0, 1])
        X, S, S_ = tf.meshgrid(alphabet, alphabet, alphabet, indexing='ij')
        self.combinations = tf.cast(tf.stack([X, S, S_], axis=-1), dtype=tf.int32)

    def llr(self, y):
        tmp = tf.tile(tf.expand_dims(tf.expand_dims(self.combinations, 0), 0),
                      [tf.shape(y)[0], tf.shape(y)[1], 1, 1, 1, 1])
        tmp2 = tf.tile(y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis],
                       [1, 1, self.cardinality, self.cardinality, self.cardinality, 1])
        tmp3 = tf.concat([tmp, tf.cast(tmp2, dtype=tf.int32)], axis=-1)

        prob = tf.gather_nd(self.joint, tmp3)
        return tf.cast(tf.math.log(prob+ 1e-10), dtype)

    def kernel(self, x):
        batch_size = tf.shape(x)[0]
        noise = self.sample_noise(batch_size, self.p)
        y = tf.where(tf.equal(noise, 0), x, self.state)
        self.state.assign(x)
        return y


class Trapdoor(ChannelMemory):
    """
    """
    def __init__(self, batch_size, dim=1):
        super(Trapdoor, self).__init__(batch_size, dim, name='ising')
        self.p = tf.constant(0.5, dtype=dtype)
        self.state = self.add_weight(name="state", shape=(batch_size,  dim), dtype=dtype, trainable=False,
                                     initializer=initializers.constant(0.0))
        self.cardinality_s = 2
        self.save_name = f"trapdoor"

        def f_s(x, s, y, s_prime):
            # if (x == s_prime and y == s) or (x == y and s_prime == s):
            if tf.math.floormod(x+s+y, 2) == s_prime:
                # print(x,s,y,tf.math.floormod(x + s + y, 2))
                p_s_prime = 1.0
            else:
                p_s_prime = 0.0
            return p_s_prime

        def f_y(x, s, y):
            if x == s:
                p_y = (x == y) * 1
            elif x == y:
                p_y = 0.5
            elif s == y:
                p_y = 0.5
            else:
                p_y = 0.
            return p_y

        self.cardinality = cardinality = 2

        P_out = np.array([[[f_y(x, s, y) for y in range(cardinality)]
                           for s in range(cardinality)]
                          for x in range(cardinality)])

        P_state = np.array([[[[f_s(x, s, y, s_prime) for y in range(cardinality)]
                              for s_prime in range(cardinality)]
                             for s in range(cardinality)]
                            for x in range(cardinality)])

        self.joint = (P_out[:, :,  tf.newaxis, :] * P_state) * 0.5
        alphabet = tf.constant([0, 1])
        X, S, S_ = tf.meshgrid(alphabet, alphabet, alphabet, indexing='ij')
        self.combinations = tf.cast(tf.stack([X, S, S_], axis=-1), dtype=tf.int32)

    def llr(self, y):
        tmp = tf.tile(tf.expand_dims(tf.expand_dims(self.combinations, 0), 0),
                      [tf.shape(y)[0], tf.shape(y)[1], 1, 1, 1, 1])
        tmp2 = tf.tile(y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis],
                       [1, 1, self.cardinality, self.cardinality, self.cardinality, 1])
        tmp3 = tf.concat([tmp, tf.cast(tmp2, dtype=tf.int32)], axis=-1)

        prob = tf.gather_nd(self.joint, tmp3)
        return tf.cast(tf.math.log(prob + 1e-10), dtype)

    def kernel(self, x):
        batch_size = tf.shape(x)[0]
        noise = self.sample_noise(batch_size, self.p)
        y = tf.where(tf.equal(noise, 0), x, self.state)
        s = tf.math.floormod(self.state + x + y, 2)
        self.state.assign(s)
        return y


class GE(ChannelMemory):
    """
    """
    def __init__(self, batch_size, dim=1, Pg=0.05, Pb=0.15, b=0.1, g=0.1):
        super(GE, self).__init__(batch_size, dim, name='ge')
        self.Pg = tf.constant(Pg, dtype=dtype)
        self.Pb = tf.constant(Pb, dtype=dtype)
        self.g = tf.constant(g, dtype=dtype)
        self.b = tf.constant(b, dtype=dtype)
        self.state = self.add_weight(name="state", shape=(batch_size,  dim), dtype=dtype, trainable=False,
                                     initializer=initializers.constant(0.0, dtype=dtype))
        # def f_s(x, s, y, s_prime):
        #     if (x == s_prime and y == s) or (x == y and s_prime == s):
        #         p_s_prime = 1.0
        #     else:
        #         p_s_prime = 0.0
        #     return p_s_prime
        #
        # def f_y(x, s, y):
        #     if x == s:
        #         p_y = (x == y) * 1
        #     elif x == y:
        #         p_y = 0.5
        #     elif s == y:
        #         p_y = 0.5
        #     else:
        #         p_y = 0.
        #     return p_y
        #
        # self.cardinality = cardinality = 2
        #
        # P_out = np.array([[[f_y(x, s, y) for y in range(cardinality)]
        #                    for s in range(cardinality)]
        #                   for x in range(cardinality)])
        #
        # P_state = np.array([[[[f_s(x, s, y, s_prime) for y in range(cardinality)]
        #                       for s_prime in range(cardinality)]
        #                      for s in range(cardinality)]
        #                     for x in range(cardinality)])
        #
        # self.joint = (P_out[:, :,  tf.newaxis, :] * P_state) * 0.5
        # alphabet = tf.constant([0, 1])
        # X, S, S_ = tf.meshgrid(alphabet, alphabet, alphabet, indexing='ij')
        # self.combinations = tf.cast(tf.stack([X, S, S_], axis=-1), dtype=tf.int32)

    def llr(self, y):
        tmp = tf.tile(tf.expand_dims(tf.expand_dims(self.combinations, 0), 0),
                      [tf.shape(y)[0], tf.shape(y)[1], 1, 1, 1, 1])
        tmp2 = tf.tile(y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis],
                       [1, 1, self.cardinality, self.cardinality, self.cardinality, 1])
        tmp3 = tf.concat([tmp, tf.cast(tmp2, dtype=tf.int32)], axis=-1)

        prob = tf.gather_nd(self.joint, tmp3)
        # return np.log(prob)
        return prob

    def kernel(self, x):
        batch_size = tf.shape(x)[0]
        noise_g = self.sample_noise(batch_size, self.Pg)
        noise_b = self.sample_noise(batch_size, self.Pb)
        y = tf.where(tf.equal(self.state, 0),
                     tf.math.floormod(x + noise_g, 2),
                     tf.math.floormod(x + noise_b, 2))
        noise_g = self.sample_noise(batch_size, self.g)
        noise_b = self.sample_noise(batch_size, self.b)
        s = tf.where(tf.equal(self.state, 0),
                     tf.math.floormod(self.state + noise_g, 2),
                     tf.math.floormod(self.state + noise_b, 2))
        self.state.assign(s)
        return y


class ISI(ChannelMemory):
    """
    """
    def __init__(self, batch_size,  length=1, alpha=1.0, dim=1, var=0.5):
        super(ISI, self).__init__(batch_size, dim, name='interference')
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.h = tf.math.pow(self.alpha, tf.cast(tf.range(length + 1), dtype))[:, tf.newaxis]
        # self.h = tf.math.pow(self.alpha, tf.cast(tf.range(length + 1), tf.float32))[:, tf.newaxis] / \
        #          tf.constant((1 - tf.math.pow(alpha, length + 1)) / (1 - alpha), dtype=tf.float32)

        self.var = tf.constant(var, dtype=tf.float32)

        self.state = self.add_weight(name="state", shape=(batch_size, length),
                                     dtype=tf.float32, trainable=False,
                                     initializer=initializers.constant(1.0))
        self.length = length
        self.save_name = f"memory-{length}-var-{var}-alpha-{alpha}-isi"

        def f_s(x, s, s_prime):
            format_ = "{:0" + f"{length}b" + "}"
            s_bin = format_.format(s)
            s_prime_bin = format_.format(s_prime)
            if (s_prime_bin[1:] == s_bin[:-1]) & (int(s_prime_bin[0]) == x):
                p_s_prime = 1.0
            else:
                p_s_prime = 0.0
            # print(x, s_bin, s_prime_bin, p_s_prime)

            return p_s_prime

        self.cardinality_s = cardinality_s = 2 ** self.length
        self.cardinality_x = cardinality = 2
        self.cardinality = cardinality
        P_state = np.array([[[f_s(x, s, s_prime) for s_prime in range(cardinality_s)]
                             for s in range(cardinality_s)]
                            for x in range(cardinality)])
        self.joint_no_y = tf.cast(P_state * 0.5, dtype)

        # s = self.decimal_to_binary(tf.constant(tf.range(cardinality_s)), length)
        # s = 2 * s - 1
        # xs0 = tf.concat((tf.constant(-1, shape=[cardinality_s, 1]), s), axis=1)
        # xs1 = tf.concat((tf.constant(1, shape=[cardinality_s, 1]), s), axis=1)
        # xs = tf.concat((xs0[tf.newaxis, :], xs1[tf.newaxis, :]), axis=0)
        # xs = tf.reshape(xs, [cardinality_x * cardinality_s, length + 1])
        # means = tf.matmul(tf.cast(xs, tf.float32), self.h)
        # self.means = tf.reshape(means, [cardinality_x, cardinality_s, 1])
        def f_y(x, s, s_prime):
            format_ = "{:0" + f"{length}b" + "}"
            s_bin = list(format_.format(s))
            s_bin = [int(a) for a in list(s_bin)]
            xs = tf.concat(((x,), s_bin), axis=0) * 2 - 1
            mean = tf.matmul(tf.cast(xs[tf.newaxis, :], tf.float32), self.h)
            return tf.squeeze(mean)
            # print(x, s_bin, s_prime_bin, p_s_prime)

        means = [[[f_y(x, s, s_prime) for s_prime in range(cardinality_s)]
                  for s in range(cardinality_s)]
                 for x in range(cardinality)]
        self.means = tf.convert_to_tensor(means)

    def llr(self, y):
        means = tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(self.means, 0), 0),
                                       [tf.shape(y)[0], tf.shape(y)[1], 1, 1, 1]), -1)
        y_batch = tf.tile(y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis],
                          [1, 1, self.cardinality_x, self.cardinality_s, self.cardinality_s, 1])

        prob_y = self.p_out(means, self.var, y_batch)
        joint_no_y = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.joint_no_y, 0), 0), -1),
                             [tf.shape(y)[0], tf.shape(y)[1], 1, 1, 1, 1])
        prob = joint_no_y * prob_y
        return tf.math.log(tf.squeeze(prob, axis=-1))

    def kernel(self, x):
        x = 2 * x - 1
        s = tf.identity(self.state)
        xs = tf.concat((x, s), axis=1)
        y = tf.matmul(xs, self.h) + tf.sqrt(self.var) * tf.random.normal(tf.shape(x), dtype=tf.float32)
        new_s = tf.concat((x, tf.slice(s, [0, 0], [tf.shape(s)[0], tf.shape(s)[1] - 1])), axis=1)
        self.state.assign(new_s)
        return y

    @tf.function
    def p_out(self, mu, sigma2, y):
        p_out = 1.0 / tf.math.sqrt(2.0 * tf.constant(3.1415926535, dtype=tf.float32) * sigma2) * \
                tf.math.exp(-0.5 * tf.math.square(y - mu) / sigma2)
        return p_out

    # @tf.function
    def decimal_to_binary(self, x, num_bits):
        """Convert decimal numbers to their binary representation with num_bits bits."""
        x = tf.cast(x, tf.int32)  # Cast input to integer
        bits = []
        for i in range(num_bits):
            bits.append(x % 2)
            x //= 2
        bits.reverse()  # Reverse the order of bits
        return tf.transpose(tf.convert_to_tensor(bits), [1, 0])


class MovingAverageAGN(ChannelMemory):
    """
    """
    def __init__(self, batch_size, order=5, alpha=0.8, dim=1, var=0.5):
        super(MovingAverageAGN, self).__init__(batch_size, dim, name='interference')
        self.alpha = tf.constant(alpha, dtype=dtype)
        self.h = tf.math.pow(self.alpha, tf.cast(tf.range(order+1), dtype))[:, tf.newaxis]
        self.var = tf.constant(var, dtype=dtype)
        self.state = self.add_weight(name="state", shape=(batch_size, tf.cast(order, tf.int32)),
                                     dtype=dtype, trainable=False,
                                     initializer=initializers.constant(0.0, dtype=dtype))
        # def f_s(x, s, y, s_prime):
        #     if (x == s_prime and y == s) or (x == y and s_prime == s):
        #         p_s_prime = 1.0
        #     else:
        #         p_s_prime = 0.0
        #     return p_s_prime
        #
        # def f_y(x, s, y):
        #     if x == s:
        #         p_y = (x == y) * 1
        #     elif x == y:
        #         p_y = 0.5
        #     elif s == y:
        #         p_y = 0.5
        #     else:
        #         p_y = 0.
        #     return p_y
        #
        # self.cardinality = cardinality = 2
        #
        # P_out = np.array([[[f_y(x, s, y) for y in range(cardinality)]
        #                    for s in range(cardinality)]
        #                   for x in range(cardinality)])
        #
        # P_state = np.array([[[[f_s(x, s, y, s_prime) for y in range(cardinality)]
        #                       for s_prime in range(cardinality)]
        #                      for s in range(cardinality)]
        #                     for x in range(cardinality)])
        #
        # self.joint = (P_out[:, :,  tf.newaxis, :] * P_state) * 0.5
        # alphabet = tf.constant([0, 1])
        # X, S, S_ = tf.meshgrid(alphabet, alphabet, alphabet, indexing='ij')
        # self.combinations = tf.cast(tf.stack([X, S, S_], axis=-1), dtype=tf.int32)

    def llr(self, y):
        raise NotImplementedError

    def kernel(self, x):
        n_tm1 = tf.identity(self.state)
        n_t = tf.sqrt(self.var) * tf.random.normal(tf.shape(x), dtype=dtype)
        n_cat = tf.concat((n_t, n_tm1), axis=1)
        z_t = tf.matmul(n_cat, self.h)
        y = x * 2 - 1 + z_t
        new_s = tf.concat((n_t, tf.slice(n_tm1, [0, 0], [tf.shape(n_tm1)[0], tf.shape(n_tm1)[1]-1])), axis=1)
        self.state.assign(new_s)
        return y


class InputDistribution(Layer):
    def __init__(self, logits=(0.5, 0.5), alphabet_size=2):
        super(InputDistribution, self).__init__()
        self.alphabet_size = tf.constant(alphabet_size, dtype=dtype)
        self.logits = tf.constant(logits, dtype=dtype)

    def get_config(self):
        pass

    def call(self, inputs, training=None, *args, **kwargs):
        batch = inputs
        # return tf.ones([batch, tf.cast(self.alphabet_size, tf.int32)], dtype=dtype) / self.alphabet_size
        return tf.tile(self.logits[tf.newaxis, :], [batch, 1])

