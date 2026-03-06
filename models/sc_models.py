import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import numpy as np
from sionna.phy.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder, PolarSCDecoder
from sionna.phy.fec.polar.utils import generate_5g_ranking

from tensorflow.keras.layers import Input, Layer, Dense, Concatenate, Lambda, LSTM, LSTMCell, RNN, Dropout, \
    LayerNormalization, Activation, Lambda
tf.keras.backend.set_floatx('float32')
dtype = tf.keras.backend.floatx()


def hard_dec(x):
    return tf.where(tf.greater(x, 0), 1.0, 0.0)

class CheckNodeVanilla(Model):
    def __init__(self, clip=1000.0, name='checknode'):
        super(CheckNodeVanilla, self).__init__(name=name)
        self.clip = clip

    def call(self, inputs, **kwargs):
        e1, e2 = inputs
        return tf.clip_by_value(-2 * tf.math.atanh(tf.math.tanh(e1 / 2) * tf.math.tanh(e2 / 2)), -self.clip, self.clip)


class CheckNodeMinSum(Model):
    def __init__(self, name='checknode_minsum'):
        super(CheckNodeMinSum, self).__init__(name=name)
        self._llr_max = 30.  # internal max LLR value (not very critical for SC)

    def call(self, inputs, **kwargs):
        # e1, e2 = inputs
        # return tf.math.sign(e1)*tf.math.sign(e2)*tf.minimum(tf.abs(e1), tf.abs(e2))
        x, y = inputs
        x_in = tf.clip_by_value(x,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)
        y_in = tf.clip_by_value(y,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)

        # Avoid division for numerical stability
        # Implements log(1+e^(x+y))
        llr_out = tf.math.softplus((x_in + y_in))
        # Implements log(e^x+e^y)
        llr_out -= tf.math.reduce_logsumexp(tf.stack([x_in, y_in], axis=-1),
                                            axis=-1)

        return llr_out



class BitNodeVanilla(Model):
    def __init__(self, name='bitnode'):
        super(BitNodeVanilla, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        e1, e2, uhat = inputs
        return e2 + (1. - 2. * tf.cast(uhat, dtype)) * e1


class CheckNodeTrellis(Model):
    def __init__(self, batch_dims=2, state_size=2, name='checknode'):
        super(CheckNodeTrellis, self).__init__(name=name)
        self.batch_dims = batch_dims
        self.state_size = state_size

    def call(self, inputs, **kwargs):
        e1, e2 = inputs
        s0, u1, s2 = tf.meshgrid(tf.range(tf.shape(e1)[-2]), [0, 1], tf.range(tf.shape(e1)[-1]))
        arg = tf.ones_like(u1)
        res_ = list()
        # tf.autograph.experimental.set_loop_options(
        #     shape_invariants=[(res_, tf.shape(e1))]
        # )
        repmat = tf.concat((tf.shape(e1)[:self.batch_dims], [1, 1, 1, 1]), axis=0)
        for u2 in range(2):
            for s1 in range(self.state_size):
                arg1 = tf.stack([tf.math.floormod(u1 + u2, 2), s0, s1 * arg], axis=-1)
                for _ in range(self.batch_dims):
                    arg1 = tf.expand_dims(arg1, 0)
                indices1 = tf.tile(arg1, repmat)
                arg2 = tf.stack([u2 * arg, s1 * arg, s2], axis=-1)
                for _ in range(self.batch_dims):
                    arg2 = tf.expand_dims(arg2, 0)
                indices2 = tf.tile(arg2, repmat)
                res_.append(tf.gather_nd(e1, indices1, batch_dims=self.batch_dims) +
                            tf.gather_nd(e2, indices2, batch_dims=self.batch_dims))
        res_ = tf.stack(res_, axis=-1)
        # res = tf.reduce_logsumexp(tf.math.abs(res_), axis=-1) * tf.reduce_prod(tf.math.sign(res_), axis=-1)
        res = tf.reduce_logsumexp(res_, axis=-1)

        # res = res_ / tf.reduce_mean(tf.reduce_sum(res_, axis=(2, 3, 4)))
        return res


class BitNodeTrellis(Model):
    def __init__(self, batch_dims=2, state_size=2, name='bitnode'):
        super(BitNodeTrellis, self).__init__(name=name)
        self.batch_dims = batch_dims
        self.state_size = state_size

    def call(self, inputs, **kwargs):
        e1, e2, uhat = inputs

        s0, u2, s2 = tf.meshgrid(tf.range(tf.shape(e1)[-2]), [0, 1], tf.range(tf.shape(e1)[-1]))
        uhat_ = tf.cast(uhat, tf.int32)
        arg0 = tf.expand_dims(tf.expand_dims(uhat_, -1), -1)
        repmat = tf.concat((tf.ones([self.batch_dims], tf.int32), tf.shape(e1)[self.batch_dims:]), axis=0)
        uhat_t = tf.tile(arg0, repmat)
        u2_tiled = u2
        for i in range(self.batch_dims):
            u2_tiled = tf.expand_dims(u2_tiled, 0)
        repmat = tf.concat((tf.shape(e1)[:self.batch_dims], [1, 1, 1]), axis=0)
        u2_t = tf.tile(u2_tiled, repmat)
        u_xor = tf.math.floormod(u2_t + uhat_t, 2)

        arg = tf.ones_like(u2)
        res_ = list()
        repmat = tf.concat((tf.shape(e1)[:self.batch_dims], [1, 1, 1, 1]), axis=0)

        for s1 in range(self.state_size):
            arg1 = tf.stack([s0, s1 * arg], axis=-1)
            for i in range(self.batch_dims):
                arg1 = tf.expand_dims(arg1, 0)
            indices1 = tf.tile(arg1, repmat)
            indices1 = tf.concat([tf.expand_dims(u_xor, -1), indices1], axis=-1)

            arg2 = tf.stack([u2 * arg, s1 * arg, s2], axis=-1)
            for i in range(self.batch_dims):
                arg2 = tf.expand_dims(arg2, 0)
            indices2 = tf.tile(arg2, repmat)

            res_.append(tf.gather_nd(e1, indices1, batch_dims=self.batch_dims) +
                        tf.gather_nd(e2, indices2, batch_dims=self.batch_dims))

        res_ = tf.stack(res_, axis=-1)
        res = tf.reduce_logsumexp(res_, axis=-1)
        return res


class Embedding2LLRTrellis(Model):
    def __init__(self, batch_dims=2, name='emb2prob_trellis'):
        super(Embedding2LLRTrellis, self).__init__(name=name)
        self.batch_dims = batch_dims

    def call(self, inputs, training=None, **kwargs):
        e = inputs

        e1 = tf.squeeze(tf.gather(e, indices=[1], axis=self.batch_dims), axis=self.batch_dims)
        e0 = tf.squeeze(tf.gather(e, indices=[0], axis=self.batch_dims), axis=self.batch_dims)
        p1 = tf.reduce_logsumexp(e1, axis=(-1, -2))
        p0 = tf.reduce_logsumexp(e0, axis=(-1, -2))
        # tf.einsum(f'{self.ein_str}lm->{self.ein_str}', e1)
        # p0 = tf.einsum(f'{self.ein_str}lm->{self.ein_str}', e0)
        # e = tf.sigmoid(tf.math.log(tf.where(p1 > p0, 2., 0.5)))
        e = tf.cast(p1-p0, tf.float32)
        return tf.expand_dims(e, -1)


class EmbeddingX(Model):
    def __init__(self, logits, name='embedding_x'):
        """

        Returns:
            object:
        """
        super(EmbeddingX, self).__init__(name=name)
        self.input_logits = self.add_weight(name="logits", shape=logits.shape.as_list(), dtype=dtype, trainable=True,
                                            initializer=tf.keras.initializers.constant(logits))
        self.logits_shape = (1,) if len(tf.shape(self.input_logits)) == 0 else tf.shape(self.input_logits)

    def call(self, inputs, training=None, **kwargs):
        e = tf.broadcast_to(self.input_logits,
                            shape=tf.concat([inputs, self.logits_shape], axis=0))
        return e


class EmbeddingY(Model):
    def __init__(self, hidden_size, embedding_size, activation='elu', use_bias=True,
                 layer_normalization=False, name='emb_y'):
        super(EmbeddingY, self).__init__(name=name)

        self._layers = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer1"),
                        Dense(embedding_size, activation=None, use_bias=use_bias, name=f"{name}-layer2")]

        if layer_normalization:
            self._layers.insert(0, tf.keras.layers.LayerNormalization())

    def call(self, inputs, training=None, **kwargs):
        e = inputs
        for layer in self._layers:
            e = layer.__call__(e, training=training)
        return e


class CheckNodeNNEmb(Model):
    def __init__(self, hidden_size, embedding_size, layers_per_op, activation='elu',
                 use_bias=True, name='checknode_nnops'):
        super(CheckNodeNNEmb, self).__init__(name=name)

        self._layers = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer{i}")
                        for i in range(layers_per_op)] + \
                       [Dense(embedding_size, activation=None, use_bias=use_bias, name=f"{name}-layer{layers_per_op}")]
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None, *args):
        e1, e2 = inputs
        e = tf.concat([e1, e2], axis=-1)
        for layer in self._layers:
            e = layer.__call__(e, training=training)
        ### norm ###
        # if len(e.shape) == 4:
        #     origshape = e.shape
        #     ex1est = tf.reshape(
        #         self.layer_norm1(tf.reshape(e, (origshape[0] * origshape[1], origshape[2], origshape[3]))), origshape)
        #     return ex1est
        #
        # return self.layer_norm1(e)
        ### norm ###
        return e + e1 + e2


class BitNodeNNEmb(Model):
    def __init__(self, hidden_size, embedding_size, layers_per_op, activation='elu',
                 use_bias=True, name='bitnode_nnops'):
        super(BitNodeNNEmb, self).__init__(name=name)
        self._layers = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer{i}")
                        for i in range(layers_per_op)] + \
                       [Dense(embedding_size, activation=None, use_bias=use_bias, name=f"{name}-layer{layers_per_op}")]

        # self._layers2 = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer{i}2")
        #                 for i in range(layers_per_op)] + \
        #                [Dense(embedding_size, activation=None, use_bias=use_bias, name=f"{name}-layer{layers_per_op}2")]
        self.layer_norm1 = tf.keras.layers.LayerNormalization()


    def call(self, inputs, training=None, *args):
        e1, e2, u = inputs
        u_sign = 2. * tf.cast(u, tf.float32) - 1.

        e = tf.concat([e1 * u_sign, e2], axis=-1)
        #e = tf.concat([e1, e2, u_sign], axis=-1)
        for layer in self._layers:
            e = layer.__call__(e, training=training)
        return e + e1 * u_sign + e2
        #return self.layer_norm1(e + e1 * u_sign + e2)



class BatchNormModel(Model):
    def __init__(self, num_layers=20):
        super(BatchNormModel, self).__init__()
        # Create a list of BatchNormalization layers
        self.layer_norms = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]


    def call(self, inputs, ind=0):
        # Apply the batch normalization for each layer
        x = self.layer_norms[int(ind)](inputs)
        return x


class Embedding2LLR(Model):
    def __init__(self, hidden_size, layers_per_op, activation='elu', use_bias=True, name='emb2llr_nnops', llr_max=None):
        super(Embedding2LLR, self).__init__(name=name)

        self._layers = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer{i}")
                        for i in range(layers_per_op)] + \
                       [Dense(1, activation=None, use_bias=use_bias, name=f"{name}-layer{layers_per_op}")]
        self._llr_max = llr_max

    def call(self, inputs, training=None, **kwargs):
        e = inputs
        for layer in self._layers:
            e = layer.__call__(e, training=training)
        if self._llr_max is not None:
            e = tf.clip_by_value(e, clip_value_min=-self._llr_max, clip_value_max=self._llr_max)
        return e


class Embedding2LLRwithSNR(Model):
    """Like Embedding2LLR but also receives log(no) as an extra input feature.

    Internally appends log(channel.no) to the embedding before the Dense layers,
    so decode_list call signature stays unchanged.  At init the SNR column is
    zero-padded from the pre-trained Embedding2LLR weights, so the model starts
    identical to the pre-trained state and learns SNR-adaptive magnitude calibration.

    Args:
        no_source: tf.Variable (e.g. polar.channel.no) — set by sample_channel_outputs
                   before decode_list is called.  Works for scalar or (batch,1) shape.
        hidden_size, layers_per_op, activation: same as Embedding2LLR.
    """
    def __init__(self, no_source, hidden_size, layers_per_op,
                 activation='elu', use_bias=True, name='emb2llr_nnops_snr', llr_max=30.):
        super(Embedding2LLRwithSNR, self).__init__(name=name)
        object.__setattr__(self, 'no_source', no_source)  # bypass Keras tracking — not an owned weight
        # input dim = emb_dim + 1  (embedding + log_no feature)
        # NOTE: must NOT use self._layers — that name is reserved by Keras Model internals
        self._nnops = [Dense(hidden_size, activation=activation, use_bias=use_bias,
                             name=f"{name}-layer{i}")
                       for i in range(layers_per_op)] + \
                      [Dense(1, activation=None, use_bias=use_bias,
                             name=f"{name}-layer{layers_per_op}")]
        self._llr_max = llr_max

    def call(self, inputs, training=None, **kwargs):
        no_val = tf.abs(tf.cast(self.no_source, tf.float32))
        batch_size = tf.shape(inputs)[0]
        # scalar no (e.g. AWGN fixed SNR) → tile to (batch,)
        # batched no (e.g. 5G per-sample, shape (batch,1)) → per-sample mean → (batch,)
        log_no = tf.cond(
            tf.equal(tf.size(no_val), 1),
            true_fn=lambda: tf.repeat(
                tf.math.log(tf.reshape(no_val, [1]) + 1e-10), batch_size),
            false_fn=lambda: tf.math.log(
                tf.reduce_mean(tf.reshape(no_val, [batch_size, -1]), axis=1) + 1e-10)
        )
        # Reshape log_no to [batch, 1, ..., 1] to broadcast over all dims after batch.
        # Works for both (batch, N, emb) and list mode (batch, list, N, emb).
        log_no_bc = tf.reshape(log_no, [batch_size] + [1] * (inputs.shape.rank - 1))
        log_no_feat = log_no_bc * tf.ones_like(inputs[..., :1])  # (..., N, 1)
        e = tf.concat([inputs, log_no_feat], axis=-1)  # (..., N, emb_dim+1)
        for layer in self._nnops:
            e = layer.__call__(e, training=training)
        if self._llr_max is not None:
            e = tf.clip_by_value(e, clip_value_min=-self._llr_max, clip_value_max=self._llr_max)
        return e


class NeuralReranker(Model):
    """Small MLP that scores each SCL candidate path for oracle-supervised reranking (Option 4).

    For each candidate, computes a matched embedding:
        matched = eyx * (2 * uhat - 1)   — positive where channel agrees with candidate bit
    then pools over bit positions and passes through a 2-layer MLP to produce a scalar score.
    The candidate with the highest score is selected as the decoded codeword.

    Args:
        emb_dim: embedding dimension (must match embedding_size_polar, e.g. 128)
        hidden:  hidden layer width (default 128)
        activation: hidden layer activation (default 'elu')
    """
    def __init__(self, emb_dim=128, hidden=128, activation='elu', name='neural_reranker'):
        super(NeuralReranker, self).__init__(name=name)
        self.dense1 = Dense(hidden, activation=activation, use_bias=True)
        self.dense2 = Dense(1, activation=None, use_bias=True)

    def call(self, eyx, uhat_l):
        """Score one candidate path.

        Args:
            eyx:    (batch, N, emb_dim)  channel embeddings
            uhat_l: (batch, N, 1)        candidate bits in {0.0, 1.0}
        Returns:
            score:  (batch, 1)           unnormalized log-score (higher = better)
        """
        # matched[b, n, d] > 0 where channel embedding agrees with candidate bit
        matched = eyx * (2.0 * uhat_l - 1.0)          # (batch, N, emb_dim)
        pooled  = tf.reduce_mean(matched, axis=1)      # (batch, emb_dim)
        h       = self.dense1(pooled)                  # (batch, hidden)
        score   = self.dense2(h)                       # (batch, 1)
        return score


class EyModel(Model):
    def __init__(self, embedding_size, BPS, activation):
        super(EyModel, self).__init__()
        # Create a list of BatchNormalization layers
        self._layers = []
        for i in range(BPS):
            self._layers.append(Sequential([Dense(50, activation=activation, use_bias=True) for i in range(2)] + \
                              [Dense(embedding_size, use_bias=True, activation=None)]))


    def call(self, y):
        # Apply the batch normalization for each layer
        ey1 = self._layers[0](y)
        ey2 = self._layers[1](y)
        array1 = np.arange(ey1.shape[1])
        array2 = np.arange(ey1.shape[1],ey1.shape[1]*2)
        combined_array = np.empty((array1.size + array2.size,), dtype=array1.dtype)
        combined_array[0::2] = array1
        combined_array[1::2] = array2
        ey = tf.gather(tf.concat([ey1, ey2], axis=1), combined_array, axis=1)
        return ey

class InterleavingMethod(Layer):
        def __init__(self, n, channel_interleaver_flag=True):
            super().__init__()
            self.channel_interleaver_flag = channel_interleaver_flag
            self._n_target = n
            self._n_polar = n
            # encode part
            self._ind_rate_matching = self._init_rate_matching(self._n_polar)
            # decode part
            self._init_interleavers()

        def _init_rate_matching(self, n_polar):

            #ind_input_int = None

            # Generate tf.gather indices for sub-block interleaver
            ind_sub_int = self.subblock_interleaving(np.arange(n_polar))

            # Rate matching via circular buffer as defined in Sec. 5.4.1.2
            c_int = np.arange(n_polar)
            idx_c_matched = np.zeros([n_polar])
            for ind in range(n_polar):
                idx_c_matched[ind] = c_int[np.mod(ind, n_polar)]

            # For uplink only: generate input bit interleaver
            if self.channel_interleaver:
                ind_channel_int = self.channel_interleaver(np.arange(n_polar))
                # Combine indices for single tf.gather operation
                ind_t = idx_c_matched[ind_channel_int].astype(int)
                idx_rate_matched = ind_sub_int[ind_t]
            else:  # no channel interleaver for downlink
                idx_rate_matched = ind_sub_int[idx_c_matched.astype(int)]
            return idx_rate_matched

        def subblock_interleaving(self, u):
            """Input bit interleaving as defined in Sec 5.4.1.1 [3GPPTS38212]_.

            Input
            -----
                u: ndarray
                    1D array to be interleaved. Length of ``u`` must be a multiple
                    of 32.

            Output
            ------
                : ndarray
                    Interleaved version of ``u`` with same shape and dtype as ``u``.

            Raises
            ------
                AssertionError
                    If length of ``u`` is not a multiple of 32.

            """

            k = u.shape[-1]
            assert np.mod(k, 32) == 0, \
                "length for sub-block interleaving must be a multiple of 32."
            y = np.zeros_like(u)

            # Permutation according to Tab 5.4.1.1.1-1 in 38.212
            perm = np.array([0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
                             12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27,
                             29, 30, 31])

            for n in range(k):
                i = int(np.floor(32 * n / k))
                j = perm[i] * k / 32 + np.mod(n, k / 32)
                j = int(j)
                y[n] = u[j]

            return y

        def channel_interleaver(self, c):
            """Triangular interleaver following Sec. 5.4.1.3 in [3GPPTS38212]_.

            Input
            -----
                c: ndarray
                    1D array to be interleaved.

            Output
            ------
                : ndarray
                    Interleaved version of ``c`` with same shape and dtype as ``c``.

            """

            n = c.shape[-1]  # Denoted as E in 38.212
            c_int = np.zeros_like(c)

            # Find smallest T s.t. T*(T+1)/2 >= n
            t = 0
            while t * (t + 1) / 2 < n:
                t += 1

            v = np.zeros([t, t])
            ind_k = 0
            for ind_i in range(t):
                for ind_j in range(t - ind_i):
                    if ind_k < n:
                        v[ind_i, ind_j] = c[ind_k]
                    else:
                        v[ind_i, ind_j] = np.nan  # NULL
                    # Store nothing otherwise
                    ind_k += 1
            ind_k = 0
            for ind_j in range(t):
                for ind_i in range(t - ind_j):
                    if not np.isnan(v[ind_i, ind_j]):
                        c_int[ind_k] = v[ind_i, ind_j]
                        ind_k += 1
            return c_int

        def _init_interleavers(self):
            """Initialize inverse interleaver patterns for rate-recovery."""

            # Channel interleaver
            ind_ch_int = self.channel_interleaver(
                np.arange(self._n_target))
            self.ind_ch_int_inv = np.argsort(ind_ch_int)  # Find inverse perm

            # Sub-block interleaver
            ind_sub_int = self.subblock_interleaving(
                np.arange(self._n_polar))
            self.ind_sub_int_inv = np.argsort(ind_sub_int)  # Find inverse perm

            # input bit interleaver
            self.ind_iil_inv = None


        def apply_interleaving(self, x):
            # Sub-block interleaving with 32 sub-blocks as in Sec. 5.4.1.1
            # Rate matching via circular buffer as defined in Sec. 5.4.1.2
            # For uplink only: channel interleaving (i_bil=True)
            x_matched = tf.gather(x, self._ind_rate_matching, axis=1)
            return x_matched
        def remove_interleaving(self, llr_ch):
            if self.channel_interleaver:
                # Undo channel interleaving
                llr_deint = tf.gather(llr_ch, self.ind_ch_int_inv, axis=1)
            else:
                llr_deint = llr_ch
            # Remove sub-block interleaving
            llr_matched = tf.gather(llr_deint, self.ind_sub_int_inv, axis=1)
            return llr_matched

class MyPolar5GEncoder(Polar5GEncoder):
    def __init__(self,
                 k,
                 n,
                 channel_type="uplink",
                 verbose=False,
                 dtype=tf.float32,
                 list_size=1,
                 return_u_and_llrs=False,
                 return_u=False,
                 ch_ranking=None):
        self.return_u = return_u
        self.list_size = list_size
        self.return_u_and_llrs = return_u_and_llrs
        self.ch_ranking_my = ch_ranking

        super(MyPolar5GEncoder, self).__init__(k,
                 n,
                 channel_type=channel_type,
                 verbose=True,
                 dtype=tf.float32)


    def _init_rate_match(self, k_target, n_target):
        """Implementing polar rate matching according to [3GPPTS38212]_.

        Please note that this part of the code only runs during the
        initialization and, thus, is not performance critical. For easier
        alignment and traceability with the standard document [3GPPTS38212]_
        the implementation prefers `for loop`-based indexing.

        The relation of terminology between [3GPPTS38212]_ and this code is
        given as:
        `A`...`k_target`
        `E`...`n_target`
        `K`...`k_polar`
        `N`...`n_polar`
        `L`...`k_crc`.
        """

        # Check input for consistency (see Sec. 6.3.1.2.1 for UL)

        # currently not relevant (segmentation not supported)
        # assert k_target<=1706, "Maximum supported codeword length for" \
        # "Polar  coding is 1706."

        assert n_target >= k_target, "n must be larger or equal k."
        assert n_target >= 18, \
            "n<18 is not supported by the 5G Polar coding scheme."
        assert k_target <= 1013, \
            "k too large - no codeword segmentation supported at the moment."
        assert n_target <= 1088, \
            "n too large - no codeword segmentation supported at the moment."

        # Select CRC polynomials (see Sec. 6.3.1.2.1 for UL)

        if self._channel_type == "uplink":
            if 12 <= k_target <= 19:
                crc_pol = "CRC6"
                k_crc = 6
            elif k_target >= 20:
                crc_pol = "CRC11"
                k_crc = 11
            else:
                raise ValueError("k_target<12 is not supported in 5G NR for " \
                                 "the uplink; please use 'channel coding of small block  " \
                                 "lengths' scheme from Sec. 5.3.3 in 3GPP 38.212 instead.")

            # PC bit for k_target = 12-19 bits (see Sec. 6.3.1.3.1 for UL)
            n_pc = 0
            # n_pc_wm = 0
            if k_target <= 19:
                # n_pc = 3
                n_pc = 0  # Currently deactivated
                print("Warning: For 12<=k<=19 additional 3 parity-check bits " \
                      "are defined in 38.212. They are currently not " \
                      "implemented by this encoder and, thus, ignored.")
                if n_target - k_target > 175:
                    # n_pc_wm = 1 # not implemented
                    pass

        else:  # downlink channel
            # for downlink CRC24 is used
            # remark: in PDCCH messages are limited to k=140
            # as the input interleaver does not support longer sequences
            assert k_target <= 140, \
                "k too large for downlink channel configuration."
            assert n_target >= 25, \
                "n too small for downlink channel configuration with 24 bit " \
                "CRC."
            # assert n_target <= 576, \
            #     "n too large for downlink channel configuration."
            crc_pol = "CRC24C"  # following 7.3.2
            k_crc = 24
            n_pc = 0


        # No input interleaving for uplink needed

        # Calculate Polar payload length (CRC bits are treated as info bits)
        if self.list_size==1:
            k_crc = 0
        k_polar = k_target + k_crc + n_pc

        assert k_polar <= n_target, "Device is not expected to be configured " \
                                    "with k_polar + k_crc + n_pc > n_target."

        # Select polar mother code length n_polar
        n_min = 5
        if self._channel_type == "downlink":
            n_max = 9
        else:
            n_max = 10  # For uplink; otherwise 9

        # Select rate-matching scheme following Sec. 5.3.1
        if (n_target <= ((9 / 8) * 2 ** (np.ceil(np.log2(n_target)) - 1)) and
                k_polar / n_target < 9 / 16):
            n1 = np.ceil(np.log2(n_target)) - 1
        else:
            n1 = np.ceil(np.log2(n_target))
        n2 = np.ceil(np.log2(8 * k_polar))  # Lower bound such that rate > 1/8
        n_polar = int(2 ** np.max((np.min([n1, n2, n_max]), n_min)))

        # Puncturing and shortening as defined in Sec. 5.4.1.1
        prefrozen_pos = []  # List containing the pre-frozen indices
        if n_target < n_polar:
            if k_polar / n_target <= 7 / 16:
                # Puncturing
                if self._verbose:
                    print("Using puncturing for rate-matching.")
                n_int = 32 * np.ceil((n_polar - n_target) / 32)
                int_pattern = self.subblock_interleaving(np.arange(n_int))
                for i in range(n_polar - n_target):
                    # Freeze additional bits
                    prefrozen_pos.append(int(int_pattern[i]))
                if n_target >= 3 * n_polar / 4:
                    t = int(np.ceil(3 / 4 * n_polar - n_target / 2) - 1)
                else:
                    t = int(np.ceil(9 / 16 * n_polar - n_target / 4) - 1)
                # Extra freezing
                for i in range(t):
                    prefrozen_pos.append(i)
            else:
                # Shortening ("through" sub-block interleaver)
                if self._verbose:
                    print("Using shortening for rate-matching.")
                n_int = 32 * np.ceil((n_polar) / 32)
                int_pattern = self.subblock_interleaving(np.arange(n_int))
                for i in range(n_target, n_polar):
                    prefrozen_pos.append(int_pattern[i])

        # Remove duplicates
        prefrozen_pos = np.unique(prefrozen_pos)

        # Find the remaining n_polar - k_polar - |frozen_set|

        # Load full channel ranking
        if  self.ch_ranking_my is not None:
            ch_ranking = self.ch_ranking_my
        else:
            ch_ranking, _ = generate_5g_ranking(0, n_polar, sort=False)

        # Remove positions that are already frozen by `pre-freezing` stage
        info_cand = np.setdiff1d(ch_ranking, prefrozen_pos, assume_unique=True)

        # Identify k_polar most reliable positions from candidate positions
        info_pos = []
        for i in range(k_polar):
            info_pos.append(info_cand[-i - 1])

        # Sort and create frozen positions for n_polar indices (no shortening)
        info_pos = np.sort(info_pos).astype(int)
        frozen_pos = np.setdiff1d(np.arange(n_polar),
                                  info_pos,
                                  assume_unique=True)

        # For downlink only: generate input bit interleaver
        if self._channel_type == "downlink":
            if self._verbose:
                print("Using input bit interleaver for downlink.")
            ind_input_int = self.input_interleaver(np.arange(k_polar))
        else:
            ind_input_int = None

        # Generate tf.gather indices for sub-block interleaver
        ind_sub_int = self.subblock_interleaving(np.arange(n_polar))

        # Rate matching via circular buffer as defined in Sec. 5.4.1.2
        c_int = np.arange(n_polar)
        idx_c_matched = np.zeros([n_target])
        if n_target >= n_polar:
            # Repetition coding
            if self._verbose:
                print("Using repetition coding for rate-matching")
            for ind in range(n_target):
                idx_c_matched[ind] = c_int[np.mod(ind, n_polar)]
        else:
            if k_polar / n_target <= 7 / 16:
                # Puncturing
                for ind in range(n_target):
                    idx_c_matched[ind] = c_int[ind + n_polar - n_target]
            else:
                # Shortening
                for ind in range(n_target):
                    idx_c_matched[ind] = c_int[ind]

        # For uplink only: generate input bit interleaver
        if self._channel_type == "uplink":
            if self._verbose:
                print("Using channel interleaver for uplink.")
            ind_channel_int = self.channel_interleaver(np.arange(n_target))

            # Combine indices for single tf.gather operation
            ind_t = idx_c_matched[ind_channel_int].astype(int)
            idx_rate_matched = ind_sub_int[ind_t]
        else:  # no channel interleaver for downlink
            idx_rate_matched = ind_sub_int[idx_c_matched.astype(int)]

        if self._verbose:
            print("Code parameters after rate-matching: " \
                  f"k = {k_target}, n = {n_target}")
            print(f"Polar mother code: k_polar = {k_polar}, " \
                  f"n_polar = {n_polar}")
            print("Using", crc_pol)
            print("Frozen positions: ", frozen_pos)
            print("Channel type: " + self._channel_type)

        return crc_pol, n_polar, frozen_pos, idx_rate_matched, ind_input_int

    def call(self, inputs):
        """Polar encoding function including rate-matching and CRC encoding.

        This function returns the polar encoded codewords for the given
        information bits ``inputs`` following [3GPPTS38212]_ including
        rate-matching.

        Args:
            inputs (tf.float32): Tensor of shape `[...,k]` containing the
            information bits to be encoded.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]`.

        Raises:
            TypeError: If ``inputs`` is not `tf.float32`.

            InvalidArgumentError: When rank(``inputs``)<2.

            InvalidArgumentError: When shape of last dim is not ``k``.
        """

        # Reshape inputs to [...,k]
        tf.debugging.assert_greater(tf.rank(inputs), 1)
        input_shape = inputs.shape
        new_shape = [-1, input_shape[-1]]
        u = tf.reshape(inputs, new_shape)

        # Consistency check (i.e., binary) of inputs will be done in super_class

        # CRC encode
        if self.list_size!=1 and not self.return_u_and_llrs:
            u_crc = self._enc_crc(u)
        else:
            u_crc = u

        # For downlink only: apply input bit interleaver
        if self._channel_type=="downlink":
            u_crc = tf.gather(u_crc, self._ind_input_int, axis=-1)

        # Encode bits (= channel allocation + Polar transform)
        c = super(Polar5GEncoder, self).call(u_crc)

        # Sub-block interleaving with 32 sub-blocks as in Sec. 5.4.1.1
        # Rate matching via circular buffer as defined in Sec. 5.4.1.2
        # For uplink only: channel interleaving (i_bil=True)
        c_matched = tf.gather(c, self._ind_rate_matching, axis=1)

        # Restore original shape
        input_shape_list = input_shape.as_list()
        output_shape = input_shape_list[0:-1] + [self._n_target]
        output_shape[0] = -1 # To support dynamic shapes
        c_reshaped = tf.reshape(c_matched, output_shape)
        if self.return_u:
            return c_reshaped, u_crc
        return c_reshaped

class MyPreDecoder(Polar5GDecoder):
    def __init__(self,
                 enc_polar,
                 dec_type="SC",
                 list_size=8,
                 num_iter=20,
                 return_crc_status=False,
                 output_dtype=tf.float32,
                 **kwargs):
        self.list_size = list_size
        super(MyPreDecoder, self).__init__(
                 enc_polar,
                 dec_type=dec_type,
                 list_size=list_size,
                 num_iter=num_iter,
                 return_crc_status=return_crc_status,
                 output_dtype=tf.float32,)
    def call(self, inputs):
        """Polar decoding and rate-recovery for uplink 5G Polar codes.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,k]` containing
                hard-decided estimates of all ``k`` information bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[..., n]`
                or `dtype` is not `output_dtype`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """

        # tf.debugging.assert_type(inputs, self._output_dtype,
        #                          "Invalid input dtype.")
        # internal calculations still in tf.float32
        inputs = tf.cast(inputs, tf.float32)

        # Reshape inputs to [-1, n]
        tf.debugging.assert_greater(tf.rank(inputs), 1)
        input_shape = inputs.shape
        new_shape = [-1, self._n_target]
        llr_ch = tf.reshape(inputs, new_shape)

        # Note: logits are not inverted here; this is done in the decoder itself

        # 1.) Undo channel interleaving
        if self._bil:
            llr_deint = tf.gather(llr_ch, self.ind_ch_int_inv, axis=1)
        else:
            llr_deint = llr_ch

        # 2.) Remove puncturing, shortening, repetition (see Sec. 5.4.1.2)
        # a) Puncturing: set LLRs to 0
        # b) Shortening: set LLRs to infinity
        # c) Repetition: combine LLRs
        if self._n_target >= self._n_polar:
            # Repetition coding
            # Add the last n_rep positions to the first llr positions
            n_rep = self._n_target - self._n_polar
            llr_1 = llr_deint[:,:n_rep]
            llr_2 = llr_deint[:,n_rep:self._n_polar]
            llr_3 = llr_deint[:,self._n_polar:]
            llr_dematched = tf.concat([llr_1+llr_3, llr_2], 1)
        else:
            if self._k_polar/self._n_target <= 7/16:
                # Puncturing
                # Append n_polar - n_target "zero" llrs to first positions
                llr_zero = tf.zeros([tf.shape(llr_deint)[0],
                                     self._n_polar-self._n_target])
                llr_dematched = tf.concat([llr_zero, llr_deint], 1)
            else:
                # Shortening
                # Append n_polar - n_target "-infinity" llrs to last positions
                # Remark: we still operate with logits here, thus the neg. sign
                llr_infty = -self._llr_max * tf.ones([tf.shape(llr_deint)[0],
                                                self._n_polar-self._n_target])
                llr_dematched = tf.concat([llr_deint, llr_infty], 1)

        # 3.) Remove subblock interleaving
        llr_dec = tf.gather(llr_dematched, self.ind_sub_int_inv, axis=1)

        return llr_dec
