import tensorflow as tf
import numpy as np
import warnings
from sionna.phy import Block
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.polar.encoding import Polar5GEncoder
import numbers

#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for Polar decoding such as successive cancellation (SC), successive
cancellation list (SCL) and iterative belief propagation (BP) decoding."""

import tensorflow as tf
import numpy as np
import warnings
from sionna.phy import Block
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.polar.encoding import Polar5GEncoder
import numbers

class PolarSCDecoder_prefectfrozen(Block):
    """Successive cancellation (SC) decoder [Arikan_Polar]_ for Polar codes and
    Polar-like codes.

    Parameters
    ----------
    frozen_pos: ndarray
        Array of `int` defining the ``n-k`` indices of the frozen positions.

    n: int
        Defining the codeword length.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    llr_ch: [...,n], tf.float
        Tensor containing the channel LLR values (as logits).

    Output
    ------
    : [...,k], tf.float
        Tensor  containing hard-decided estimations of all ``k``
        information bits.

    Note
    ----
    This block implements the SC decoder as described in
    [Arikan_Polar]_. However, the implementation follows the `recursive
    tree` [Gross_Fast_SCL]_ terminology and combines nodes for increased
    throughputs without changing the outcome of the algorithm.

    As commonly done, we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting frozen
    bits to `1` may result in `affine` codes instead of linear code as the
    `all-zero` codeword is not necessarily part of the code any more.
    """

    def __init__(self, frozen_pos, n, precision=None, **kwargs):

        super().__init__(precision=precision, **kwargs)

        # assert error if r>1 or k, n are negative
        if not isinstance(n, numbers.Number):
            raise TypeError( "n must be a number.")
        n = int(n) # n can be float (e.g. as result of n=k*r)

        if not np.issubdtype(frozen_pos.dtype, int):
            raise TypeError("frozen_pos contains non int.")
        if len(frozen_pos)>n:
            msg = "Num. of elements in frozen_pos cannot be greater than n."
            raise ValueError(msg)
        if np.log2(n)!=int(np.log2(n)):
            raise ValueError("n must be a power of 2.")

        # store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        if self._k!=len(self._info_pos):
            msg = "Internal error: invalid info_pos generated."
            raise ArithmeticError(msg)

        self._llr_max = 30. # internal max LLR value (uncritical for SC dec)
        # and create a frozen bit vector for simpler encoding
        self._frozen_ind = np.zeros(self._n)
        self._frozen_ind[self._frozen_pos] = 1

        # enable graph pruning
        self._use_fast_sc = False

    ###############################
    # Public methods and properties
    ###############################

    @property
    def n(self):
        """Codeword length"""
        return self._n

    @property
    def k(self):
        """Number of information bits"""
        return self._k

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding"""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding"""
        return self._info_pos

    @property
    def llr_max(self):
        """Maximum LLR value for internal calculations"""
        return self._llr_max

    #################
    # Utility methods
    #################

    def _cn_op_tf(self, x, y):
        """Check-node update (boxplus) for LLR inputs.

        Operations are performed element-wise.

        See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations.
        """
        x_in = tf.clip_by_value(x,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)
        y_in = tf.clip_by_value(y,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)

        # avoid division for numerical stability
        llr_out = tf.math.log(1 + tf.math.exp(x_in + y_in))
        llr_out -= tf.math.log(tf.math.exp(x_in) + tf.math.exp(y_in))

        return llr_out

    def _vn_op_tf(self, x, y, u_hat):
        """VN update for LLR inputs."""
        return tf.multiply((1-2*u_hat), x) + y

    def _polar_decode_sc_tf(self, llr_ch, frozen_ind):
        """Recursive SC decoding function.

        Recursively branch decoding tree and split into decoding of `upper`
        and `lower` path until reaching a leaf node.

        The function returns the u_hat decisions at stage `0` and the bit
        decisions of the intermediate stage `s` (i.e., the re-encoded version of
        `u_hat` until the current stage `s`).

        Note:
            This decoder parallelizes over the batch-dimension, i.e., the tree
            is processed for all samples in the batch in parallel. This yields a
            higher throughput, but does not improve the latency.
        """

        # calculate current codeword length
        n = frozen_ind.shape[1]

        # branch if leaf is not reached yet
        if n>1:
            if self._use_fast_sc:
                if np.sum(frozen_ind)==n:
                    #print("rate-0 detected! Length: ", n)
                    u_hat = tf.zeros_like(llr_ch)
                    return u_hat, u_hat

            llr_ch1 = llr_ch[...,0:int(n/2)]
            llr_ch2 = llr_ch[...,int(n/2):]
            frozen_ind1 = frozen_ind[:,0:int(n/2)]
            frozen_ind2 = frozen_ind[:, int(n/2):]

            # upper path
            x_llr1_in = self._cn_op_tf(llr_ch1, llr_ch2)

            # and call the decoding function (with upper half)
            u_hat1, u_hat1_up, llr_hat1= self._polar_decode_sc_tf(x_llr1_in, frozen_ind1)

            # lower path
            x_llr2_in = self._vn_op_tf(llr_ch1, llr_ch2, u_hat1_up)
            # and call the decoding function again (with lower half)
            u_hat2, u_hat2_up, llr_hat2 = self._polar_decode_sc_tf(x_llr2_in, frozen_ind2)
            llr_hat = tf.concat([llr_hat1, llr_hat2], -1)
            # combine u_hat from both branches
            u_hat = tf.concat([u_hat1, u_hat2], -1)

            # calculate re-encoded version of u_hat at current stage
            # u_hat1_up = tf.math.mod(u_hat1_up + u_hat2_up, 2)
            # combine u_hat via bitwise_xor (more efficient than mod2)
            u_hat1_up_int = tf.cast(u_hat1_up, tf.int8)
            u_hat2_up_int = tf.cast(u_hat2_up, tf.int8)
            u_hat1_up_int = tf.bitwise.bitwise_xor(u_hat1_up_int,
                                                   u_hat2_up_int)
            u_hat1_up = tf.cast(u_hat1_up_int , self.rdtype)
            u_hat_up = tf.concat([u_hat1_up, u_hat2_up], -1)

        else: # if leaf is reached perform basic decoding op (=decision)
            u_hat = frozen_ind
            u_hat_up = u_hat
            llr_hat = llr_ch
            # if frozen_ind==1: # position is frozen
            #     u_hat = tf.expand_dims(tf.zeros_like(llr_ch[:,0]), axis=-1)
            #     u_hat_up = u_hat
            # else: # otherwise hard decide
            #     u_hat = 0.5 * (1. - tf.sign(llr_ch))
            #     #remove "exact 0 llrs" leading to u_hat=0.5
            #     u_hat = tf.where(tf.equal(u_hat, 0.5),
            #                      tf.ones_like(u_hat),
            #                      u_hat)
            #     u_hat_up = u_hat
        return u_hat, u_hat_up, llr_hat

    ########################
    # Sionna Block functions
    ########################

    # def build(self, input_shape):
    #     """Check if shape of input is invalid."""
    #
    #     if input_shape[-1]!=self._n:
    #         raise ValueError("Invalid input shape.")

    def call(self, llr_ch, u):
        """Successive cancellation (SC) decoding function.

        Performs successive cancellation decoding and returns the estimated
        information bits.

        Args:
            llr_ch (tf.float): Tensor of shape `[...,n]` containing the
                channel LLR values (as logits).

        Returns:
            `tf.float`: Tensor of shape `[...,k]` containing
            hard-decided estimations of all ``k`` information bits.

        Note:
            This function recursively unrolls the SC decoding tree, thus,
            for larger values of ``n`` building the decoding graph can become
            time consuming.
        """

        # Reshape inputs to [-1, n]
        input_shape = llr_ch.shape
        new_shape = [-1, self._n]
        llr_ch = tf.reshape(llr_ch, new_shape)

        llr_ch = -1. * llr_ch # logits are converted into "true" llrs

        # and decode
        u_hat_n, _ , llr_hat= self._polar_decode_sc_tf(llr_ch, u)

        # and recover the k information bit positions
        # u_hat = tf.gather(u_hat_n, self._info_pos, axis=1)
        #
        # # and reconstruct input shape
        # output_shape = input_shape.as_list()
        # output_shape[-1] = self.k
        # output_shape[0] = -1 # first dim can be dynamic (None)
        # u_hat_reshape = tf.reshape(u_hat, output_shape)
        return u_hat_n, llr_hat



class Polar5GDecoder_design(Block):
    # pylint: disable=line-too-long
    """Wrapper for 5G compliant decoding including rate-recovery and CRC removal.

    Parameters
    ----------
    enc_polar: Polar5GEncoder
        Instance of the :class:`~sionna.phy.fec.polar.encoding.Polar5GEncoder`
        used for encoding including rate-matching.

    dec_type: "SC" (default) | "SCL" | "hybSCL" | "BP"
        Defining the decoder to be used.
        Must be one of the following `{"SC", "SCL", "hybSCL", "BP"}`.

    list_size: int, (default 8)
        Defining the list size `iff` list-decoding is used.
        Only required for ``dec_types`` `{"SCL", "hybSCL"}`.

    num_iter: int, (default 20)
        Defining the number of BP iterations. Only required for ``dec_type``
        `"BP"`.

    return_crc_status: `bool`, (default `False`)
        If `True`,  the decoder additionally returns the CRC status indicating
        if a codeword was (most likely) correctly recovered.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    llr_ch: [...,n], tf.float
        Tensor containing the channel logits/llr values.

    Output
    ------
    b_hat : [...,k], tf.float
        Binary tensor containing hard-decided estimations of all `k`
        information bits.

    crc_status : [...], tf.bool
        CRC status indicating if a codeword was (most likely) correctly
        recovered. This is only returned if ``return_crc_status`` is True.
        Note that false positives are possible.

    Note
    ----
    This block supports the uplink and downlink Polar rate-matching scheme
    without `codeword segmentation`.

    Although the decoding `list size` is not provided by 3GPP
    [3GPPTS38212]_, the consortium has agreed on a `list size` of 8 for the
    5G decoding reference curves [Bioglio_Design]_.

    All list-decoders apply `CRC-aided` decoding, however, the non-list
    decoders (`"SC"` and `"BP"`) cannot materialize the CRC leading to an
    effective rate-loss.

    """

    def __init__(self,
                 enc_polar,
                 dec_type="SC",
                 list_size=8,
                 num_iter=20,
                 return_crc_status=False,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if not isinstance(enc_polar, Polar5GEncoder):
            raise TypeError("enc_polar must be Polar5GEncoder.")
        if not isinstance(dec_type, str):
            raise TypeError("dec_type must be str.")

        # list_size and num_iter are not checked here (done during decoder init)

        # Store internal attributes
        self._n_target = enc_polar.n_target
        self._k_target = enc_polar.k_target
        self._n_polar = enc_polar.n_polar
        self._k_polar = enc_polar.k_polar
        self._k_crc = enc_polar.enc_crc.crc_length
        self._bil = enc_polar._channel_type == "uplink"
        self._iil = enc_polar._channel_type == "downlink"
        self._llr_max = 100 # Internal max LLR value (for punctured positions)
        self._enc_polar = enc_polar
        self._dec_type = dec_type

        # Initialize the de-interleaver patterns
        self._init_interleavers()

        # Initialize decoder
        print("Warning: 5G Polar codes use an integrated CRC that " \
              "cannot be materialized with SC decoding and, thus, " \
              "causes a degraded performance. Please consider SCL " \
              "decoding instead.")
        self._polar_dec = PolarSCDecoder_prefectfrozen(self._enc_polar.frozen_pos,
                                         self._n_polar)


        if not isinstance(return_crc_status, bool):
            raise TypeError("return_crc_status must be bool.")

        self._return_crc_status = return_crc_status
        if self._return_crc_status: # init crc decoder
            if dec_type in ("SCL", "hybSCL"):
                # re-use CRC decoder from list decoder
                self._dec_crc = self._polar_dec._crc_decoder
            else: # init new CRC decoder for BP and SC
                self._dec_crc = CRCDecoder(self._enc_polar._enc_crc)

    ###############################
    # Public methods and properties
    ###############################

    @property
    def k_target(self):
        """Number of information bits including rate-matching"""
        return self._k_target

    @property
    def n_target(self):
        """Codeword length including rate-matching"""
        return self._n_target

    @property
    def k_polar(self):
        """Number of information bits of mother Polar code"""
        return self._k_polar

    @property
    def n_polar(self):
        """Codeword length of mother Polar code"""
        return self._n_polar

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding"""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding"""
        return self._info_pos

    @property
    def llr_max(self):
        """Maximum LLR value for internal calculations"""
        return self._llr_max

    @property
    def dec_type(self):
        """Decoder type used for decoding as str"""
        return self._dec_type

    @property
    def polar_dec(self):
        """Decoder instance used for decoding"""
        return self._polar_dec

    #################
    # Utility methods
    #################

    def _init_interleavers(self):
        """Initialize inverse interleaver patterns for rate-recovery."""

        # Channel interleaver
        ind_ch_int = self._enc_polar.channel_interleaver(
                                                np.arange(self._n_target))
        self.ind_ch_int_inv = np.argsort(ind_ch_int) # Find inverse perm

        # Sub-block interleaver
        ind_sub_int = self._enc_polar.subblock_interleaving(
                                                np.arange(self._n_polar))
        self.ind_sub_int_inv = np.argsort(ind_sub_int) # Find inverse perm

        # input bit interleaver
        if self._iil:
            self.ind_iil_inv = np.argsort(self._enc_polar.input_interleaver(
                                                np.arange(self._k_polar)))
        else:
            self.ind_iil_inv = None

    ########################
    # Sionna Block functions
    ########################

    # def build(self, input_shape):
    #     """Build and check if shape of input is invalid."""
    #     if input_shape[-1]!=self._n_target:
    #         raise ValueError("Invalid input shape.")

    def call(self, llr_ch, u):
        """Polar decoding and rate-recovery for uplink 5G Polar codes.

        Args:
            llr_ch (tf.float): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float`: Tensor of shape `[...,k]` containing
                hard-decided estimates of all ``k`` information bits.
        """

        input_shape = llr_ch.shape
        new_shape = [-1, self._n_target]
        llr_ch = tf.reshape(llr_ch, new_shape)

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
                                     self._n_polar-self._n_target], self.rdtype)
                llr_dematched = tf.concat([llr_zero, llr_deint], 1)
            else:
                # Shortening
                # Append n_polar - n_target "-infinity" llrs to last positions
                # Remark: we still operate with logits here, thus the neg. sign
                llr_infty = -self._llr_max * tf.ones([tf.shape(llr_deint)[0],
                                                self._n_polar-self._n_target],
                                                self.rdtype)
                llr_dematched = tf.concat([llr_deint, llr_infty], 1)

        # 3.) Remove subblock interleaving
        llr_dec = tf.gather(llr_dematched, self.ind_sub_int_inv, axis=1)

        # 4.) Run main decoder
        u_hat_crc, llrs = self._polar_dec(llr_dec, u)
        return llrs
        # 5.) Shortening should be implicitly recovered by decoder

        # 6.) Remove input bit interleaving for downlink channels only
        # if self._iil:
        #     u_hat_crc = tf.gather(u_hat_crc, self.ind_iil_inv, axis=1)
        #
        # # 7.) Evaluate or remove CRC (and PC)
        # if self._return_crc_status:
        #     # for compatibility with SC/BP, a dedicated CRC decoder is
        #     # used here (instead of accessing the interal SCL)
        #     u_hat, crc_status = self._dec_crc(u_hat_crc)
        # else: # just remove CRC bits
        #     u_hat = u_hat_crc
        #
        # # And reconstruct input shape
        # output_shape = input_shape.as_list()
        # output_shape[-1] = self._k_target
        # output_shape[0] = -1 # First dim can be dynamic (None)
        # u_hat_reshape = tf.reshape(u_hat, output_shape)
        # # and cast to internal rdtype (as subblocks may have different configs)
        # u_hat_reshape = tf.cast(u_hat_reshape, dtype=self.rdtype)
        #
        # if self._return_crc_status:
        #     # reconstruct CRC shape
        #     output_shape.pop() # remove last dimension
        #     crc_status = tf.reshape(crc_status, output_shape)
        #     return u_hat_reshape, crc_status
        #
        # else:
        #     return u_hat_reshape, llrs
