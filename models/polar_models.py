import re
import os
# from polarcodes import PolarCode, Construct

from time import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from click.core import batch
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Layer
from tensorflow.keras.optimizers import Adam, SGD
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
#from tensorflow_datasets.scripts.download_and_prepare import dataset

from models.sc_models import CheckNodeVanilla, BitNodeVanilla, CheckNodeTrellis, BitNodeTrellis, \
    BitNodeNNEmb, CheckNodeNNEmb, Embedding2LLR, Embedding2LLRwithSNR, Embedding2LLRTrellis, EmbeddingY, EmbeddingX, hard_dec, BatchNormModel, MyPolar5GEncoder,InterleavingMethod, EyModel, MyPreDecoder
from models.layers import SplitEvenOdd, Interleave, F2
from models.input_models import BinaryRNN
import wandb
from tensorflow import keras
import pandas as pd
import sionna as sn
from sionna.phy.fec.polar.encoding import PolarEncoder, Polar5GEncoder
from sionna.phy.fec.polar.decoding import PolarSCDecoder, PolarSCLDecoder, Polar5GDecoder
from sionna.phy.fec.polar.utils import generate_5g_ranking
from tqdm import tqdm
# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# tf.keras.backend.set_floatx('float32')
# dtype = tf.keras.backend.floatx()
dtype = tf.float32

class SCDecoder(Model):

    def __init__(self, channel, batch=100, eyN0=False, *args, **kwargs):
        Model.__init__(self)
        self.channel = channel[0]
        self.channel_design = channel[1]
        self.eyN0 = eyN0
        self.eyN0 = True
        self.batch = batch
        self.llr_enc_shape = self.llr_dec_shape = (1,)
        self.input_logits = tf.constant(0.0, dtype=dtype)
        self.Ex = self.Ex_enc = EmbeddingX(self.input_logits)
        self.Ey = self.channel.llr
        self.checknode_enc = self.checknode = CheckNodeVanilla()
        self.bitnode_enc = self.bitnode = BitNodeVanilla()
        self.emb2llr_enc = self.emb2llr = Activation(tf.identity)
        self.layer_norms_ex_enc = lambda x, i: x
        self.layer_norms_ex = lambda x, i: x
        self.layer_norms_ey = lambda x, i: x
        self.llr2prob = Activation(tf.math.sigmoid)
        self.split_even_odd = SplitEvenOdd(axis=1)
        self.f2 = F2()
        self.interleave = Interleave()
        self.hard_decision = Activation(hard_dec)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.NONE)
        # self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, axis=0,
        #                                                   reduction=tf.keras.losses.Reduction.NONE)
        self.loss_bn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.G2 = np.array([[1, 0], [1, 1]])
        self.v_lists = []
        self.pre_decoder = None
        self.encoder5g = None
        self.sorted_arg_errors = dict()

    def NeuralRateMatching(self, x, n_polar=512, E_size=576, sorted_indices=[]):
        batch_size, _, feature_dim = x.shape
        N = n_polar
        m = N.bit_length() - 1  # number of bits needed to represent indices
        indices = np.arange(N)
        self.bit_reversed_indices = np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])
        #np.arange(N)#np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])
        # self.bit_reversed_indices = self.relibility_seq_symbol(np.array(sorted_indices), self.channel.BPS)

        if E_size > n_polar:
            # Repetition coding
            indices = tf.concat([np.arange(N),self.bit_reversed_indices[:E_size-n_polar]], axis=0)
            # indices = []
            # for i in np.arange(N):
            #     if i in self.bit_reversed_indices[:E_size-n_polar]:
            #         indices.append(i)
            #     indices.append(i)
            E = tf.gather(x, indices, axis=1)

        elif E_size<n_polar:
            # puncturing
            #self.bit_reversed_indices = np.roll(self.bit_reversed_indices, shift=1)

            #E = tf.gather(x, np.sort(self.bit_reversed_indices[n_polar-E_size:]), axis=1)

            #### option 2 ######
            temp = np.sort(self.bit_reversed_indices[:(n_polar - E_size) // self.channel.BPS])
            if self.channel.BPS == 2:
                expanded_array = np.ravel([[x, x - 1] if x % 2 != 0 else [x, x + 1] for x in temp])
            else:
                expanded_array = temp
            all_indices = np.arange(n_polar)
            remaining_indices = np.setdiff1d(all_indices, expanded_array)
            E = tf.gather(x, remaining_indices, axis=1)


        else:
            E = x
        return E, indices

    def RecoverNeuralRateMatching(self, ey, N, bits_size, sorted_indices=[]):
        batch = ey.shape[0]
        E_size = ey.shape[1]
        #ey_norm = self.layer_norms_ey(ey, 0)
        # ex = self.Ex((ey.shape[0], ey.shape[1]//self.channel.BPS))
        # ex = tf.reshape(ex, (batch, bits_size, -1))

        ex = self.Ex([tf.constant(batch), tf.constant( N//self.channel.BPS)])
        #ex = self.Ex((batch, N))
        ex = tf.reshape(ex, (batch, N, -1))
        #self.bit_reversed_indices = sorted_indices
        ey = ey #+ ex
        if E_size > N:
            ind_rep = self.bit_reversed_indices[:bits_size - N]
            ind_rep2 = np.arange(N,bits_size)
            v_xor =  tf.zeros((batch, bits_size - N, 1))
            rep1 = tf.gather(ey, ind_rep, axis=1)
            rep2 = tf.gather(ey, ind_rep2, axis=1)
            ey_rep = self.bitnode.call((rep1,rep2,v_xor))
            ey_rev = tf.gather(ey[:,:N], self.bit_reversed_indices, axis=1)
            ey_ = tf.concat([ey_rep, ey_rev[:, bits_size - N:]], axis=1)
            ey_ = tf.gather(ey_, self.bit_reversed_indices, axis=1)
            #### adject
            # indices = []
            # for i in np.arange(N):
            #     if i in self.bit_reversed_indices[:E_size-N]:
            #         indices.append(i)
            #     indices.append(i)
            # ind_rep = np.where(np.diff(indices)==0)[0]
            # ind_rep2 = np.where(np.diff(indices)==0)[0] + 1
            # rep1 = tf.gather(ey, ind_rep, axis=1)
            # rep2 = tf.gather(ey, ind_rep2, axis=1)
            # v_xor =  tf.zeros((batch, bits_size - N, 1))
            # ey_rep = self.bitnode.call((rep1,rep2,v_xor))
            # ind = []
            # for i in np.arange(ey.shape[1]):
            #     if i not in ind_rep:
            #         ind.append(i)
            # ey_norep = tf.gather(ey, ind, axis=1)
            # X, Y, Z = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(self.bit_reversed_indices[:E_size-N], tf.int32),tf.range(ey_norep.shape[2], dtype=tf.int32), indexing='ij')
            # indices_rep = tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1), tf.reshape(Z, -1)], axis=1)
            # ey_ = tf.tensor_scatter_nd_update(ey_norep, indices_rep, tf.reshape(ey_rep, (-1)))

        elif E_size < N:
            ey_temp = self.Ex(tf.constant([batch, N // self.channel.BPS]))
            ey_temp = tf.reshape(ey_temp, (batch, N, -1))

            #X, Y, Z = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(np.sort(self.bit_reversed_indices[N-E_size:]), tf.int32),tf.range(ey_temp.shape[2], dtype=tf.int32), indexing='ij')

            ## work ###
            temp = np.sort(self.bit_reversed_indices[:(N - E_size) // self.channel.BPS])
            if self.channel.BPS == 2:
                expanded_array = np.ravel([[x, x - 1] if x % 2 != 0 else [x, x + 1] for x in temp])
            else:
                expanded_array = temp
            all_indices = np.arange(N)
            remaining_indices = np.setdiff1d(all_indices, expanded_array)
            X, Y, Z = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(np.sort(remaining_indices), tf.int32),tf.range(ey_temp.shape[2], dtype=tf.int32), indexing='ij')
            #######

            indices = tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1), tf.reshape(Z, -1)], axis=1)
            ey_ = tf.tensor_scatter_nd_update(ey_temp, indices, tf.reshape(ey, (-1)))
        else:
            ey_ = ey
        eyx_ = ey_
        return eyx_, ex

    @tf.function
    def encode(self, e, f, N, r, sample=True):
        """ sample=True indicates sampling bits at the recursion leaves.
            sample=False indicates taking the argmax at the recursion leaves.
        """
        if N == 1:
            llr_u = self.emb2llr_enc(e)
            if sample:
                pu = tf.math.sigmoid(llr_u)
                sampled_u = tf.where(tf.greater_equal(r, pu), 0.0, 1.0)
                x = tf.where(tf.equal(f, 0.5), sampled_u, f)
            else:
                x = tf.where(tf.equal(f, 0.5),
                             self.hard_decision(llr_u),
                             f)
            return x, tf.identity(x), llr_u

        e_odd, e_even = self.split_even_odd.call(e)
        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=1)
        r_left, r_right = r_halves

        # Compute soft mapping back one stage
        u1est = self.checknode_enc.call((e_odd, e_even))
        u1est = self.layer_norms_ex_enc(u1est, int(np.log2(N)))

        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, llr_u_left = self.encode(u1est, f_left, N // 2, r_left, sample=sample)


        # Using u1est and x1hard, we can estimate u2
        u2est = self.bitnode_enc.call((e_odd, e_even, u1hardprev))
        u2est = self.layer_norms_ex_enc(u2est, int(np.log2(N)))
        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, llr_u_right = self.encode(u2est, f_right, N // 2, r_right, sample=sample)

        u = tf.concat([uhat1, uhat2], axis=1)
        llr_u = tf.concat([llr_u_left, llr_u_right], axis=1)
        x = self.f2.call((u1hardprev, u2hardprev))
        x = self.interleave.call(x)
        return u, x, llr_u

    @tf.function
    def decode(self, ex, ey, f, N, r, sample=True):
        if N == 1:
            llr_u = self.emb2llr(ex)

            if sample:
                pu = tf.math.sigmoid(llr_u)
                frozen = tf.where(tf.greater_equal(r, pu), 0.0, 1.0)
            else:
                frozen = self.hard_decision(llr_u)
            # frozen = f
            llr_uy = self.emb2llr(ey)
            is_frozen = tf.logical_or(tf.equal(f, 0.0), tf.equal(f, 1.0))
            x = tf.where(is_frozen, frozen, self.hard_decision(llr_uy))
            return x, tf.identity(x), llr_u, llr_uy

        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=1)
        r_left, r_right = r_halves

        ey_odd, ey_even = self.split_even_odd.call(ey)
        ex_odd, ex_even = self.split_even_odd.call(ex)
        e_odd, e_even = tf.concat([ex_odd, ey_odd], axis=0), tf.concat([ex_even, ey_even], axis=0)

        # Compute soft mapping back one stage
        e1est = self.checknode.call((e_odd, e_even))
        ex1est, ey1est = tf.split(e1est, num_or_size_splits=2, axis=0)
        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, llr_u_left, llr_uy_left = self.decode(ex1est, ey1est, f_left, N // 2, r_left,
                                                                  sample=sample)

        # Using u1est and x1hard, we can estimate u2
        u1_hardprev_dup = tf.tile(u1hardprev, [2, 1, 1])
        e2est = self.bitnode.call((e_odd, e_even, u1_hardprev_dup))
        ex2est, ey2est = tf.split(e2est, num_or_size_splits=2, axis=0)
        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, llr_u_right, llr_uy_right = self.decode(ex2est, ey2est, f_right, N // 2, r_right,
                                                                    sample=sample)

        u = tf.concat([uhat1, uhat2], axis=1)
        llr_u = tf.concat([llr_u_left, llr_u_right], axis=1)
        llr_uy = tf.concat([llr_uy_left, llr_uy_right], axis=1)
        x = self.f2.call((u1hardprev, u2hardprev))
        x = self.interleave.call(x)
        return u, x, llr_u, llr_uy

    @tf.function
    def transform(self, u, N):
        if N == 1:
            return u
        else:
            # R_N maps odd/even indices (i.e., u1u2/u2) to first/second half
            # Compute odd/even outputs of (I_{N/2} \otimes G_2) transform
            u_odd, u_even = self.split_even_odd.call(u)
            x_left = self.transform(tf.math.floormod(u_odd + u_even, 2), N // 2)
            x_right = self.transform(u_even, N // 2)
            return tf.concat([x_left, x_right], axis=1)

    def f_rec(self, x, n):
        N = 2**n
        if n == 1:
            return np.array([tf.math.floormod(x[0] + x[1], 2),x[1]])
        v = self.f_rec(x[:N//2], n-1)
        v_tag = self.f_rec(x[N//2:N], n-1)
        xor_v = tf.math.floormod(v + v_tag, 2)
        f = np.ones((N))*np.nan
        f[::2] = xor_v
        f[1::2] = v_tag
        return f

    def NSCloss(self,e,u,L):
        N = u.shape[1]
        batch = u.shape[0]
        if N == 1:
            L+=self.loss_bn(self.emb2llr(e), u)
            return L,u
        e_odd, e_even = self.split_even_odd.call(e)
        # Compute soft mapping back one stage
        ec = self.checknode.call((e_odd, e_even)) # F theta

        # R_N^T maps u1est to top polar code
        L, v1 = self.NSCloss(ec, u[:,:N//2],L)

        # Using u1est and x1hard, we can estimate u2
        eb = self.bitnode.call((e_even, e_odd, v1))

        # R_N^T maps u2est to bottom polar code
        L, v2 = self.NSCloss(eb, u[:, N//2:], L)

        vt0 = tf.concat([v1, v2], axis=1)
        vt0 = tf.transpose(vt0, perm=[0, 2, 1])
        #vt0 = tf.ones((batch,1,N))
        kron = np.repeat(np.kron(tf.eye(N//2),self.G2).reshape(1,N,N), batch,axis=0)
        vt1 = tf.math.floormod(tf.matmul(vt0, kron),2) # v (IN/2 ⊗ G2)
        vt1 = tf.transpose(vt1, perm=[0, 2, 1])
        v = tf.concat([vt1[:,::2],vt1[:,1::2]],axis=1) # R_N
        L += self.loss_bn(self.emb2llr(e),v)

        return L, v

    def interleave_halves(self,N):
        # Ensure N is even for simplicity
        assert N % 2 == 0, "N must be even"
        return np.ravel(np.column_stack((np.arange(0, N // 2), np.arange(N // 2, N)))).tolist()

    @tf.function
    def fast_ce(self, e, v, full_bu_depth=None):
        batch = e.shape[0]
        N = e.shape[1]
        d = e.shape[2]

        if full_bu_depth is None:
            full_bu_depth = N
        loss_array = list()
        v_array = list()
        norm_array = list()
        pred_array = list()
        pred = self.emb2llr.call(e)
        loss = self.loss_fn(v, pred)
        norms = tf.norm(e, ord='euclidean', axis=-1)
        v_array.append(v)
        loss_array.append(loss)
        pred_array.append(tf.squeeze(pred, axis=2))
        norm_array.append(norms)

        layer_norm_pointer = 1
        # iterate over decoding stage
        while N > full_bu_depth:
            # compute bits amd embeddings in next layer
            v_odd, v_even = self.split_even_odd.call(v)
            e_odd, e_even = self.split_even_odd.call(e)

            # compute all the bits in the next stage
            v_xor = tf.math.floormod(v_odd + v_even, 2)

            # compute all the embeddings in the next stage
            e1_left = self.checknode.call((e_odd, e_even))
            e1_right = self.bitnode.call((e_odd, e_even, v_xor))

            e = tf.concat((e1_left, e1_right), axis=1)
            layer_norm_pointer = int(np.log2(N))
            e = self.layer_norms_ey(e, layer_norm_pointer)

            e_split = tf.split(e, num_or_size_splits=2, axis=1)
            e1_left, e1_right = e_split

            noise = tf.random.uniform(shape=(batch,), minval=0.0, maxval=1.0, dtype=tf.float32)
            noise = noise[:, None, None]
            cond_v = tf.tile(tf.greater_equal(noise, 0.5), [1, v_xor.shape[1], 1])
            cond_e = tf.tile(tf.greater_equal(noise, 0.5), [1, e_odd.shape[1], d])
            v = tf.where(cond_v, v_xor, v_even)
            e = tf.where(cond_e, e1_left, e1_right)
            norms = tf.norm(e, ord='euclidean', axis=-1)
            pred = self.emb2llr.call(e)
            loss = self.loss_fn(v, pred)
            loss_array.append(loss)
            pred_array.append(tf.squeeze(pred, axis=2))
            norm_array.append(norms)
            N //= 2

        # e = tf.stop_gradient(e)
        num_of_splits = 1
        V = list([v])
        E = list([e])
        # iterate over decoding stage
        while N > 1:
            V_1 = list([])
            V_2 = list([])
            E_1 = list([])
            E_2 = list([])
            # split into even and odd indices with respect to the depth
            for v, e in zip(V, E):
                # compute bits amd embeddings in next layer
                v_odd, v_even = self.split_even_odd.call(v)
                V_1.append(v_odd)
                V_2.append(v_even)
                e_odd, e_even = self.split_even_odd.call(e)
                E_1.append(e_odd)
                E_2.append(e_even)

            # compute all the bits in the next stage
            V_odd = tf.concat(V_1, axis=1)
            V_even = tf.concat(V_2, axis=1)
            v_xor = tf.math.floormod(V_odd + V_even, 2)
            V_xor = tf.split(v_xor, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            V_identity = tf.split(V_even, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            v = tf.concat([elem for pair in zip(V_xor, V_identity) for elem in pair], axis=1)
            V_ = tf.split(v, num_or_size_splits=2 ** num_of_splits, axis=1)

            # compute all the embeddings in the next stage
            E_odd = tf.concat(E_1, axis=1)
            E_even = tf.concat(E_2, axis=1)

            e1_left = self.checknode.call((E_odd, E_even))
            e1_right = self.bitnode.call((E_odd, E_even, v_xor))
            E1_left = tf.split(e1_left, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            E1_right = tf.split(e1_right, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            e = tf.concat([elem for pair in zip(E1_left, E1_right) for elem in pair], axis=1)
            layer_norm_pointer = int(np.log2(N))
            e = self.layer_norms_ey(e, layer_norm_pointer)
            norms = tf.norm(e, ord='euclidean', axis=-1)

            E_ = tf.split(e, num_or_size_splits=2 ** num_of_splits, axis=1)

            # on the last depth compute the CE of the synthetic channels

            pred = self.emb2llr.call(e)
            loss = self.loss_fn(v, pred)
            v_array.append(v)
            loss_array.append(loss)
            pred_array.append(tf.squeeze(pred, axis=2))
            norm_array.append(norms)

            V = V_
            E = E_

            N //= 2
            num_of_splits += 1
        return v_array, loss_array, pred_array, norm_array

    def relibility_seq_symbol(self, seq, bps):
        seq_sym = np.zeros((seq.shape[0]), dtype=seq.dtype)
        for i in np.arange(0, seq.shape[0], bps):
            for j in range(i, bps + i):
                seq_sym[i:i + bps] += np.where(j == seq)[0]
        return np.argsort(seq_sym)

    #@tf.function
    def forward_design(self, batch, N, ebno_db=None):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        ex_enc = self.Ex_enc.call(batch_N_shape)

        # generate shared randomness
        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)

        # create frozen bits for encoding. encoded bits need to be 0.5.
        f_enc = 0.5 * tf.ones(shape=(batch, N, 1))
        u, x, llr_u1 = self.encode(ex_enc, f_enc, N, r, sample=True)
        llr_u = tf.squeeze(tf.where(tf.equal(u, 1.0), llr_u1, -llr_u1), axis=2)
        h_u = tf.reduce_sum(-tf.math.log(tf.math.sigmoid(llr_u)), axis=0)
        # 5G encoder mode 
        # if self.encoder5g:
        #     x = self.encoder5g(u[...,0])[...,None]
        x_before_rm = x
        x, rep_ind = self.NeuralRateMatching(x, N, self.channel_design.E*self.channel_design.BPS)
        bits_size = x.shape[1]
        y = self.channel_design.sample_channel_outputs(x, ebno_db)

        #no = tf.repeat(self.channel_design.no, axis=1, repeats=y.shape[1])[..., None]

        #ey = self.Ey(tf.concat([y, no], axis=2))
        if self.eyN0:
            no = tf.ones((batch, y.shape[1], 1)) * self.channel.no
            if hasattr(self.emb2llr, 'no_source'):
                self.emb2llr.no_source = self.channel.no
                y = tf.concat([y, tf.math.log(no + 1e-10)], axis=2)
            else:
                y = tf.concat([y, no], axis=2)
        ey = self.Ey(y)
        ey = tf.reshape(ey, (batch, bits_size, -1))
        eyx_, ex = self.RecoverNeuralRateMatching(ey, N, bits_size)
        # if self.pre_decoder:
        #     eyx_ = self.pre_decoder.call(eyx_[...,0])[...,None]
        #     eyx_ = tf.gather(eyx_, self.bit_reversed_indices, axis=1)
        #     x_before_rm = self.pre_decoder.call(x[..., 0])[..., None]
        #     x_before_rm = tf.gather(x_before_rm, self.bit_reversed_indices, axis=1)

        u_2, loss_array, pred, norm_array = self.fast_ce( eyx_, x_before_rm[:,:N])
        errors = tf.cast(tf.not_equal(u[..., 0], self.hard_decision(pred[-1])), tf.float32)
        errors = tf.reduce_sum(errors, axis=0)
        h_uy = tf.reduce_sum(loss_array[-1], axis=0)
        return h_u, h_uy, errors

    #@tf.function
    def forward_eval(self, batch, N, info_indices, frozen_indices, A, Ac, ebno=None):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        # generate the information bits
        bits = tf.cast(tf.random.uniform((batch, N), minval=0, maxval=2, dtype=tf.int32), dtype)
        # create frozen bits for encoding. encoded bits need to be 0.5. info bit 0/1
        updates = 0.5 * tf.ones([batch * tf.shape(Ac)[0]], dtype=dtype)
        f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(bits, frozen_indices, updates), axis=2)
        #f_enc = tf.ones_like(f_enc,  dtype=dtype) * 0.5
        # generate shared randomness
        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)
        #r = tf.ones(shape=(batch, N, 1), dtype=tf.float32)
        # encode the bits into x^N and u^N
        ex_enc = self.Ex_enc.call(batch_N_shape)
        ex_enc = self.layer_norms_ex_enc(ex_enc, 0)
        u, x, llr_u1_enc = self.encode(ex_enc, f_enc, N, r, sample=True)
        y = self.channel.sample_channel_outputs(x, ebno)

        # create frozen bits for decoding. decoded bits need to be 0.5.
        # frozen bits are 0 decoded using argmax like in the encoder
        # tensor = tf.zeros(shape=(batch, N))
        tensor = tf.squeeze(u, axis=2)
        updates = 0.5 * tf.ones([batch * tf.shape(A)[0]], dtype=dtype)
        f_dec = tf.expand_dims(tf.tensor_scatter_nd_update(tensor, info_indices, updates), axis=2)
        # decode and compute the errors
        if self.eyN0:
            no = tf.ones((batch, y.shape[1], 1)) * self.channel.no
            if hasattr(self.emb2llr, 'no_source'):
                self.emb2llr.no_source = self.channel.no
                y = tf.concat([y, tf.math.log(no + 1e-10)], axis=2)
            else:
                y = tf.concat([y, no], axis=2)
        ey = self.Ey(y)
        ey = tf.reshape(ey, (batch, N, -1))

        ex = self.Ex.call(batch_N_shape)
        ex_ey_norm = self.layer_norms_ey(ex + ey, 0)
        ex_norm = self.layer_norms_ex(ex, 0)
        uhat, xhat, llr_u1, llr_uy1 = self.decode(ex_norm, ex_ey_norm, f_dec, f_dec.shape[1], r, sample=True)
        #uhat = self.hard_decision(llr_uy1)
        errors = tf.squeeze(tf.cast(tf.where(tf.equal(uhat, u), 0, 1), dtype), axis=-1)
        info_bit_errors = tf.gather(params=errors,
                                    indices=A,
                                    axis=1)
        return tf.reduce_mean(info_bit_errors, axis=1), errors

    def polar_code_design(self, n, batch, mc, tol=100, ebno_db=None):
        biterrd = tf.zeros([2 ** n], dtype=dtype)
        Hu = tf.zeros([2 ** n], dtype=dtype)
        Huy = tf.zeros([2 ** n], dtype=dtype)

        count = 0
        stop_err = max(int((2**n)*self.channel.CODERATE), 3)
        #stop_err = max(int((2**n)*0.25), 3)

        #stop_err = max(self.channel.rg.num_data_symbols.numpy()*self.channel.CODERATE, 3)


        def stop_criteria(arr):
            return np.sum(arr < tol)

        t = time()
        mi_list = []
        while stop_criteria(biterrd) > stop_err and count < mc:
            h_u, h_uy, errors = self.forward_design(batch, 2 ** n, ebno_db)
            Hu += h_u
            Huy += h_uy
            mi_list.append((h_u - h_uy)/(np.log(2)*batch))
            biterrd += errors
            count += batch

            if time()-t > 30:
                print(f'iter: {count/mc*100 :5.3f}% | bits w/o {tol} errors {stop_criteria(biterrd)} > {stop_err}')
                print(f'ber for k={stop_err}',
                      (np.cumsum(np.sort(biterrd / count)) / np.arange(1, 2 ** n + 1))[int(stop_err-1)])
                print(f"mc info bit:{(count*stop_err):.1e}")
                print('errors in info bits', np.sum(np.sort(biterrd)[:stop_err]))
                print(f'bler for k={stop_err} number bit', np.sum(np.sort(biterrd / count)[:int(stop_err)]))
                t = time()

        # print('no errors for K:', np.sum(biterrd == 0))
        # print(f'ber for k={np.sum(biterrd == 0)}', (np.cumsum(np.sort(biterrd/count)) / np.arange(1, 2 ** n + 1))[np.sum(biterrd == 0)])


        print(f'iter: {count/mc*100}% | bits w/o {tol} errors {stop_criteria(biterrd)} > {stop_err}')
        # N = 2**n
        # m = N.bit_length() - 1  # number of bits needed to represent indices
        # indices = np.arange(N)
        # bit_reversed_indices = np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])
        # f, _= generate_5g_ranking(0, N, sort=False)
        # plt.figure(); plt.plot(tf.gather(tf.gather(biterrd, bit_reversed_indices), f)); plt.show()
        # plt.figure(); plt.plot(tf.gather(biterrd, bit_reversed_indices)); plt.show()
        # plt.figure(); plt.plot(biterrd); plt.show()
        print(f'ber for k={stop_err}',
              (np.cumsum(np.sort(biterrd / count)) / np.arange(1, 2 ** n + 1))[int(stop_err - 1)])
        print(f"mc info bit:{(count * stop_err):.1e}")
        print(f'bler for k={stop_err}', np.sum(np.sort(biterrd / count)[:int(stop_err)]))

        biterrd /= count # plt.figure();plt.semilogy(np.cumsum(np.sort(biterrd))/np.arange(1, 2**n+1)); plt.show()
        bercumsums = np.cumsum(np.sort(biterrd)) / np.arange(1, 2 ** n + 1)
        # wandb.define_metric("bercumsum", step_metric="bit_num")
        # for i, bercumsum in enumerate(bercumsums):
        #     wandb.log({"bercumsum": bercumsum, "bit_num": i})
        #print(f'bler for k={stop_err} number bit', np.sum(bercumsums[:stop_err]))

        sorted_indices = np.argsort(biterrd)
        mi_array = np.array(mi_list)
        mi_array = mi_array[:, sorted_indices]
        micumsums = np.cumsum(mi_array, axis=0) / np.arange(1, mi_array.shape[0] + 1).reshape(-1, 1)
        df = pd.DataFrame(micumsums, columns=[f'Column {i}' for i in range(micumsums.shape[1])])
        data = df.values.tolist()
        #table = wandb.Table(data=data, columns=df.columns.tolist())
        #wandb.log({"micumsums_table": table})

        Hu /= count
        Huy /= count

        Huy = np.where(Huy > 0.693, 0.693, Huy)
        print(f'conditional entropies of effective bit channels:\n'
              f'Hu: {tf.reduce_mean(Hu).numpy()/np.log(2) : 5.4f} Huy: {tf.reduce_mean(Huy).numpy()/np.log(2) : 5.4f} '
              f'MI: {tf.reduce_mean(Hu - Huy).numpy()/np.log(2) : 5.4f}')
        mi = np.array(np.squeeze(Hu/np.log(2) - Huy/np.log(2)))
        #mi[mi < 0] = 0
        #  #np.argsort(-mi)
        sorted_arg_errors = np.argsort(-mi)

        sorted_arg_errors = np.argsort(biterrd)
        # alpha = 1
        # N = 2**n
        # position_penalty = alpha * (1.0 - np.arange(N) / N)
        # weighted_biterrd = biterrd * (1.0 + position_penalty)
        # sorted_arg_errors = np.argsort(weighted_biterrd)

        print('biterrd <= 1e-3', np.sum(np.cumsum(np.sort(biterrd)) <= 1e-3))
        print('BLER', np.sum(tf.gather(biterrd,sorted_arg_errors[:stop_err])))
        #sorted_arg_errors = self.relibility_seq_symbol(sorted_arg_errors, self.channel.BPS)

        #self.sorted_arg_errors[float(self.channel.snrdb[0])] = sorted_arg_errors
        #print(sorted_arg_errors[:stop_err])
        # # Read the existing CSV file
        # df = pd.read_csv("sorted_arg_errors.csv")
        # # Add a new column with sequential values
        # df[f"{self.channel.snrdb[0]} Sorted Errors"] = sorted_arg_errors
        # df[f"{self.channel.snrdb[0]} mi"] = mi
        # df[f"{self.channel.snrdb[0]} bercumsums"] = bercumsums
        # # Save the updated DataFrame back to the CSV file
        # df.to_csv("sorted_arg_errors.csv", index=False)
        #
        # print("New column added to sorted_arg_errors.csv")
        ch_ranking, _ = generate_5g_ranking(0, 2 ** n, sort=False)
        N = 2 ** n
        m = N.bit_length() - 1  # number of bits needed to represent indices
        indices = np.arange(N)
        self.bit_reversed_indices = np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])
        # ch_ranking_rev = ch_ranking[self.bit_reversed_indices]
        # plt.figure()
        # plt.plot(mi[sorted_arg_errors])
        # plt.show()

        return Hu/np.log(2), Huy/np.log(2), sorted_arg_errors

    def polar_code_err_prob(self, n, mc_err, batch, sorted_bit_channels, k, num_target_block_errors=100):
        A, Ac = self.choose_information_and_frozen_sets(sorted_bit_channels, k)
        #Ac, A = generate_5g_ranking(k, 2**n)

        X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(A, tf.int32))
        info_indices = tf.stack([tf.reshape(tf.transpose(X, perm=[1, 0]), -1),
                                 tf.reshape(tf.transpose(Y, perm=[1, 0]), -1)], axis=1)
        X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(Ac, tf.int32))
        frozen_indices = tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1)], axis=1)

        mc_err = (mc_err // batch + 1) * batch
        err = np.zeros(shape=0)
        t = time()
        biterrd = tf.zeros([2 ** n], dtype=dtype)
        count = 0
        block_errors = 0
        for i in range(0, mc_err, batch):
            bit_errors, errors, u, uhat = self.forward_eval(batch, 2 ** n, info_indices, frozen_indices, A, Ac)
            biterrd += np.sum(errors, axis=0)
            err = np.concatenate((err, bit_errors))
            block_errors += np.sum(bit_errors > 0)

            count+=batch
            try:
                froze_bit_errors = tf.gather(params=errors,
                                             indices=Ac,
                                             axis=1)
                if np.sum(froze_bit_errors) > 0:
                    print(f'frozen bit error detected',np.sum(froze_bit_errors))
            except:
                pass
            if time()-t > 30:
                ber = np.mean(err)
                fer = np.mean(err > 0)
                print(f'iter: {i/mc_err*100 :5.3f}% | ber: {ber : 5.3e} fer {fer : 5.3e}| block errors: {block_errors}')
                t = time()
                if block_errors >= num_target_block_errors:
                    break
        biterrd /= count # plt.figure();plt.semilogy(np.cumsum(np.sort(biterrd))/np.arange(1, 2**n+1)); plt.show()
        # bercumsums = np.cumsum(np.sort(biterrd)) / np.arange(1, 2 ** n + 1)
        # print('# zero error bits:', np.sum(biterrd==0))
        # print('# design eq to zero bits:', np.sum([i in A for i in np.argsort(np.array(biterrd))[0:307]]))
        # wandb.define_metric("bercumsum_decode", step_metric="bit_num_decode")
        # for i, bercumsum in enumerate(bercumsums):
        #     wandb.log({"bercumsum_decode": bercumsum, "bit_num_decode": i})
        return err

    def eval(self, Ns, mc_length=100000, code_rate=0.25, batch=100, tol=100, load_nsc_path=None, design_path=None, design_load=False, mc_design=10e7):
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
            decoder_name = load_nsc_path
        else:
            decoder_name = 'sc'

        bers, fers = list(), list()
        for n in Ns:
            log_dict = {}
            print(n)
            t = time()
            if design_path is None:
                try:
                    if design_load:
                        design_name = f"{self.construction_name(n, decoder_name)}:latest"
                        sorted_bit_channels = self.load_design(design_name)
                        print(f"Design loaded: {design_name}")
                    else:
                        raise Exception("design_load flags is False")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    Hu, Huy, sorted_bit_channels = self.polar_code_design(n, batch, mc_design, tol=tol)
                    self.save_design(n, sorted_bit_channels, decoder_name)
                    log_dict.update(self.design2dict(n, Hu, Huy))
                    log_dict.update({'mi': tf.reduce_mean(Hu*np.log(2) - Huy*np.log(2)).numpy() / np.log(2)})
            else:
                sorted_bit_channels = self.load_design(design_path)
            design_time = time() - t
            k = int(code_rate * (2 ** n))

            t = time()
            err = self.polar_code_err_prob(n, mc_length, batch, sorted_bit_channels, k)
            mc_time = time() - t
            ber = np.mean(err)
            fer = np.mean(err > 0)
            bers.append(ber)
            fers.append(fer)
            print(f"n: {n: 2d} design time: {design_time: 4.1f} "
                  f"code rate: {code_rate: 5.4f} #of mc-blocks: {mc_length} mc time: {mc_time: 4.1f} "
                  f"ber: {ber: 4.3e} fer: {fer: 4.3e}")
            log_dict.update({"n": n,
                             "ber": ber,
                             "fer": fer,
                             "code_rate": code_rate})
            wandb.log(log_dict)

        if len(Ns) != 1:
            x_values = np.array(Ns)
            y_values = np.array(bers)
            z_values = np.array(fers)
            data = [[x, y, z] for (x, y, z) in zip(x_values, y_values, z_values)]
            table = wandb.Table(data=data, columns=["n", "ber", "fer"])
            wandb.log({f"ber": table})

    def choose_information_and_frozen_sets(self, sorted_bit_channels, k):
        A = sorted_bit_channels[:k]
        #A = sorted(A)
        Ac = sorted_bit_channels[k:]
        #Ac = sorted(Ac)
        return A, Ac

    def set_input_logits(self, logits):
        self.Ex.input_logits.assign(logits)
        self.Ex_enc.input_logits.assign(logits)

    def construction_name(self, n, decoder_name):
        if len(decoder_name.split("/")) > 1:
            decoder_name = decoder_name.split("/")[-1]
        save_name = f'dec-{decoder_name}_channel-{self.channel.save_name}_n-{n}'.replace(':', "-")
        if len(save_name) >= 122:
            save_name = f'{decoder_name}{self.channel.save_name}{n}'.replace(':', "-")
        return save_name

    def save_design(self, n, sorted_bit_channels, decoder_name):
        try:
            save_name = self.construction_name(n, decoder_name)
            save_tmp_path = f"./artifacts/tmp/{wandb.run.name}/design.npy"
            # Extract the directory part of the path
            directory = os.path.dirname(save_tmp_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(save_tmp_path, sorted_bit_channels)
            weights_artifact = wandb.Artifact(save_name, type='design')

            # Add the model weights file to the artifact
            weights_artifact.add_file(save_tmp_path)

            # Log the artifact to W&B
            artifact_log = wandb.log_artifact(weights_artifact)
            artifact_log.wait()
            # print(f"model is saved as {save_name}")
            artifact = wandb.run.use_artifact(f"{save_name}:latest")
            print(f"Design: {artifact.name}")
        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def load_design(load_path, extend=''):
        artifact = wandb.use_artifact(load_path, type=f'design{extend}')
        artifact_dir = artifact.download()
        if extend:
            array_data = np.load(f'{artifact_dir}/design.npy', allow_pickle=True)
            array_data = dict(array_data.tolist())
        else:
            array_data = np.load(f'{artifact_dir}/design.npy')
        print(f"design is loaded from {load_path}")
        return array_data

    @staticmethod
    def log2wandb(n, ber, fer, Hu, Huy, code_rate):
        log_dict = {"n": n,
                    "ber": ber,
                    "fer": fer,
                    "code_rate": code_rate}
        Hu = np.squeeze(Hu)
        Huy = np.squeeze(Huy)
        mi = np.squeeze(Hu - Huy)

        x_values = np.array(range(len(Hu))).astype(int)
        y_values = np.array(np.sort(Hu))
        data_Hu = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_Hu = wandb.Table(data=data_Hu, columns=["channel_idx", "Hu"])
        y_values = np.array(np.sort(Huy))
        data_Huy = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_Huy = wandb.Table(data=data_Huy, columns=["channel_idx", "Huy"])
        y_values = np.array(np.sort(mi))
        data_mi = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_mi = wandb.Table(data=data_mi, columns=["channel_idx", "mi"])
        log_dict.update({f"Hu-n-{n}": wandb.plot.scatter(table_Hu, "channel_idx", "Hu", title=f"Hu-n-{n}"),
                         f"Huy-n-{n}": wandb.plot.scatter(table_Huy, "channel_idx", "Huy", title=f"Huy-n-{n}"),
                         f"mi-n-{n}": wandb.plot.scatter(table_mi, "channel_idx", "mi", title=f"mi-n-{n}")})
        wandb.log(log_dict)

    @staticmethod
    def design2dict(n, Hu, Huy):
        Hu = np.squeeze(Hu)
        Huy = np.squeeze(Huy)
        mi = np.squeeze(Hu - Huy)

        x_values = np.array(range(len(Hu))).astype(int)
        y_values = np.array(np.sort(Hu))
        data_Hu = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_Hu = wandb.Table(data=data_Hu, columns=["channel_idx", "Hu"])
        y_values = np.array(np.sort(Huy))
        data_Huy = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_Huy = wandb.Table(data=data_Huy, columns=["channel_idx", "Huy"])
        y_values = np.array(np.sort(mi))
        data_mi = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_mi = wandb.Table(data=data_mi, columns=["channel_idx", "mi"])
        log_dict = {f"Hu-n-{n}": wandb.plot.scatter(table_Hu, "channel_idx", "Hu", title=f"Hu-n-{n}"),
                    f"Huy-n-{n}": wandb.plot.scatter(table_Huy, "channel_idx", "Huy", title=f"Huy-n-{n}"),
                    f"mi-n-{n}": wandb.plot.scatter(table_mi, "channel_idx", "mi", title=f"mi-n-{n}")}
        return log_dict

    def get_parameters(self, decoder_name, Ns, mc_length=100000, code_rate=0.25, batch=100, tol=100, load_nsc_path=None, design_path=None,
             design_load=False, mc_design=10e7, ebno_db=None, design5G=False):

        #design5G = True
        bers, fers = list(), list()
        for n in Ns:
            log_dict = {}
            #print(n)
            t = time()
            k = int(code_rate * (2 ** n))
            if design5G:
                Ac, A = generate_5g_ranking(k, 2**n, sort=False)
                # ch_ranking, _  = generate_5g_ranking(0, 2**n, sort=False)
                # N = 2**n
                # m = N.bit_length() - 1  # number of bits needed to represent indices
                # indices = np.arange(N)
                # self.bit_reversed_indices = np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])
                # ch_ranking_rev = ch_ranking[self.bit_reversed_indices]
                # Ac = ch_ranking_rev[:k]
                # A = ch_ranking_rev[k:]

                # Ac = list(Ac)
                # Ac.append(A[-1])
                # A = A[:-1]
            else:
                if design_path is None:
                    if design_load:
                        design_name = f"{self.construction_name(n, decoder_name)}:latest"
                        sorted_bit_channels = self.load_design(design_name)
                        print(f"Design loaded: {design_name}")
                    else:
                        Hu, Huy, sorted_bit_channels = self.polar_code_design(n, batch, mc_design, tol=tol, ebno_db=ebno_db)
                        #self.save_design(n, sorted_bit_channels, decoder_name)
                        log_dict.update(self.design2dict(n, Hu, Huy))
                        log_dict.update({'mi': tf.reduce_mean(Hu*np.log(2) - Huy*np.log(2)).numpy() / np.log(2)})
                else:
                    if "all" in design_path:
                        sorted_bit_channels = self.load_design(design_path,"_all")[int(ebno_db)][2][2**n]
                    else:
                        sorted_bit_channels = self.load_design(design_path)
                design_time = time() - t
                t = time()
                A, Ac = self.choose_information_and_frozen_sets(sorted_bit_channels, k)

            X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(A, tf.int32))
            info_indices = tf.stack([tf.reshape(tf.transpose(X, perm=[1, 0]), -1),
                                     tf.reshape(tf.transpose(Y, perm=[1, 0]), -1)], axis=1)
            X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(Ac, tf.int32))
            frozen_indices = tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1)], axis=1)

        return batch, n, info_indices, frozen_indices, A, Ac

class SCListDecoder(SCDecoder):
    def __init__(self, channel, batch=100, list_num=4, crc=None, crc_oracle=None, sn=False, eyN0=False, *args, **kwargs):
        SCDecoder.__init__(self, channel, batch, eyN0=eyN0,*args, **kwargs)
        self.nL = list_num
        self.crc = crc
        if list_num == 1:
            self.crc = None
        if self.crc is not None:
            self.crc_enc = CRCEncoder(crc_degree=crc)
            self.crc_dec = CRCDecoder(crc_encoder=self.crc_enc)
            self.crc_oracle = crc_oracle

        self.checknode_list = CheckNodeVanilla()
        self.bitnode_list = BitNodeVanilla()
        self.emb2llr_list = Activation(tf.identity)
        self.split_even_odd_list = SplitEvenOdd(axis=2)
        self.f2_list = F2(axis=2)
        self.interleave_list = Interleave(axis=2)
        sn = True
        # self.sn = sn
        # self.encoder5g = None
        # self.pre_decoder = None
        # if sn:
        #     N = int(self.channel.N)
        #     k = int(N * self.channel.CODERATE)
        #     self.encoder5g = MyPolar5GEncoder(k=k, n=self.channel.E * self.channel.BPS, verbose=True, list_size=list_num,
        #                                     channel_type='uplink', return_u_and_llrs=True)
        #     m = N.bit_length() - 1  # number of bits needed to represent indices
        #     indices = np.arange(N)
        #     self.reversed_indices = np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])
        #     dec_type = 'SCL'
        #     self.pre_decoder = MyPreDecoder(self.encoder5g, dec_type='SCL', list_size=list_num)
        #     if list_num == 1:
        #         self.pre_decoder._k_crc = -k

    @tf.function
    def decode_list(self, ex, ey, f, pm, N, r, sample=True, s0=None):
        if N == 1:
            llr_u = self.emb2llr_list(ex)
            if sample:
                pu = tf.math.sigmoid(llr_u)
                frozen = tf.where(tf.greater_equal(r, pu), 0.0, 1.0)
            else:
                frozen = self.hard_decision(llr_u)
            frozen = f
            dm = self.emb2llr_list.call(ey)
            hd_ = self.hard_decision.call(tf.squeeze(dm, axis=(2, 3)))
            hd = tf.concat((hd_, 1 - hd_), axis=1)
            pm_dup = tf.concat((pm, pm + tf.abs(tf.squeeze(dm, axis=(2, 3)))), -1)
            pm_prune, prune_idx_ = tf.math.top_k(-pm_dup, k=self.nL, sorted=True)
            pm_prune = -pm_prune
            prune_idx = tf.sort(prune_idx_, axis=1)
            idx = tf.argsort(prune_idx_, axis=1)
            pm_prune = tf.gather(pm_prune, idx, axis=1, batch_dims=1)
            u_survived = tf.gather(hd, prune_idx, axis=1, batch_dims=1)[:, :, tf.newaxis, tf.newaxis]

            is_frozen = tf.not_equal(f, 0.5)
            x = tf.where(is_frozen, frozen, u_survived)
            pm_ = tf.where(tf.squeeze(is_frozen, axis=(2, 3)),
                           pm
                           + tf.abs(tf.squeeze(dm, axis=(2, 3))) *
                           tf.cast(tf.squeeze(tf.not_equal(tf.expand_dims(tf.expand_dims(hd_, -1), -1), frozen),
                                             axis=(2, 3)), tf.float32),
                           pm_prune)
            new_order = tf.tile(tf.expand_dims(tf.range(self.nL), 0), [ey.shape[0], 1]) \
                if f[0, 0, 0, 0] != 0.5 else (prune_idx % self.nL)
            return x, tf.identity(x), llr_u, dm, pm_, new_order

        f_halves = tf.split(f, num_or_size_splits=2, axis=2)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=2)
        r_left, r_right = r_halves

        ey_odd, ey_even = self.split_even_odd_list.call(ey)
        ex_odd, ex_even = self.split_even_odd_list.call(ex)
        e_odd, e_even = tf.concat([ex_odd, ey_odd], axis=0), tf.concat([ex_even, ey_even], axis=0)

        # Compute soft mapping back one stage
        e1est = self.checknode_list.call((e_odd, e_even))
        ex1est, ey1est = tf.split(e1est, num_or_size_splits=2, axis=0)

        uhat1, u1hardprev, llr_u_left, llr_uy_left, pm, new_order = self.decode_list(ex1est, ey1est, f_left, pm,
                                                                                     N // 2, r_left, sample=sample, s0=s0)

        new_order_dup = tf.concat((new_order,
                                   new_order), axis=0)
        e_odd = tf.gather(e_odd, new_order_dup, axis=1, batch_dims=1)
        e_even = tf.gather(e_even, new_order_dup, axis=1, batch_dims=1)

        # Using u1est and x1hard, we can estimate u2
        u1_hardprev_dup = tf.tile(u1hardprev, [2, 1, 1, 1])
        e2est = self.bitnode_list.call((e_odd, e_even, u1_hardprev_dup))
        ex2est, ey2est = tf.split(e2est, num_or_size_splits=2, axis=0)

        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, llr_u_right, llr_uy_right, pm, new_order2 = self.decode_list(ex2est, ey2est, f_right, pm,
                                                                                        N // 2, r_right, sample=sample, s0=s0)
        uhat1 = tf.gather(uhat1, new_order2, axis=1, batch_dims=1)
        llr_u_left = tf.gather(llr_u_left, new_order2, axis=1, batch_dims=1)
        llr_uy_left = tf.gather(llr_uy_left, new_order2, axis=1, batch_dims=1)
        u1hardprev = tf.gather(u1hardprev, new_order2, axis=1, batch_dims=1)
        new_order = tf.gather(new_order, new_order2, axis=1, batch_dims=1)
        u = tf.concat([uhat1, uhat2], axis=2)
        llr_u = tf.concat([llr_u_left, llr_u_right], axis=2)
        llr_uy = tf.concat([llr_uy_left, llr_uy_right], axis=2)
        x = self.f2_list.call((u1hardprev, u2hardprev))
        x = self.interleave_list.call(x)
        return u, x, llr_u, llr_uy, pm, new_order


    #@tf.function
    def forward_eval(self, batch, N, info_indices, frozen_indices, A, Ac, ebno=None):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        f_enc = tf.ones(shape=(batch, N)) * 0.5
        if self.crc is None:
            info_bits = tf.cast(tf.random.uniform((batch, tf.shape(A)[0],), minval=0, maxval=2, dtype=tf.int32), dtype)
            f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(f_enc, info_indices, tf.reshape(info_bits, [-1])), axis=2)
        else:
            info_bits_num = len(A) - self.crc_enc.crc_length
            info_bits = tf.cast(tf.random.uniform((batch, info_bits_num,), minval=0, maxval=2, dtype=tf.int32), dtype)
            bits = self.crc_enc(info_bits)
            bits = tf.reverse(bits, axis=[-1])
            # bits = tf.transpose(bits)
            f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(f_enc, info_indices, tf.reshape(bits, [-1])), axis=2)

        # generate shared randomness
        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)
        #r = tf.ones_like(r)

        # encode the bits into x^N and u^N
        ex_enc = self.Ex_enc.call(batch_N_shape)
        u, x, llr_u1_enc = self.encode(ex_enc, f_enc, N, r, sample=True)


        ### Rate Matching ###
        x, rep_ind = self.NeuralRateMatching(x, N, self.channel.E*self.channel.BPS, np.concatenate([A,Ac]))
        bits_size = x.shape[1]
        y = self.channel.sample_channel_outputs(x, ebno)
        # create frozen bits for decoding. decoded bits need to be 0.5.
        # frozen bits are 0 decoded using argmax like in the encoder
        #tensor = tf.zeros(shape=(batch, N))
        tensor = tf.squeeze(u, axis=2)
        updates = 0.5 * tf.ones([batch * tf.shape(A)[0]], dtype=dtype)
        f_dec = tf.expand_dims(tf.tensor_scatter_nd_update(tensor, info_indices, updates), axis=2)
        f_dec = tf.tile(tf.expand_dims(f_dec, 1), [1, self.nL, 1, 1])
        r = tf.tile(tf.expand_dims(r, 1), [1, self.nL, 1, 1])

        if self.eyN0:
            no = tf.ones((batch, y.shape[1], 1)) * self.channel.no
            if hasattr(self.emb2llr, 'no_source'):
                self.emb2llr.no_source = self.channel.no
                y = tf.concat([y, tf.math.log(no+1e-10)], axis=2)
            else:
                y = tf.concat([y, no], axis=2)

        ey = self.Ey(y)
        ey = tf.reshape(ey, (batch, bits_size, -1))
        eyx, ex = self.RecoverNeuralRateMatching(ey, N, bits_size, np.concatenate([A, Ac]))
        ey_ = tf.expand_dims(eyx, 1)
        ex_ = tf.expand_dims(ex, 1)
        repmat = tf.tensor_scatter_nd_update(tensor=tf.ones_like(tf.shape(ex_)),
                                             indices=tf.constant([[1]]),
                                             updates=tf.constant([self.nL]))
        ex_dup = tf.tile(ex_, repmat)
        ey_dup = tf.tile(ey_, repmat)
        pm = tf.concat([tf.zeros([1]), tf.ones([self.nL-1])*float('inf')], 0)
        pm = tf.tile(tf.expand_dims(pm, 0), [u.shape[0], 1])
        t = time()
        uhat_list, xhat, llr_u, llr_uy, pm, new_order = self.decode_list(ex_dup, ey_dup, f_dec, pm,
                                                                         f_dec.shape[2], r, sample=True)
        if self.crc is None:
            len_crc = 0
            uhat = self.choose_codeword_pm(uhat_list, pm)
        else:
            len_crc = self.crc_enc.crc_length

            if self.crc_oracle:
                uhat = self.choose_codeword_oracle(uhat_list, u)
            else:
                uhat = self.choose_codeword_crc(uhat_list, pm, A)
        errors = tf.cast(tf.where(tf.equal(uhat, u), 0, 1), tf.float32)
        u_info = tf.gather(params=u,
                                    indices=A[len_crc:],
                                    axis=1)[...,0]
        uhat_info = tf.gather(params=uhat,
                                    indices=A[len_crc:],
                                    axis=1)[...,0]
        info_bit_errors = tf.gather(params=errors,
                                    indices=A[len_crc:],
                                    axis=1)

        return tf.squeeze(tf.reduce_mean(info_bit_errors, axis=1), axis=1), errors, u_info, uhat_info

    def forward_eval_debug(self, batch, N, info_indices, frozen_indices, A, Ac, ebno=None):
        """Like forward_eval but also computes per-block diagnostics to understand BER/BLER gap.

        Returns:
            ber_per_block  (batch,)     mean info-bit error fraction per block
            errors         (batch,N,1)  per-bit errors of the selected codeword
            u_info         (batch,k)    true info bits
            uhat_info      (batch,k)    decoded info bits
            diag           dict with numpy arrays of shape (batch,):
              true_in_list          1 if oracle can recover the true codeword from the list
              any_crc_pass          1 if at least one list path passes CRC
              num_crc_pass          number of list paths that pass CRC
              has_error             1 if block has >= 1 info-bit error
              info_errors_count     number of wrong info bits in the block
              type_true_not_in_list errored blocks where true codeword was not in list (irreversible)
              type_crc_no_pass      errored blocks where true was in list but no path passed CRC
              type_crc_false_pos    errored blocks where a wrong path was selected despite true being in list
        """
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        f_enc = tf.ones(shape=(batch, N)) * 0.5
        if self.crc is None:
            len_crc = 0
            info_bits = tf.cast(tf.random.uniform((batch, tf.shape(A)[0],), minval=0, maxval=2, dtype=tf.int32), dtype)
            f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(f_enc, info_indices, tf.reshape(info_bits, [-1])), axis=2)
        else:
            len_crc = self.crc_enc.crc_length
            info_bits_num = len(A) - len_crc
            info_bits = tf.cast(tf.random.uniform((batch, info_bits_num,), minval=0, maxval=2, dtype=tf.int32), dtype)
            bits = self.crc_enc(info_bits)
            bits = tf.reverse(bits, axis=[-1])
            f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(f_enc, info_indices, tf.reshape(bits, [-1])), axis=2)

        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)
        ex_enc = self.Ex_enc.call(batch_N_shape)
        u, x, llr_u1_enc = self.encode(ex_enc, f_enc, N, r, sample=True)

        x, rep_ind = self.NeuralRateMatching(x, N, self.channel.E * self.channel.BPS, np.concatenate([A, Ac]))
        bits_size = x.shape[1]
        y = self.channel.sample_channel_outputs(x, ebno)

        tensor = tf.squeeze(u, axis=2)
        updates = 0.5 * tf.ones([batch * tf.shape(A)[0]], dtype=dtype)
        f_dec = tf.expand_dims(tf.tensor_scatter_nd_update(tensor, info_indices, updates), axis=2)
        f_dec = tf.tile(tf.expand_dims(f_dec, 1), [1, self.nL, 1, 1])
        r = tf.tile(tf.expand_dims(r, 1), [1, self.nL, 1, 1])

        if self.eyN0:
            no = tf.ones((batch, y.shape[1], 1)) * self.channel.no
            if hasattr(self.emb2llr, 'no_source'):
                self.emb2llr.no_source = self.channel.no
                y = tf.concat([y, tf.math.log(no + 1e-10)], axis=2)
            else:
                y = tf.concat([y, no], axis=2)

        ey = self.Ey(y)
        ey = tf.reshape(ey, (batch, bits_size, -1))
        eyx, ex = self.RecoverNeuralRateMatching(ey, N, bits_size, np.concatenate([A, Ac]))
        ey_ = tf.expand_dims(eyx, 1)
        ex_ = tf.expand_dims(ex, 1)
        repmat = tf.tensor_scatter_nd_update(tensor=tf.ones_like(tf.shape(ex_)),
                                             indices=tf.constant([[1]]),
                                             updates=tf.constant([self.nL]))
        ex_dup = tf.tile(ex_, repmat)
        ey_dup = tf.tile(ey_, repmat)
        pm_init = tf.concat([tf.zeros([1]), tf.ones([self.nL - 1]) * float('inf')], 0)
        pm_init = tf.tile(tf.expand_dims(pm_init, 0), [u.shape[0], 1])

        uhat_list, xhat, llr_u, llr_uy, pm, new_order = self.decode_list(
            ex_dup, ey_dup, f_dec, pm_init, f_dec.shape[2], r, sample=True)

        # --- path metric statistics (batch, nL) -> (batch,) ---
        # pm lower = more likely; all nL paths have finite values after full decoding
        pm_best   = tf.reduce_min(pm, axis=1)               # PM of the selected (most likely) path
        pm_worst  = tf.reduce_max(pm, axis=1)               # PM of the least likely surviving path
        pm_spread = pm_worst - pm_best                      # differentiation between paths
        pm_mean   = tf.reduce_mean(pm, axis=1)              # average PM across all paths
        pm_std    = tf.math.reduce_std(pm, axis=1)          # spread around the mean

        # regular codeword selection (same logic as forward_eval)
        if self.crc is None:
            uhat = self.choose_codeword_pm(uhat_list, pm)
        else:
            uhat = self.choose_codeword_crc(uhat_list, pm, A)

        errors = tf.cast(tf.where(tf.equal(uhat, u), 0, 1), tf.float32)
        u_info = tf.gather(params=u, indices=A[len_crc:], axis=1)[..., 0]
        uhat_info = tf.gather(params=uhat, indices=A[len_crc:], axis=1)[..., 0]
        info_bit_errors = tf.gather(params=errors, indices=A[len_crc:], axis=1)
        ber_per_block = tf.squeeze(tf.reduce_mean(info_bit_errors, axis=1), axis=1)
        info_errors_count = tf.cast(tf.reduce_sum(info_bit_errors, axis=[1, 2]), tf.float32)
        has_error = tf.cast(info_errors_count > 0, tf.float32)

        # oracle: is the true codeword anywhere in the list?
        uhat_oracle = self.choose_codeword_oracle(uhat_list, u)
        oracle_errors = tf.cast(tf.where(tf.equal(uhat_oracle, u), 0, 1), tf.float32)
        oracle_info_err_count = tf.cast(tf.reduce_sum(
            tf.gather(params=oracle_errors, indices=A[len_crc:], axis=1), axis=[1, 2]), tf.float32)
        true_in_list = tf.cast(tf.equal(oracle_info_err_count, 0), tf.float32)

        # CRC pass count per path (paths sorted by path metric)
        if self.crc is not None:
            sort_ = tf.argsort(pm)
            uhat_list_sorted = tf.gather(params=uhat_list, indices=sort_, axis=1, batch_dims=1)
            crcs_per_path = []
            for i in range(self.nL):
                uhat_info_hat = tf.gather(
                    params=uhat_list_sorted[:, i, :, 0],
                    indices=tf.tile(tf.expand_dims(A, 0), [tf.shape(uhat_list)[0], 1]),
                    batch_dims=1)
                _, crc_v = self.crc_dec(tf.reverse(uhat_info_hat, axis=[-1]))
                crcs_per_path.append(tf.cast(crc_v, tf.float32))
            crc_valid_all = tf.concat(crcs_per_path, axis=-1)  # (batch, nL)
            any_crc_pass = tf.cast(tf.reduce_any(tf.cast(crc_valid_all, tf.bool), axis=-1), tf.float32)
            num_crc_pass = tf.reduce_sum(crc_valid_all, axis=-1)
        else:
            any_crc_pass = tf.ones((batch,), dtype=tf.float32)
            num_crc_pass = tf.ones((batch,), dtype=tf.float32)

        # classify errored blocks into three mutually exclusive types
        type_true_not_in_list = (1.0 - true_in_list) * has_error
        type_crc_no_pass      = true_in_list * (1.0 - any_crc_pass) * has_error
        type_crc_false_pos    = true_in_list * any_crc_pass * has_error

        diag = {
            'true_in_list':          true_in_list.numpy(),
            'any_crc_pass':          any_crc_pass.numpy(),
            'num_crc_pass':          num_crc_pass.numpy(),
            'has_error':             has_error.numpy(),
            'info_errors_count':     info_errors_count.numpy(),
            'type_true_not_in_list': type_true_not_in_list.numpy(),
            'type_crc_no_pass':      type_crc_no_pass.numpy(),
            'type_crc_false_pos':    type_crc_false_pos.numpy(),
            # path metric stats — shape (batch,)
            'pm_best':               pm_best.numpy(),    # PM of selected path (lower = more confident)
            'pm_worst':              pm_worst.numpy(),   # PM of least likely survivor
            'pm_spread':             pm_spread.numpy(),  # max-min: how differentiated the list is
            'pm_mean':               pm_mean.numpy(),    # average PM across paths
            'pm_std':                pm_std.numpy(),     # std across paths
        }
        return ber_per_block, errors, u_info, uhat_info, diag

    def forward_finetune(self, batch, N, info_indices, frozen_indices, A, Ac, ebno=None):
        """Like forward_eval but returns (pm, u, uhat_list, errors) for path-metric fine-tuning.

        pm:        (batch, nL)       path metrics after decoding; lower = more likely
        u:         (batch, N, 1)     true codeword bits
        uhat_list: (batch, nL, N, 1) all surviving list paths (hard decisions)
        errors:    (batch, N, 1)     per-bit errors of the CRC/PM selected path

        Gradient flows through pm (via pm += |emb2llr(ey)|) to emb2llr weights.
        """
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        f_enc = tf.ones(shape=(batch, N)) * 0.5
        if self.crc is None:
            info_bits = tf.cast(tf.random.uniform((batch, tf.shape(A)[0],), minval=0, maxval=2, dtype=tf.int32), dtype)
            f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(f_enc, info_indices, tf.reshape(info_bits, [-1])), axis=2)
        else:
            info_bits_num = len(A) - self.crc_enc.crc_length
            info_bits = tf.cast(tf.random.uniform((batch, info_bits_num,), minval=0, maxval=2, dtype=tf.int32), dtype)
            bits = self.crc_enc(info_bits)
            bits = tf.reverse(bits, axis=[-1])
            f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(f_enc, info_indices, tf.reshape(bits, [-1])), axis=2)

        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)
        ex_enc = self.Ex_enc.call(batch_N_shape)
        u, x, llr_u1_enc = self.encode(ex_enc, f_enc, N, r, sample=True)

        x, rep_ind = self.NeuralRateMatching(x, N, self.channel.E * self.channel.BPS, np.concatenate([A, Ac]))
        bits_size = x.shape[1]
        y = self.channel.sample_channel_outputs(x, ebno)

        tensor = tf.squeeze(u, axis=2)
        updates = 0.5 * tf.ones([batch * tf.shape(A)[0]], dtype=dtype)
        f_dec = tf.expand_dims(tf.tensor_scatter_nd_update(tensor, info_indices, updates), axis=2)
        f_dec = tf.tile(tf.expand_dims(f_dec, 1), [1, self.nL, 1, 1])
        r = tf.tile(tf.expand_dims(r, 1), [1, self.nL, 1, 1])

        if self.eyN0:
            no = tf.ones((batch, y.shape[1], 1)) * self.channel.no
            y = tf.concat([y, no], axis=2)

        ey = self.Ey(y)
        ey = tf.reshape(ey, (batch, bits_size, -1))
        eyx, ex = self.RecoverNeuralRateMatching(ey, N, bits_size, np.concatenate([A, Ac]))
        ey_ = tf.expand_dims(eyx, 1)
        ex_ = tf.expand_dims(ex, 1)
        repmat = tf.tensor_scatter_nd_update(tensor=tf.ones_like(tf.shape(ex_)),
                                             indices=tf.constant([[1]]),
                                             updates=tf.constant([self.nL]))
        ex_dup = tf.tile(ex_, repmat)
        ey_dup = tf.tile(ey_, repmat)
        pm_init = tf.concat([tf.zeros([1]), tf.ones([self.nL - 1]) * float('inf')], 0)
        pm_init = tf.tile(tf.expand_dims(pm_init, 0), [u.shape[0], 1])

        uhat_list, xhat, llr_u, llr_uy, pm, new_order = self.decode_list(
            ex_dup, ey_dup, f_dec, pm_init, f_dec.shape[2], r, sample=True)

        if self.crc is None:
            uhat = self.choose_codeword_pm(uhat_list, pm)
        else:
            uhat = self.choose_codeword_crc(uhat_list, pm, A)
        errors = tf.cast(tf.where(tf.equal(uhat, u), 0, 1), tf.float32)
        return pm, u, uhat_list, errors, eyx

    @staticmethod
    def choose_codeword_pm(uhat_list, pm):
        uhat = tf.gather(uhat_list, tf.argmin(pm, axis=1), axis=1, batch_dims=1)
        return uhat

    def choose_codeword_crc(self, uhat_list, pm, A):
        sort_ = tf.argsort(pm)
        uhat_list = tf.gather(params=uhat_list, indices=sort_, axis=1, batch_dims=1)
        crcs = []
        #uhat_list2 = []
        for i in range(self.nL):
            uhat_info_hat = tf.gather(params=uhat_list[:, i, :, 0],
                                      indices=tf.tile(tf.expand_dims(A, 0), [tf.shape(uhat_list)[0], 1]),
                                      batch_dims=1)
            _, crc_valid = self.crc_dec(tf.reverse(uhat_info_hat, axis=[-1]))
            #_, crc_valid = self.crc_dec(uhat_info_hat)
            #uhat_list2.append(uhat)
            crcs.append(crc_valid)
        crc_valid = tf.concat(crcs, axis=-1)
        chosen_idx = tf.argmax(crc_valid, axis=-1)
        uhat = tf.gather(uhat_list, indices=chosen_idx, batch_dims=1, axis=1)
        return uhat

    def choose_codeword_oracle(self, uhat_list, u):
        oracle_valid = tf.reduce_all(tf.equal(uhat_list,
                                              tf.tile(tf.expand_dims(u, 1), [1, self.nL, 1, 1])),
                                     axis=(2, 3))
        chosen_idx_oracle = tf.argmax(oracle_valid, axis=-1)
        uhat_oracle = tf.gather(uhat_list, indices=chosen_idx_oracle, batch_dims=1)
        return uhat_oracle

    def polar_code_err_prob_debug(self, n, mc_err, batch, sorted_bit_channels, k,
                                   num_target_block_errors=200):
        """Monte-Carlo loop using forward_eval_debug to diagnose the BER/BLER gap.

        Returns:
            err     numpy array of per-block mean info-bit error fractions
            summary dict with aggregated statistics
        """
        A, Ac = self.choose_information_and_frozen_sets(sorted_bit_channels, k)
        X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(A, tf.int32))
        info_indices = tf.stack([tf.reshape(tf.transpose(X, perm=[1, 0]), -1),
                                 tf.reshape(tf.transpose(Y, perm=[1, 0]), -1)], axis=1)
        X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(Ac, tf.int32))
        frozen_indices = tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1)], axis=1)

        mc_err = (mc_err // batch + 1) * batch
        err = np.zeros(0)
        t = time()
        block_errors = 0
        total_blocks = 0
        total_true_not_in_list = 0
        total_crc_no_pass = 0
        total_crc_false_pos = 0
        all_error_counts = []
        # path metric accumulators — separate correct vs errored blocks
        pm_best_correct,  pm_spread_correct,  pm_mean_correct,  pm_std_correct  = [], [], [], []
        pm_best_errored,  pm_spread_errored,  pm_mean_errored,  pm_std_errored  = [], [], [], []

        for i in range(0, mc_err, batch):
            ber_per_block, errors, u, uhat, diag = self.forward_eval_debug(
                batch, 2 ** n, info_indices, frozen_indices, A, Ac)
            err = np.concatenate((err, ber_per_block.numpy()))
            block_errors += int(np.sum(ber_per_block.numpy() > 0))
            total_blocks += batch
            total_true_not_in_list += int(np.sum(diag['type_true_not_in_list']))
            total_crc_no_pass += int(np.sum(diag['type_crc_no_pass']))
            total_crc_false_pos += int(np.sum(diag['type_crc_false_pos']))
            all_error_counts.append(diag['info_errors_count'])

            # split PM stats by outcome
            mask_err = diag['has_error'].astype(bool)
            mask_ok  = ~mask_err
            for arr_ok, arr_err, key in [
                (pm_best_correct,   pm_best_errored,   'pm_best'),
                (pm_spread_correct, pm_spread_errored, 'pm_spread'),
                (pm_mean_correct,   pm_mean_errored,   'pm_mean'),
                (pm_std_correct,    pm_std_errored,    'pm_std'),
            ]:
                if mask_ok.any():  arr_ok.append(diag[key][mask_ok])
                if mask_err.any(): arr_err.append(diag[key][mask_err])

            if time() - t > 30:
                ber = np.mean(err)
                fer = np.mean(err > 0)
                print(f'iter: {i / mc_err * 100:5.3f}% | ber: {ber:5.3e} fer {fer:5.3e} | block errors: {block_errors}')
                t = time()
                if block_errors >= num_target_block_errors:
                    break

        all_error_counts = np.concatenate(all_error_counts)
        errored_counts = all_error_counts[all_error_counts > 0]
        ber = np.mean(err)
        fer = np.mean(err > 0)
        k_info = len(A) - (self.crc_enc.crc_length if self.crc is not None else 0)
        safe_div = max(block_errors, 1)
        mean_err_per_block = float(np.mean(errored_counts)) if len(errored_counts) > 0 else 0.0

        # flatten PM arrays
        def _cat(lst): return np.concatenate(lst) if lst else np.array([])
        pm_best_ok   = _cat(pm_best_correct);   pm_best_err   = _cat(pm_best_errored)
        pm_spread_ok = _cat(pm_spread_correct); pm_spread_err = _cat(pm_spread_errored)
        pm_mean_ok   = _cat(pm_mean_correct);   pm_mean_err   = _cat(pm_mean_errored)
        pm_std_ok    = _cat(pm_std_correct);    pm_std_err    = _cat(pm_std_errored)

        def _fmt(arr): return f"{np.mean(arr):.3f} ± {np.std(arr):.3f}" if len(arr) else "n/a"

        print("\n===== BER/BLER Gap Diagnostics =====")
        print(f"  BER:  {ber:.4e}")
        print(f"  BLER: {fer:.4e}")
        print(f"  BLER/BER ratio: {fer / ber if ber > 0 else float('inf'):.1f}")
        print(f"  Mean bit-errors per errored block: {mean_err_per_block:.2f} / {k_info} info bits")
        print(f"  Total blocks: {total_blocks},  block errors: {block_errors}")
        print(f"  Error type breakdown (% of errored blocks):")
        print(f"    true_not_in_list : {100 * total_true_not_in_list / safe_div:.1f}%  (n={total_true_not_in_list})")
        print(f"    crc_no_pass      : {100 * total_crc_no_pass / safe_div:.1f}%  (n={total_crc_no_pass})")
        print(f"    crc_false_pos    : {100 * total_crc_false_pos / safe_div:.1f}%  (n={total_crc_false_pos})")
        if len(errored_counts) > 0:
            pct = np.percentile(errored_counts, [25, 50, 75, 90, 99])
            print(f"  Bit-errors/errored block [p25 p50 p75 p90 p99]: "
                  f"{pct[0]:.0f}  {pct[1]:.0f}  {pct[2]:.0f}  {pct[3]:.0f}  {pct[4]:.0f}")

        print(f"\n  ---- Path Metric Analysis (mean ± std) ----")
        print(f"  {'metric':<14}  {'correct blocks':>20}  {'errored blocks':>20}  {'interpretation'}")
        print(f"  {'pm_best':<14}  {_fmt(pm_best_ok):>20}  {_fmt(pm_best_err):>20}  "
              f"  selected-path PM (lower=more confident)")
        print(f"  {'pm_spread':<14}  {_fmt(pm_spread_ok):>20}  {_fmt(pm_spread_err):>20}  "
              f"  max-min PM (higher=list is differentiated)")
        print(f"  {'pm_mean':<14}  {_fmt(pm_mean_ok):>20}  {_fmt(pm_mean_err):>20}  "
              f"  average PM across all {self.nL} paths")
        print(f"  {'pm_std':<14}  {_fmt(pm_std_ok):>20}  {_fmt(pm_std_err):>20}  "
              f"  std of PM across paths")
        if len(pm_best_ok) and len(pm_best_err):
            ratio = np.mean(pm_best_err) / max(np.mean(pm_best_ok), 1e-9)
            if ratio < 1.5:
                print(f"  => pm_best similar for correct/errored ({ratio:.2f}x): "
                      f"wrong paths look as confident as correct ones — LLRs not discriminative")
            else:
                print(f"  => pm_best higher for errored blocks ({ratio:.2f}x): "
                      f"decoder knew it was unsure but still failed — list too small")
        if len(pm_spread_ok) and len(pm_spread_err):
            ratio_sp = np.mean(pm_spread_err) / max(np.mean(pm_spread_ok), 1e-9)
            if ratio_sp < 0.5:
                print(f"  => pm_spread collapsed for errored blocks ({ratio_sp:.2f}x vs correct): "
                      f"all paths look equally good — PM is uninformative, LLR calibration issue")

        summary = {
            'ber': ber,
            'bler': fer,
            'bler_ber_ratio': fer / ber if ber > 0 else float('inf'),
            'mean_errors_per_errored_block': mean_err_per_block,
            'k_info': k_info,
            'total_blocks': total_blocks,
            'block_errors': block_errors,
            'frac_true_not_in_list': total_true_not_in_list / safe_div,
            'frac_crc_no_pass':      total_crc_no_pass / safe_div,
            'frac_crc_false_pos':    total_crc_false_pos / safe_div,
            'errored_counts': errored_counts,
            'pm_best_correct':   pm_best_ok,
            'pm_best_errored':   pm_best_err,
            'pm_spread_correct': pm_spread_ok,
            'pm_spread_errored': pm_spread_err,
        }
        return err, summary

    def eval_debug(self, Ns, mc_length=100000, code_rate=0.25, batch=100, tol=100,
                   load_nsc_path=None, design_path=None, design_load=False, mc_design=10e7):
        """Like eval() but calls polar_code_err_prob_debug and logs gap diagnostics to wandb."""
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
            decoder_name = load_nsc_path
        else:
            decoder_name = 'sc'

        for n in Ns:
            print(n)
            t = time()
            if design_path is None:
                try:
                    if design_load:
                        design_name = f"{self.construction_name(n, decoder_name)}:latest"
                        sorted_bit_channels = self.load_design(design_name)
                        print(f"Design loaded: {design_name}")
                    else:
                        raise Exception("design_load flag is False")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    Hu, Huy, sorted_bit_channels = self.polar_code_design(n, batch, mc_design, tol=tol, ebno_db=5)
            else:
                sorted_bit_channels = self.load_design(design_path)
            design_time = time() - t
            k = int(code_rate * (2 ** n))

            t = time()
            err, summary = self.polar_code_err_prob_debug(n, mc_length, batch, sorted_bit_channels, k)
            mc_time = time() - t
            print(f"n: {n: 2d} design time: {design_time: 4.1f} "
                  f"code rate: {code_rate: 5.4f} #of mc-blocks: {mc_length} mc time: {mc_time: 4.1f} "
                  f"ber: {summary['ber']: 4.3e} fer: {summary['bler']: 4.3e}")

            log_dict = {
                'n': n,
                'ber': summary['ber'],
                'fer': summary['bler'],
                'code_rate': code_rate,
                'bler_ber_ratio':                 summary['bler_ber_ratio'],
                'mean_errors_per_errored_block':  summary['mean_errors_per_errored_block'],
                'frac_true_not_in_list':          summary['frac_true_not_in_list'],
                'frac_crc_no_pass':               summary['frac_crc_no_pass'],
                'frac_crc_false_pos':             summary['frac_crc_false_pos'],
            }
            if len(summary['errored_counts']) > 0:
                log_dict['error_count_hist'] = wandb.Histogram(summary['errored_counts'])
            # path metric histograms split by correct / errored
            if len(summary['pm_best_correct']) > 0:
                log_dict['pm_best_correct_hist']   = wandb.Histogram(summary['pm_best_correct'])
                log_dict['pm_spread_correct_hist'] = wandb.Histogram(summary['pm_spread_correct'])
                log_dict['pm_best_correct_mean']   = float(np.mean(summary['pm_best_correct']))
                log_dict['pm_spread_correct_mean'] = float(np.mean(summary['pm_spread_correct']))
            if len(summary['pm_best_errored']) > 0:
                log_dict['pm_best_errored_hist']   = wandb.Histogram(summary['pm_best_errored'])
                log_dict['pm_spread_errored_hist'] = wandb.Histogram(summary['pm_spread_errored'])
                log_dict['pm_best_errored_mean']   = float(np.mean(summary['pm_best_errored']))
                log_dict['pm_spread_errored_mean'] = float(np.mean(summary['pm_spread_errored']))
            wandb.log(log_dict)

    def choose_information_and_frozen_sets(self, sorted_bit_channels, k):
        if self.crc is None:
            A = sorted_bit_channels[:k]
            A = sorted(A)
            Ac = sorted_bit_channels[k:]
            Ac = sorted(Ac)
        else:
            k_crc = self.crc_enc.crc_length
            A = sorted_bit_channels[:k+k_crc]
            A = sorted(A)
            Ac = sorted_bit_channels[k+k_crc:]
            Ac = sorted(Ac)
        return A, Ac

class system_model(tf.keras.Model):
    def __init__(self, model, mc_length, code_rate, batch, tol, Ns, design_path, design_load, mc_design, load_nsc_path=None, design5G=False, designGA=False):
        super().__init__()
        self.batch = batch
        self.mc_length = mc_length
        self.code_rate = code_rate
        self.tol = tol
        self.Ns = Ns
        self.design_path = design_path
        self.design_load = design_load
        self.mc_design = mc_design
        self.load_nsc_path = load_nsc_path
        self.model = model
        self.ebno_db = None
        self.design5G = design5G
        self.designGA = designGA
        self.model.channel.CODERATE = code_rate
        if load_nsc_path is not None:
            self.model.load_model(load_nsc_path)
            self.decoder_name = load_nsc_path
        else:
            self.decoder_name = 'sc'


    def call(self, batch_size, ebno_db):
        self.model.channel.snrdb = [ebno_db, ebno_db]
        self.model.channel.save_name = f"5g{ebno_db}"
        #self.ebno_db = ebno_db
        if self.ebno_db is None or self.ebno_db != ebno_db:
            self.batch,self.n,self.info_indices, self.frozen_indices, self.A, self.Ac = self.model.get_parameters(decoder_name=self.decoder_name,
                                                                                                               mc_length=self.mc_length,
                                          code_rate=self.code_rate,
                                          batch=self.batch,
                                          tol=self.tol,
                                          Ns=self.Ns,
                                          design_path=self.design_path,
                                          design_load=self.design_load,
                                          mc_design=self.mc_design,
                                          load_nsc_path=self.load_nsc_path,
                                          ebno_db=ebno_db,
                                          design5G=self.design5G) #
            self.ebno_db = ebno_db

        (err_info, errors, u, u_hat) = self.model.forward_eval(batch_size, 2 ** self.n,
                                self.info_indices, self.frozen_indices, self.A, self.Ac, ebno_db)
        return u, u_hat

class SCTrellisDecoder(SCDecoder):

    def __init__(self, channel, batch=100, *args, **kwargs):
        SCDecoder.__init__(self, channel, batch, *args, **kwargs)

        self.llr_enc_shape = self.llr_dec_shape = (self.channel.cardinality_x,
                                                   self.channel.cardinality_s,
                                                   self.channel.cardinality_s)

        unnormalized_logits = tf.zeros(shape=(self.channel.cardinality_x,
                                              self.channel.cardinality_s,
                                              self.channel.cardinality_s))
        logits = unnormalized_logits - tf.math.reduce_logsumexp(unnormalized_logits, axis=(0, 2), keepdims=True)
        self.input_logits = tf.constant(logits, dtype=dtype)
        self.Ex = self.Ex_enc = EmbeddingX(self.input_logits)

        self.checknode_enc = self.checknode = CheckNodeTrellis(state_size=self.channel.cardinality_s)
        self.bitnode_enc = self.bitnode = BitNodeTrellis(state_size=self.channel.cardinality_s)
        self.emb2llr_enc = self.emb2llr = Embedding2LLRTrellis()

class SCTrellisListDecoder(SCListDecoder):
    def __init__(self, channel, batch=100, list_num=4, crc=None, *args, **kwargs):
        SCListDecoder.__init__(self, channel, batch, list_num=list_num, crc=crc, *args, **kwargs)

        self.llr_enc_shape = self.llr_dec_shape = (self.channel.cardinality_x,
                                                   self.channel.cardinality_s,
                                                   self.channel.cardinality_s)

        unnormalized_logits = tf.zeros(shape=(self.channel.cardinality_x,
                                              self.channel.cardinality_s,
                                              self.channel.cardinality_s))
        logits = unnormalized_logits - tf.math.reduce_logsumexp(unnormalized_logits, axis=(0, 2), keepdims=True)
        self.input_logits = tf.constant(logits, dtype=dtype)
        self.Ex = self.Ex_enc = EmbeddingX(self.input_logits)

        self.checknode_enc = self.checknode = CheckNodeTrellis(state_size=self.channel.cardinality_s)
        self.bitnode_enc = self.bitnode = BitNodeTrellis(state_size=self.channel.cardinality_s)
        self.emb2llr_enc = self.emb2llr = Embedding2LLRTrellis()
        self.checknode_list = CheckNodeTrellis(state_size=self.channel.cardinality_s, batch_dims=3)
        self.bitnode_list = BitNodeTrellis(state_size=self.channel.cardinality_s, batch_dims=3)
        self.emb2llr_list = Embedding2LLRTrellis(batch_dims=3)

class SCNeuralDecoder(SCDecoder):

    def __init__(self, channel, embedding_size, hidden_size, layers_per_op,
                 activation='elu', batch=100, eyN0=False, trained_block_norm=10,
                 layers_per_op_emb2llr=None, emb2llr_snr=False, llr_clip=None, *args, **kwargs):
        SCDecoder.__init__(self, channel, batch, eyN0, *args, **kwargs)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers_per_op = layers_per_op
        self.activation = activation
        # emb2llr can use its own depth (defaults to layers_per_op if not specified)
        self.layers_per_op_emb2llr = layers_per_op_emb2llr if layers_per_op_emb2llr is not None else layers_per_op
        self.emb2llr_snr = emb2llr_snr
        self.llr_clip = llr_clip  # None = no clipping; float = clip emb2llr output to ±llr_clip
        self.llr_dec_shape = (embedding_size,)
        self.Ey = Sequential([Dense(embedding_size * self.channel.BPS, use_bias=True, activation=None)])
        self.Ey = Sequential([Dense(50, activation=activation, use_bias=True, ) for i in range(2)] + \
                             [Dense(embedding_size * self.channel.BPS, activation=None, use_bias=True)])
        # self.Ey = EyModel(embedding_size, self.channel.BPS, activation=activation)

        #self.Ex = EmbeddingX(tf.zeros(shape=(embedding_size,)))
        # self.Ex_enc = self.Ex = EmbeddingX(tf.zeros(shape=(embedding_size * self.channel.BPS,)))
        # self.Ey = Sequential([Dense(300, activation=activation, use_bias=True, ) for i in range(3)] + \
        #                      [Dense(embedding_size * self.channel.BPS, activation=None, use_bias=True)])
        self.Ex_enc = self.Ex = EmbeddingX(tf.zeros(shape=(embedding_size * self.channel.BPS,)))
        #self.Ex = EmbeddingX(tf.zeros(shape=(embedding_size,)))

        # self.checknode = self.checknode_enc = CheckNodeNNEmb(hidden_size, embedding_size, layers_per_op,
        #                                                      use_bias=True, activation=activation)
        # self.bitnode = self.bitnode_enc = BitNodeNNEmb(hidden_size, embedding_size, layers_per_op,
        #                                                use_bias=True, activation=activation)
        # self.emb2llr = self.emb2llr_enc = Embedding2LLR(hidden_size, layers_per_op,
        #                                                 use_bias=True, activation=activation)

        self.checknode_enc = self.checknode  = CheckNodeNNEmb(hidden_size, embedding_size, layers_per_op,
                                                             use_bias=True, activation=activation)
        self.bitnode_enc = self.bitnode = BitNodeNNEmb(hidden_size, embedding_size, layers_per_op,
                                                       use_bias=True, activation=activation)
        if emb2llr_snr:
            self.emb2llr_enc = self.emb2llr = Embedding2LLRwithSNR(
                self.channel.no, hidden_size, self.layers_per_op_emb2llr,
                use_bias=True, activation=activation, llr_max=llr_clip)
        else:
            self.emb2llr_enc = self.emb2llr = Embedding2LLR(
                hidden_size, self.layers_per_op_emb2llr, use_bias=True, activation=activation,
                llr_max=llr_clip)

        # only ey
        # self.Ex_enc = EmbeddingX(self.input_logits)
        # self.checknode_enc = CheckNodeVanilla()
        # self.bitnode_enc = BitNodeVanilla()
        # self.emb2llr_enc = Activation(tf.identity)

    def load_model(self, load_path):
        if os.path.isdir(load_path):
            artifact_dir = load_path
        else:
            artifact = wandb.use_artifact(load_path, type='model_weights')
            artifact_dir = artifact.download()

        shape = (self.batch, 4)
        u = tf.zeros(shape=(self.batch, 4) + (1,))
        y = tf.zeros(shape=(self.batch, 4) + (self.channel.cardinality_y,))
        if self.channel.BPS != 1:
            e_dec = self.Ex.call((self.batch, 2))
            e_dec = tf.reshape(e_dec, (self.batch, self.channel.BPS*2, -1))[:,:4]
        else:
            e_dec = self.Ex.call((self.batch, 4))
        if self.eyN0:
            y = tf.concat([y, u], axis=2)
        self.Ey(y)


        self.checknode.call((e_dec, e_dec))
        self.bitnode.call((e_dec, e_dec, u))
        #self.emb2llr.call(e_dec)


        self.emb2llr.call(e_dec)
        #

        self.checknode.built = True
        self.bitnode.built = True
        self.emb2llr.built = True
        self.Ex.built = True
        self.Ey.built = True


        if isinstance(self.layer_norms_ey, BatchNormModel):
            for i in range(self.trained_block):
                #self.layer_norms_ex(e_dec, i)
                self.layer_norms_ey(e_dec, i)
            #self.layer_norms_ex.built = True
            self.layer_norms_ey.built = True
            #self.layer_norms_ex.load_weights(f"{artifact_dir}/layer_norms_ex.h5")
            self.layer_norms_ey.load_weights(f"{artifact_dir}/layer_norms_ey.h5")
        else:
            print("layer_norms are not loaded")
        # try:
        #     self.layer_norms_rep(e_dec, 0)
        #     self.layer_norms_rep.built = True
        #     self.layer_norms_rep.load_weights(f"{artifact_dir}/layer_norms_rep.h5")
        # except:
        #     print('layer_norms_rep is not loaded')
        try:
            self.checknode.load_weights(f"{artifact_dir}/checknode-weights.h5")
            self.bitnode.load_weights(f"{artifact_dir}/bitnode-weights.h5")
            self.emb2llr.load_weights(f"{artifact_dir}/emb2llr-weights.h5", skip_mismatch=True)
            self.Ex.load_weights(f"{artifact_dir}/ex-weights.h5")
            self.Ey.load_weights(f"{artifact_dir}/ey-weights.h5")
        except:
            self.checknode.load_weights(f"{artifact_dir}/checknode.weights.h5")
            self.bitnode.load_weights(f"{artifact_dir}/bitnode.weights.h5")
            self.emb2llr.load_weights(f"{artifact_dir}/emb2llr.weights.h5", skip_mismatch=True)
            self.Ex.load_weights(f"{artifact_dir}/ex.weights.h5")
            self.Ey.load_weights(f"{artifact_dir}/ey.weights.h5")

        self.Ex_enc.call(shape)
        self.Ex_enc.built = True
        #self.Ex_enc.load_weights(f"{artifact_dir}/ex-weights.h5", by_name=True, skip_mismatch=True)
        print(f"model is loaded from {load_path}")

class NeuralPolarDecoder(SCNeuralDecoder):
    def __init__(self, channel, embedding_size, hidden_size, layers_per_op,
                 activation='elu', batch=100, lr=0.001, optimizer='sgd',
                 input_distribution='sc', input_state_size=2, pred_decay=None, lr_decay=None,
                 trainEyOnly=False, eyN0=False, EyPerSnr=False,
                 layers_per_op_emb2llr=None, emb2llr_snr=False, llr_clip=30.0):
        SCNeuralDecoder.__init__(self, channel, embedding_size, hidden_size, layers_per_op,
                                 activation, eyN0=eyN0, batch=batch,
                                 layers_per_op_emb2llr=layers_per_op_emb2llr,
                                 emb2llr_snr=emb2llr_snr, llr_clip=llr_clip)

        self.channel = channel[0]
        # self.embedding_size = embedding_size
        # self.hidden_size = hidden_size
        # self.layers_per_op = layers_per_op
        # self.activation = activation
        self.lr = lr
        self.trainEyOnly = trainEyOnly
        self.input_distribution = input_distribution
        if input_distribution == 'sc':
            self.llr_enc_shape = ()
            logits = tf.zeros(shape=self.llr_enc_shape)
            self.input_logits = tf.constant(logits, dtype=dtype)
            self.Ex_enc = EmbeddingX(self.input_logits)
            self.checknode_enc = CheckNodeVanilla()
            self.layer_norms_ex_enc = lambda x, i: x
            self.layer_norms_ey_enc = lambda x, i: x
            self.bitnode_enc = BitNodeVanilla()
            self.emb2llr_enc = Activation(tf.identity)
            self.input_distribution_name = f"{self.input_distribution}"
            print("Input distribution is implemented via a SC encoder")
        elif input_distribution == 'sct':
            self.llr_enc_shape = (2, input_state_size, input_state_size)
            unnormalized_logits = tf.zeros(shape=self.llr_enc_shape)
            logits = unnormalized_logits - tf.math.reduce_logsumexp(unnormalized_logits, axis=(0, 2), keepdims=True)
            self.input_logits = tf.constant(logits, dtype=dtype)
            self.Ex_enc = EmbeddingX(self.input_logits)
            self.checknode_enc = CheckNodeTrellis(state_size=self.channel.cardinality_s)
            self.bitnode_enc = BitNodeTrellis(state_size=self.channel.cardinality_s)
            self.emb2llr_enc = Embedding2LLRTrellis()
            self.input_distribution_name = f"{self.input_distribution}-state-size-{input_state_size}"
            print("Input distribution is implemented via a SCT encoder")
        elif input_distribution == 'rnn':
            self.input_model = BinaryRNN(units=embedding_size)
            self.input_distribution_name = f"{self.input_distribution}-hidden-{embedding_size}"
            print("Input distribution is implemented via an RNN encoder")
        else:
            raise ValueError(f'input_distribution received invalid value: {input_distribution}')

        self.llr_dec_shape = (embedding_size,)
        # self.Ey = Sequential([Dense(50, activation=activation, use_bias=True) for i in range(2)] + \
        #           [Dense(embedding_size * self.channel.BPS, use_bias=True, activation=None)])
        #self.Ey = Sequential([Dense(embedding_size * self.channel.BPS, use_bias=True, activation=None)])
        #self.Ey = EyModel(embedding_size, self.channel.BPS, activation=activation)

        #self.Ey2 = Sequential([Dense(embedding_size * self.channel.BPS, use_bias=True, activation=None)])
        #
        # self.Ey = Sequential([Dense(50, activation=activation, use_bias=True,) for i in range(2)] + \
        # [Dense(embedding_size * self.channel.BPS, activation=None, use_bias=True)])
        # self.Ey = Sequential([Dense(300, activation=activation, use_bias=True,) for i in range(3)] + \
        # [Dense(embedding_size * self.channel.BPS, activation=None, use_bias=True)])
        # self.embedding_size = embedding_size
        # self.activation = activation
        # self.Ex = EmbeddingX(tf.zeros(shape=(embedding_size * self.channel.BPS,)))
        #self.Ex = EmbeddingX(tf.zeros(shape=(embedding_size)))

        # self.checknode = CheckNodeNNEmb(hidden_size, embedding_size, layers_per_op,
        #                                 use_bias=True, activation=activation)
        # self.bitnode = BitNodeNNEmb(hidden_size, embedding_size, layers_per_op,
        #                             use_bias=True, activation=activation)
        # self.emb2llr = Embedding2LLR(hidden_size, layers_per_op,
        #                              use_bias=True, activation=activation)
        self.layer_norms_ex = lambda x, i: x
        self.layer_norms_ey = lambda x, i: x

        self.split_even_odd = SplitEvenOdd(axis=1)
        self.f2 = F2()
        self.interleave = Interleave(axis=1)
        self.pred_decay = 0.0 if pred_decay is None else pred_decay

        self.optimizer_name = optimizer
        if lr_decay is not None:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                lr,
                decay_steps=1000,
                decay_rate=lr_decay,
                staircase=True)
        if optimizer == "sgd":
            self.estimation_optimizer = SGD(learning_rate=lr, clipnorm=1.0)
            self.improvement_optimizer = SGD(learning_rate=lr, clipnorm=1.0)
        elif optimizer == "adam":
            self.estimation_optimizer = Adam(learning_rate=lr, clipnorm=1.0)
            self.improvement_optimizer = Adam(learning_rate=lr, clipnorm=1.0)
        else:
            raise ValueError("invalid optimizer name")

        self.metric_ce_x = tf.keras.metrics.Mean(name="ce_x")
        self.metric_ce_y = tf.keras.metrics.Mean(name="ce_y")

    def sample_inputs(self, batch, N):
        if self.input_distribution == 'sc' or self.input_distribution == 'sct':
            batch_N_shape = [tf.constant(batch), tf.constant(N)]
            ex_enc = self.Ex_enc.call(batch_N_shape)
            # generate shared randomness
            r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)
            # create frozen bits for encoding. encoded bits need to be 0.5.
            f_enc = 0.5 * tf.ones(shape=(batch, N, 1))
            #_, x, llr_x = self.encode(ex_enc, f_enc, N, r, sample=True)
            self.u_temp, x, llr_x = self.encode(ex_enc, f_enc, N, r, sample=True)
        elif self.input_distribution == 'rnn':
            x, llr_x = self.input_model.generate_binary_sequence(batch, N)
        else:
            x, llr_x = None, None
            ValueError(f'input_distribution received invalid value: {self.input_distribution}')
        return x, llr_x

    def BitNodeFastCE(self, e, x):
        e_odd, e_even = self.split_even_odd.call(e)

    def nsc_train_loss(self, y, f, loss):
        N = y.shape[1]
        if N == 1:
            pred = self.emb2llr.__call__(y)
            loss += tf.reduce_sum(self.loss_fn(f, pred), axis=1)
            # self.update_states(tf.reshape(f, [-1, 1]), tf.reshape(pred, [-1, 1]))
            return f, loss, pred

        y_odd, y_even = self.split_even_odd.call(y)
        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves

        u1est = self.checknode.call((y_odd, y_even))

        # R_N^T maps u1est to top polar code
        u1hardprev, loss, pred1_prev = self.nsc_train_loss(u1est, f_left, loss)

        # Using u1est and x1hard, we can estimate u2
        u2est = self.bitnode.call((y_odd, y_even, u1hardprev))

        # R_N^T maps u2est to bottom polar code
        u2hardprev, loss, pred2_prev = self.nsc_train_loss(u2est, f_right, loss)

        # Interleave indices and perform the required modulo operation
        x = self.f2.__call__((u1hardprev, u2hardprev))
        x = self.interleave.__call__(x)
        pred_prev = tf.concat([pred1_prev, pred2_prev], axis=1)
        pred = self.emb2llr.__call__(y)

        loss += tf.reduce_sum(self.loss_fn(x, pred), axis=1)
        # self.update_states(tf.reshape(x, [-1, 1]), tf.reshape(pred, [-1, 1]))
        return x, loss, pred_prev

    def ce_N4(self, ey, x):
        loss_array = []
        pred = self.emb2llr.call(ey)
        loss = self.loss_fn(x, pred)
        loss_array.append(loss)
        xo, xe = self.split_even_odd.call(x)
        eo, ee = self.split_even_odd.call(ey)
        v1_xor = tf.math.floormod(xo + xe, 2)
        ec1 = self.checknode.call((eo, ee))
        eb1 = self.bitnode.call((eo, ee, v1_xor))
        e1 = tf.concat([ec1, eb1], axis=1)
        v1 = tf.concat([v1_xor, xe], axis=1)
        pred = self.emb2llr.call(e1)
        loss = self.loss_fn(v1, pred)
        loss_array.append(loss)

        v1o, v1e = self.split_even_odd.call(v1)
        e1o, e1e = self.split_even_odd.call(e1)
        v2_xor = tf.math.floormod(v1o + v1e, 2)

        ec2 = self.checknode.call((e1o, e1e))
        eb2 = self.bitnode.call((e1o, e1e, v2_xor))

        u = tf.concat([v2_xor[:, 0][:, None], v1e[:, 0][:, None], v2_xor[:, 1][:, None], v1e[:, 1][:, None]], axis=1)
        e2 = tf.concat([ec2[:, 0][:, None], eb2[:, 0][:, None], ec2[:, 1][:, None], eb2[:, 1][:, None]], axis=1)
        pred = self.emb2llr.call(e2)
        loss = self.loss_fn(u, pred)
        loss_array.append(loss)
        loss_y_array = loss_array
        return u, loss_y_array

    def design_step(self, batch, N, train_ex=False, full_bu_depth=None, snrdb=None):
        x, _ = self.sample_inputs(batch, N)

        y = self.channel.sample_channel_outputs(x, snrdb)


        if self.channel.snrdb[0] == self.channel.snrdb[1] or not (snrdb is None):
            no = tf.ones((batch, y.shape[1], 1)) * self.channel.no
        else:
            no = tf.repeat(self.channel.no, axis=1, repeats=y.shape[1])[...,None]
        if self.eyN0:
            y = tf.concat([y, no], axis=2)
        #ey = self.Ey(tf.concat([y, no], axis=2))
        ey = self.Ey(y)

        ey = tf.reshape(ey,(batch, x.shape[1], -1))
        v_array, loss_y_array, pred_y_array, norm_y_array = self.fast_ce(ey, x, full_bu_depth)
        loss_y_array = [l / tf.math.log(2.0) for l in loss_y_array]
        return  v_array, pred_y_array, loss_y_array

    #@tf.function
    def estimation_step(self, batch, N, train_ex=False, full_bu_depth=None, train_mode='all'):
        x, _ = self.sample_inputs(batch, N)
        x_ch, rep_ind = self.NeuralRateMatching(x, N, self.channel_design.E*self.channel_design.BPS)
        y = self.channel.sample_channel_outputs(x_ch)

        # keep emb2llr's SNR reference in sync with the current batch's noise level
        if hasattr(self.emb2llr, 'no_source'):
            self.emb2llr.no_source = self.channel.no

        with tf.GradientTape(persistent=True) as tape:
            if self.eyN0:
                if self.channel.snrdb[0] == self.channel.snrdb[1]:
                    no = tf.ones((batch, y.shape[1], 1)) * self.channel.no
                else:
                    no = tf.repeat(self.channel.no, axis=1, repeats=y.shape[1])[...,None]
                y = tf.concat([y, no], axis=2)


            ey = self.Ey(tf.concat([y], axis=2))
            if N != self.channel_design.E*self.channel_design.BPS:
                ey, _ = self.RecoverNeuralRateMatching(ey, N, batch)
            else:
                ey = tf.reshape(ey,(batch, x.shape[1], -1))
            # if train_ex:
            #     ex = self.Ex((y.shape[0], y.shape[1]))
            #     ex = tf.reshape(ex, (batch, x.shape[1], -1))
                # punc random train
                # random_mask = tf.random.uniform((batch, N), minval=0, maxval=1, dtype=tf.float32) < 0.1
                # expanded_mask = tf.expand_dims(random_mask, axis=-1)
                # ey = tf.where(expanded_mask, ex, ey)

            v_array, loss_y_array, pred_y_array, norm_y_array = self.fast_ce(ey, x, full_bu_depth)
            u = v_array[-1]
            loss = tf.reduce_mean(loss_y_array)
            if train_ex:
                ex_batch = tf.constant([batch, N])
                ex = self.Ex(ex_batch)
                ex = tf.reshape(ex, (batch, x.shape[1], -1))
                _, loss_x_array, _, _ = self.fast_ce(ex, x, full_bu_depth)
                loss += tf.reduce_mean(loss_x_array)
            else:
                loss_x_array = tf.zeros_like(loss_y_array)

            if train_mode == 'all':
                trainable_vars = (self.checknode.trainable_weights +
                                  self.bitnode.trainable_weights +
                                  self.emb2llr.trainable_weights +
                                  self.Ey.trainable_weights +
                                  self.Ex.trainable_weights)
            elif train_mode == 'emb2llr':
                trainable_vars = self.emb2llr.trainable_weights
            elif train_mode == 'ey_emb2llr':
                trainable_vars = self.Ey.trainable_weights + self.emb2llr.trainable_weights
            elif train_mode == 'ey':
                trainable_vars = self.Ey.trainable_weights
            else:
                raise ValueError(f"Unknown train_mode: {train_mode}")

        gradients = tape.gradient(loss, trainable_vars)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
        self.estimation_optimizer.apply_gradients(zip(gradients, trainable_vars))
        errors = np.sum(np.array(u[..., 0]) == np.array(pred_y_array[-1] <= 0, dtype=int), axis=0) / u.shape[0]
        l2_penalty_y = 0

        # polarization stats: per-position CE averaged over the batch, in bits
        # loss_y_array[0] is (batch, N) — top-level CE for all N synthetic channels
        ce_per_pos = tf.reduce_mean(loss_y_array[-1], axis=0)          # (N,) in bits already (divided below)
        ce_per_pos = ce_per_pos / tf.math.log(2.0)                    # nats → bits in [0, 1]
        n_ch = float(tf.size(ce_per_pos))
        polar_good = float(tf.reduce_sum(tf.cast(ce_per_pos < 0.01, tf.float32))) / n_ch   # near-capacity
        polar_bad  = float(tf.reduce_sum(tf.cast(ce_per_pos > 0.9, tf.float32))) / n_ch   # near-zero capacity
        polar_mid  = 1.0 - polar_good - polar_bad                                         # partially polarized

        res = {
            #self.metric_ce_x.name: self.metric_ce_x.result(),
               #self.metric_ce_y.name: self.metric_ce_y.result(),
               'grad_norm': grad_norm,
               'ce_y': tf.reduce_mean(loss_y_array[-1]) / tf.math.log(2.0),
               'ce_x': tf.reduce_mean(loss_x_array[-1]) / tf.math.log(2.0),
               'pred_penalty': self.pred_decay * (l2_penalty_y ),
               'loss_y_array': [l / tf.math.log(2.0) for l in loss_y_array],
               'norm_y_array': norm_y_array,
               'pred_y_array': pred_y_array,
               'lr': self.estimation_optimizer.learning_rate,
               'loss':  (loss) / tf.math.log(2.0),
               'errors': errors,
               'polar_good': polar_good,
               'polar_bad':  polar_bad,
               'polar_mid':  polar_mid,
        }
        return res

    #@tf.function
    def improvement_step(self, batch, N):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]

        with tf.GradientTape() as tape:
            x, llr_x1 = self.sample_inputs(batch, N)
            llr_x = tf.where(tf.equal(x, 1.0), llr_x1, -llr_x1)
            log_px = tf.math.log(tf.math.sigmoid(llr_x)+1e-10)
            y = self.channel.sample_channel_outputs(x)

            if self.channel.snrdb[0] == self.channel.snrdb[1]:
                no = tf.ones((batch, y.shape[1], 1)) * self.channel.no
            else:
                no = tf.repeat(self.channel.no, axis=1, repeats=y.shape[1])[..., None]
            ey = self.Ey(tf.concat([y, no], axis=2))
            ey = tf.reshape(ey, (batch, x.shape[1], -1))

            ex = self.Ex(y.shape[:2])
            ex = tf.reshape(ex, (batch, x.shape[1], -1))

            _, loss_x_array, _, _ = self.fast_ce(ex, x)
            _, loss_y_array, _, _ = self.fast_ce(ex+ey, x)

            loss_x = tf.reduce_mean(loss_x_array[-1], axis=1)
            loss_y = tf.reduce_mean(loss_y_array[-1], axis=1)
            reward = loss_x - loss_y
            Q = (reward - tf.reduce_mean(reward)) / (tf.math.sqrt(tf.reduce_mean(tf.square(reward))) + 1e-5)
            loss = -tf.reduce_mean(tf.stop_gradient(Q) *
                                   tf.reduce_mean(log_px, axis=(1, 2)))
            #loss = tf.reduce_mean(loss_y) + tf.reduce_mean(loss_x)
        # Compute gradients
        if self.input_distribution == 'sc' or self.input_distribution == 'sct':
            trainable_vars = self.Ex_enc.trainable_weights
        elif self.input_distribution == 'rnn':
            trainable_vars = self.input_model.trainable_weights
        else:
            raise ValueError(f'input_distribution received invalid value: {self.input_distribution}')

        gradients = tape.gradient(loss, trainable_vars)
        self.improvement_optimizer.apply_gradients(zip(gradients, trainable_vars))

        ce_x = tf.reduce_mean(loss_x_array[-1]) / tf.math.log(2.0)
        ce_y = tf.reduce_mean(loss_y_array[-1]) / tf.math.log(2.0)
        self.metric_ce_x.update_state(ce_x)
        self.metric_ce_y.update_state(ce_y)

        # Return a dict mapping metric names to current value
        res = {self.metric_ce_x.name: self.metric_ce_x.result(),
               self.metric_ce_y.name: self.metric_ce_y.result(),
               'loss': loss,
               'px': tf.reduce_mean(llr_x1)}


        return res

    def design_save(self, ce, errors,  k, n, save_name, save=False):
        sorted_bit_channels =dict()
        errors_mean = dict()
        ce_mean = dict()
        for i in range(6, n+1):
            errors_mean[int(2**i)] = errors[int(2**i)]/ k
            ce_mean[int(2**i)] = ce[int(2**i)] / k
            sorted_bit_channels[int(2**i)] = np.argsort(ce_mean[int(2**i)])
            #res = [error_mean.copy(), ce_mean.copy(), sorted_bit_channels.copy()]
            res = sorted_bit_channels[int(2**i)]
            if save:
                save_tmp_path = f"./artifacts/tmp/{wandb.run.name}/design.npy"
                # Extract the directory part of the path
                save_name_temp = save_name + f"_N{int(2**i)}"
                directory = os.path.dirname(save_tmp_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                np.save(save_tmp_path, res)
                weights_artifact = wandb.Artifact(save_name_temp, type='design')

                # Add the model weights file to the artifact
                weights_artifact.add_file(save_tmp_path)

                # Log the artifact to W&B
                artifact_log = wandb.log_artifact(weights_artifact)
                artifact_log.wait()
                # print(f"model is saved as {save_name}")
                artifact = wandb.run.use_artifact(f"{save_name_temp}:latest")
                print(f"Design: {artifact.name}")
        return sorted_bit_channels, ce_mean, errors_mean

    def design(self, train_block_length=8, train_batch=10, num_iters=10000, load_nsc_path=None, full_bu_depth=None):
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
        n = int(np.log2(train_block_length))
        errors = dict()
        ce = dict()
        res = dict()
        for i in np.arange(6, n+1, dtype=int):
            errors[int(2**i)]= tf.zeros(shape=(2**i), dtype=dtype)
            ce[int(2**i)] = tf.zeros(shape=(2**i), dtype=dtype)
        t = time()
        SNRs = np.arange(-5, 5, 0.5)
        for snr in SNRs:
            count_perfect_diff = 0
            for k in tqdm(range(1,num_iters)):
                v_array, pred_y_array, loss_y_array  = self.design_step(train_batch, train_block_length, full_bu_depth=full_bu_depth, snrdb=snr)
                for i in range(6, n+1):
                    pred = pred_y_array[i]
                    u = v_array[i]
                    errors_i = np.sum(np.array(u[..., 0]) == np.array(pred <= 0, dtype=int), axis=0) / u.shape[0]
                    ce_i = tf.reduce_mean(loss_y_array[i], axis=0)
                    errors[int(2**i)] += np.mean(np.reshape(errors_i, (int(train_block_length / (2 ** i)), -1)), axis=0)
                    ce[int(2**i)] += np.mean(np.reshape(ce_i, (int(train_block_length / (2 ** i)), -1)), axis=0)
                    # errors[int(2**i)] += np.mean(np.reshape(errors_i, (int(train_block_length / (2 ** i)), -1))[0], axis=0)
                    # ce[int(2**i)] += np.mean(np.reshape(ce_i, (int(train_block_length / (2 ** i)), -1))[0], axis=0)
                if k % 1000 == 0 and k>0:
                    save_name = f'design_un_{wandb.config["group"]}'.replace(
                        '+', '')
                    # if k % 5000 == 0 and k>0:
                    #     save = True
                    # else:
                    #     save = False
                    save = False
                    sorted_bit_channels_k, ce_mean, errors_mean = self.design_save(ce, errors, k, n, save_name, save)
                    if k==1000:
                        sorted_bit_channels = sorted_bit_channels_k
                    else:
                        num_different = np.sum(np.not_equal(sorted_bit_channels_k[train_block_length][:train_block_length//2], sorted_bit_channels[train_block_length][:train_block_length//2]))
                        print(f'num different {snr}: {num_different}')
                        # num_different = np.sum(np.not_equal(sorted_bit_channels_k[512], sorted_bit_channels[512]))
                        # print(f'num different 512: {num_different}')
                        sorted_bit_channels = sorted_bit_channels_k
                        wandb.log({f'num_different {snr}': num_different})
                        if  num_different <= 10:
                            count_perfect_diff +=1
                        if count_perfect_diff == 3:
                            res[snr] = ce_mean[train_block_length]
                            print(f'design converged {snr}')
                            break


        ce_all_snr = tf.zeros(shape=(train_block_length), dtype=dtype)
        for snr in SNRs:
            ce_all_snr += res[snr]
        ce_all_snr = ce_all_snr/len(SNRs)
        sorted_bit_channels_final = np.argsort(ce_all_snr)

        #save res
        save_tmp_path = f"./artifacts/tmp/{wandb.run.name}/res.npy"
        # Extract the directory part of the path
        save_name_temp = 'all_snr_res_' + f"{wandb.run.name}"
        directory = os.path.dirname(save_tmp_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(save_tmp_path, res)
        weights_artifact = wandb.Artifact(save_name_temp, type='design')

        # Add the model weights file to the artifact
        weights_artifact.add_file(save_tmp_path)

        # Log the artifact to W&B
        artifact_log = wandb.log_artifact(weights_artifact)
        artifact_log.wait()
        # print(f"model is saved as {save_name}")
        artifact = wandb.run.use_artifact(f"{save_name_temp}:latest")
        print(f"Design: {artifact.name}")


        # save the final design
        save_tmp_path = f"./artifacts/tmp/{wandb.run.name}/design.npy"
        # Extract the directory part of the path
        save_name_temp = save_name
        directory = os.path.dirname(save_tmp_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(save_tmp_path, sorted_bit_channels_final)
        weights_artifact = wandb.Artifact(save_name_temp, type='design')

        # Add the model weights file to the artifact
        weights_artifact.add_file(save_tmp_path)

        # Log the artifact to W&B
        artifact_log = wandb.log_artifact(weights_artifact)
        artifact_log.wait()
        # print(f"model is saved as {save_name}")
        artifact = wandb.run.use_artifact(f"{save_name_temp}:latest")
        print(f"Design: {artifact.name}")




    def train(self, train_block_length=8, train_batch=10, num_iters=10000,
              logging_freq=1000, saving_freq=3600, train_ex=False, load_nsc_path=None, save_model=False,
              logging_llr_hist=False, full_bu_depth=None, resume_train=False,
              train_mode='all'):
        save_name = f'train_' \
                    f'group-{wandb.run.group}_' \
                    f'{self.input_distribution_name}_' \
                    f'{self.channel.save_name}_' \
                    f'nt-{train_block_length}_' \
                    f'npd-{self.embedding_size}-{self.layers_per_op}x{self.hidden_size}'

        if resume_train:
            try:
                self.load_model(f'{save_name}:latest')
                print(f"Resumed from latest checkpoint: {save_name}:latest")
            except Exception as e:
                print(f"resume_train: could not load latest checkpoint ({e}), starting from scratch")

        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
            if self.trainEyOnly:
                self.Ey = Sequential([Dense(50, activation=self.activation, use_bias=True, ) for i in range(2)] + \
                                     [Dense(self.embedding_size * self.channel.BPS, activation=None, use_bias=True)])

        _valid_modes = ('all', 'emb2llr', 'ey_emb2llr', 'ey')
        if train_mode not in _valid_modes:
            raise ValueError(f"train_mode must be one of {_valid_modes}, got '{train_mode}'")
        print(f"train_mode: {train_mode}")

        self.reset_metrics()

        t_save = time()
        t = time()
        loss_y_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
        loss_x_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
        norm_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
        pred_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
        counters_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
        loss = []
        mi = []
        ce_x = []
        ce_y = []
        pred_penalty = []
        polar_good_buf, polar_bad_buf, polar_mid_buf = [], [], []
        table = wandb.Table(columns=["time_step", "depth", "loss_x", "loss_y", "norm_y", "pred_y"])
        errors = 0
        design = tf.zeros(shape=(train_block_length), dtype=dtype)
        design_mc = 1e5
        for k in tqdm(range(1,num_iters)):
            train_ex = (k % 10 == 0)
            train_ex = False
            r = self.estimation_step(train_batch, train_block_length, train_ex=train_ex,
                                     full_bu_depth=full_bu_depth, train_mode=train_mode)
            mi.append(1 - r["ce_y"].numpy())
            ce_y.append(r["ce_y"].numpy())
            ce_x.append(r["ce_x"].numpy())
            loss.append(r["loss"].numpy())
            polar_good_buf.append(r['polar_good'])
            polar_bad_buf.append(r['polar_bad'])
            polar_mid_buf.append(r['polar_mid'])
            design += tf.reduce_sum(r['loss_y_array'][-1], axis=0)
            errors+=r['errors']
            #pred_penalty.append(r["pred_penalty"].numpy())
            for i, (ly_, n_, p_) in enumerate(zip(r['loss_y_array'], r['norm_y_array'], r['pred_y_array'])):
                loss_y_dict[i] += np.sum(ly_)
                norm_dict[i] += np.sum(n_)
                pred_dict[i] += np.sum(np.abs(p_))
                counters_dict[i] += np.sum(np.prod(ly_.shape).astype(float))
                table.add_data(k, i,
                               loss_x_dict[i] / counters_dict[i],
                               loss_y_dict[i] / counters_dict[i],
                               norm_dict[i] / counters_dict[i],
                               pred_dict[i] / counters_dict[i])
            if k*train_batch % design_mc == 0:
                design /= design_mc
                errors /= k
                mi = 1 - design
                # k0275 = int(np.floor(0.275 * train_block_length))
                # k05 = int(np.floor(0.5 * train_block_length))
                # print(f'k={k0275}, bler: {np.sum(np.sort(errors)[:k0275]):4.3e} , ber:{np.mean(np.sort(errors)[:k0275]): 4.3e}')
                # print(f'k={k05}, bler: {np.sum(np.sort(errors)[:k05]):4.3e} , ber:{np.mean(np.sort(errors)[:k05]): 4.3e}')
                sorted_bit_channels = np.argsort(-mi)
                print('')
                try:
                    print('rate 1e-3', np.where(np.cumsum(errors[sorted_bit_channels])<=1e-3)[0][-1])
                except:
                    pass
                errors = 0
                design = tf.zeros(shape=(train_block_length), dtype=dtype)

                # if k % 20e3:
                #     self.save_design(train_block_length, sorted_bit_channels, 'train_design_temp')

            if k % logging_freq == 0:
                print('')
                #print('errors: ', errors/logging_freq)

                # print(f'k={k0275}, bler: {np.sum(np.sort(errors/logging_freq)[:k0275]):4.3e} , ber:{np.mean(np.sort(errors/logging_freq)[:k0275]): 4.3e}')
                # print(f'k={k05}, bler: {np.sum(np.sort(errors/logging_freq)[:k05]):4.3e} , ber:{np.mean(np.sort(errors/logging_freq)[:k05]): 4.3e}')
                #errors = 0

                if logging_llr_hist:
                    norm_y_array = r['norm_y_array'] / np.sqrt(self.embedding_size)

                    # Compute the 99th percentile for each dimension
                    percentiles_99 = tf.numpy_function(
                        lambda x: np.percentile(x, 99, axis=(0, 2)),
                        [norm_y_array],
                        tf.float32
                    )

                    # Compute histograms
                    histograms = []
                    for i in tf.range(norm_y_array.shape[1]):
                        data = norm_y_array[:, i, :]
                        # Use the 99th percentile as the upper bound for each dimension's histogram
                        hist = tf.histogram_fixed_width(data, value_range=[0.0, percentiles_99[i]], nbins=50)
                        histograms.append(hist)

                    # Log histograms using wandb
                    for i, hist in enumerate(histograms):
                        hist_np = hist.numpy()
                        # Create bin edges based on the 99th percentile
                        bin_edges = np.linspace(0, percentiles_99[i], 51)
                        wandb.log({f"Histogram_{i}": wandb.Histogram(np_histogram=(hist_np, bin_edges))})
                log_N = int(np.log2(train_block_length))
                result_dict = {"iter_scl": k,
                                "ce_y": np.mean(np.array(ce_y)),
                               "ce_x": np.mean(np.array(ce_x)),
                               "mi": np.mean(np.array(mi)),
                                "loss": np.mean(np.array(loss)),
                                #"pred_penalty": np.mean(np.array(pred_penalty)),
                               "lr": r["lr"].numpy()}
                print('')
                print("index | " + " | ".join([f"{i: 8d}" for i in range(log_N+1)]))
                print("loss_x  | " + " | ".join(
                    [f"{loss_x_dict[i] / counters_dict[i]: 8.4f}" for i in range(log_N+1)]))
                print("loss_y  | " + " | ".join(
                    [f"{loss_y_dict[i] / counters_dict[i]: 8.4f}" for i in range(log_N+1)]))
                print("norm  | " + " | ".join(
                    [f"{norm_dict[i] / counters_dict[i]: 8.4f}" for i in range(log_N+1)]))
                print("pred  | " + " | ".join(
                    [f"{pred_dict[i] / counters_dict[i]: 8.4f}" for i in range(log_N+1)]))
                print('grad_norm: ', r['grad_norm'])
                pg = np.mean(polar_good_buf)
                pb = np.mean(polar_bad_buf)
                pm = np.mean(polar_mid_buf)
                print(f"polarization: good={pg:.3f}  mid={pm:.3f}  bad={pb:.3f}  (good+bad={pg+pb:.3f})")
                result_dict.update({"stats_table": table,
                                    "polar_good": pg,
                                    "polar_bad":  pb,
                                    "polar_mid":  pm})
                polar_good_buf, polar_bad_buf, polar_mid_buf = [], [], []
                loss_y_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
                norm_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
                pred_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
                counters_dict = {i: 0.0 for i in range(int(np.log2(train_block_length) + 1))}
                wandb.log(result_dict)
                print(f"iterations: {k + 1}, "
                        f"ce_y: {np.mean(np.array(ce_y)): 6.5f} "
                        f"ce_x: {np.mean(np.array(ce_x)): 6.5f} "
                        f"mi: {np.mean(np.array(mi)): 6.5f} "
                        f"loss: {np.mean(np.array(loss)): 4.3e} "
                      f"pred_penalty: {np.mean(np.array(pred_penalty)): 6.5f} "
                      f"lr: {np.round(r['lr'],8)} "
                      f"elapsed: {time() - t: 4.3f}")
                loss = []
                mi = []
                ce_x = []
                ce_y = []
                pred_penalty = []
                t = time()

            # if k == 0:
            #     trainable_params_count = np.sum([np.prod(v.shape) for v in self.trainable_weights])
            #     wandb.run.summary['trainable_params_count'] = trainable_params_count
            #     print(f"trainable_params_count: {trainable_params_count:,}")



            if time() - t_save > saving_freq:
                self.reset_metrics()
                print("reset metrics")
                self.checknode.built = True
                self.bitnode.built = True
                self.emb2llr.built = True
                self.Ey.built = True
                self.Ex.built = True
                self.Ex_enc.built = True
                if save_model:
                    self.save_model(save_name)
                t_save = time()

        if save_model:
            self.save_model(save_name)

    def optimize(self, train_block_length=8, train_batch=10, num_iters=10000,
                 logging_freq=1000, saving_freq=3600, load_nsc_path=None, save_model=False):
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)

        save_name = f'optimize_group-{wandb.run.group}_{self.input_distribution_name}_{self.channel.save_name}_nt-{train_block_length}_' \
                    f'npd-{self.embedding_size}-{self.layers_per_op}x' \
                    f'{self.hidden_size}'

        self.reset_metrics()

        t_save = time()
        t = time()
        for k in tqdm(range(num_iters)):
            #_ = self.estimation_step(train_batch, train_block_length, train_ex=True)
            r = self.improvement_step(train_batch, train_block_length)
            # Compute gradients


            if k % logging_freq == 0:
                wandb.log({"iter_scl": k,
                           "ce_x": r["ce_x"].numpy(),
                           "ce_y": r["ce_y"].numpy(),
                           "mi": (r["ce_x"].numpy() - r["ce_y"].numpy()),
                           "loss": r["loss"].numpy()})
                print(f"iterations: {k + 1}, "
                      f"ce_x: {r['ce_x']: 6.5f} "
                      f"ce_y: {r['ce_y']: 6.5f} "
                      f"mi: {(r['ce_x'].numpy() - r['ce_y'].numpy()): 6.5f} "
                      f"loss: {r['loss']: 4.3e} "
                      f"px: {r['px']: 6.5f}",
                      f"lr: {self.improvement_optimizer.learning_rate.numpy(): 6.5e}",
                      f"elapsed: {time() - t: 4.3f}")
                t = time()
                self.estimation_optimizer.learning_rate.assign(self.estimation_optimizer.learning_rate * 0.9995)
                self.improvement_optimizer.learning_rate.assign(self.improvement_optimizer.learning_rate * 0.9995)

            if time() - t_save > saving_freq:
                print("reset metrics...")
                self.reset_metrics()
                if save_model:
                    self.save_model(save_name)
                t_save = time()

        if save_model:
            self.save_model(save_name)

    def save_model(self, save_name):
        try:
            save_tmp_path = f"./artifacts/tmp/{wandb.run.name}/design.npy"
            # Extract the directory part of the path
            directory = os.path.dirname(save_tmp_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            weights_artifact = wandb.Artifact(save_name, type='model_weights')
            #self.layer_norms_ex.save_weights(f"{directory}/layer_norms_ex.h5")
            if isinstance(self.layer_norms_ey, BatchNormModel):
                self.layer_norms_ey.save_weights(f"{directory}/layer_norms_ey.h5")
            self.checknode.save_weights(f"{directory}/checknode.weights.h5")
            self.bitnode.save_weights(f"{directory}/bitnode.weights.h5")
            self.emb2llr.save_weights(f"{directory}/emb2llr.weights.h5")
            self.Ey.save_weights(f"{directory}/ey.weights.h5")
            self.Ex.save_weights(f"{directory}/ex.weights.h5")

            # Add the model weights file to the artifact
            #weights_artifact.add_file(f"{directory}/layer_norms_ex.h5")
            if isinstance(self.layer_norms_ey, BatchNormModel):
                weights_artifact.add_file(f"{directory}/layer_norms_ey.h5")
            weights_artifact.add_file(f"{directory}/checknode.weights.h5")
            weights_artifact.add_file(f"{directory}/bitnode.weights.h5")
            weights_artifact.add_file(f"{directory}/emb2llr.weights.h5")
            weights_artifact.add_file(f"{directory}/ey.weights.h5")
            weights_artifact.add_file(f"{directory}/ex.weights.h5")

            if self.input_distribution == 'sc' or self.input_distribution == 'sct':
                self.Ex_enc.save_weights(f"{directory}/ex_enc.weights.h5")
                weights_artifact.add_file(f"{directory}/ex_enc.weights.h5")
            elif self.input_distribution == 'rnn':
                self.input_model.save_weights(f"{directory}/input_model.weights.h5")
                weights_artifact.add_file(f"{directory}/input_model.weights.h5")
            else:
                raise ValueError(f'input_distribution received invalid value: {self.input_distribution}')

            # Log the artifact to W&B
            artifact_log = wandb.log_artifact(weights_artifact)
            artifact_log.wait()
            # print(f"model is saved as {save_name}")
            artifact = wandb.run.use_artifact(f"{save_name}:latest")
            print(f"Artifact: {artifact.source_name}")
        except Exception as e:
            print(f"An error occurred: {e}")

class SCNeuralListDecoder(SCListDecoder, SCNeuralDecoder):

    def __init__(self, channel, embedding_size, hidden_size, layers_per_op, eyN0=False,
                 activation='elu', batch=100, list_num=4, crc=None, trained_block_norm=10,
                 layers_per_op_emb2llr=None, emb2llr_snr=False, llr_clip=30.0, *args, **kwargs):
        SCListDecoder.__init__(self, channel=channel, batch=batch, list_num=list_num, crc=crc,  eyN0=eyN0, *args, **kwargs)
        SCNeuralDecoder.__init__(self, channel=channel, batch=batch, embedding_size=embedding_size,
                                 hidden_size=hidden_size, layers_per_op=layers_per_op, activation=activation,
                                 trained_block_norm=trained_block_norm, eyN0=eyN0,
                                 layers_per_op_emb2llr=layers_per_op_emb2llr, emb2llr_snr=emb2llr_snr,
                                 llr_clip=llr_clip, *args, **kwargs)
        self.checknode_list = self.checknode
        self.bitnode_list = self.bitnode
        self.emb2llr_list = self.emb2llr

        # only ey
        # self.Ex_enc = EmbeddingX(self.input_logits)
        # self.checknode_enc = CheckNodeVanilla()
        # self.bitnode_enc = BitNodeVanilla()
        # self.emb2llr_enc = Activation(tf.identity)

class SCLsionnaDecoder(SCListDecoder):
    def __init__(self, channel, batch=100, list_num=4, Ns=(10,), code_rate=0.3, crc=None, crc_oracle=None, mode='sc', link_channel='uplink', *args, **kwargs):
        SCListDecoder.__init__(self, channel, batch=batch, list_num=list_num, crc=crc, crc_oracle=None, *args, **kwargs)
        n = int(2 ** Ns[0])
        k = int(n * code_rate)
        self.info_size = k
        self.nL = list_num
        self.mode = mode
        self.crc = crc
        if self.crc is not None:
            self.crc_enc = CRCEncoder(crc_degree=crc)
            self.crc_dec = CRCDecoder(crc_encoder=self.crc_enc)
        m = self.channel.N.bit_length() - 1  # number of bits needed to represent indices
        indices = np.arange(self.channel.N)
        self.bit_reversed_indices = np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])
        if self.crc is not None and mode != '5g':
            self.crc_enc = CRCEncoder(crc_degree=crc)
            self.crc_dec = CRCDecoder(crc_encoder=self.crc_enc)
            k += self.crc_enc.crc_length
            self.crc_oracle = crc_oracle
        if mode == "LDPC":
            self.encoder = sn.fec.ldpc.LDPC5GEncoder(k, n, )
            # The decoder provides hard-decisions on the information bits
            self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True, num_iter=20)

        elif mode == '5g':  # scl + crc
            self.encoder = MyPolar5GEncoder(k=k, n=self.channel.E*self.channel.BPS, verbose=True, list_size=list_num, channel_type=link_channel)
            self.pre_decoder = MyPreDecoder(self.encoder, dec_type='SCL', list_size=self.nL)

            #print('pilots,',channel.rg.num_pilot_symbols)
            #print('E,',int(n-channel.rg.num_pilot_symbols*channel.BPS))
            print('E,',self.channel.E*self.channel.BPS)
            m = self.channel.N.bit_length() - 1  # number of bits needed to represent indices
            indices = np.arange(self.channel.N)
            self.bit_reversed_indices = np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])

            dec_type = 'SCL'
            self.decoder = Polar5GDecoder(self.encoder, dec_type=dec_type, list_size=list_num)
            if list_num == 1:
                self.decoder._k_crc = -k
            self.f, self.A = generate_5g_ranking(k, n)
            #self.decoder_sc =PolarSCLDecoder(self.f, n, list_size=list_num, crc_degree=crc)
            self.decoder_sc =PolarSCDecoder(self.f, n)


        elif mode == 'scl':  # scl + crc
            self.f, self.A = generate_5g_ranking(k, n)
            self.encoder = PolarEncoder(self.f, n)
            self.decoder = PolarSCLDecoder(self.f, n, list_size=list_num, crc_degree=crc)
            self.decoder_sc = PolarSCLDecoder(self.f, n, list_size=list_num, crc_degree=crc)

        elif mode == 'sc':  # sc
            # design_name = f"{self.construction_name(10, 'sc')}:latest"
            # self.f = self.load_design(design_name)
            # self.f = self.f[k:]
            # print(f"Design loaded: {design_name}")
            self.f, self.A = generate_5g_ranking(k, n)
            self.encoder = PolarEncoder(self.f, n)
            self.decoder = PolarSCDecoder(self.f, n)

            self.decoder_sc = PolarSCLDecoder(self.f, n, list_num=1, use_fast_scl=False)


        elif mode == 'rm':
            from sionna.fec.polar.utils import generate_rm_code
            f, _, _, _, _ = generate_rm_code(4, 10)  # equals k=64 and n=128 code
            enc = PolarEncoder(f, n)
            dec = PolarSCLDecoder(f, n, list_size=list_num)
        elif mode == 'turbo':
            from sionna.fec.turbo import TurboEncoder, TurboDecoder

            self.encoder = TurboEncoder(rate=k/n, constraint_length=4,
                               terminate=False)  # no termination used due to the rate loss
            self.decoder = TurboDecoder(self.encoder, num_iter=list_num)
        # if self.interleaving:
        #     self.inter = InterleavingMethod(n, True)

    def list_decoder_with_5g(self, ey, batch, N, info_indices, A, sn=True):
        bits_hat2 = self.decoder(ey[..., 0])[..., None]
        #ey = self.pre_decoder.call(ey[...,0])[...,None]
        #if sn:
       # bits_hat3 = self.decoder_sc(ey[..., 0])[...,None]  # this is needed to initialize the decoder
        ###
        #else:

        ey = tf.gather(ey, self.bit_reversed_indices, axis=1)
        ex = tf.zeros_like(ey)
        ey_ = tf.expand_dims(ey, 1)
        ex_ = tf.expand_dims(ex, 1)
        repmat = tf.tensor_scatter_nd_update(tensor=tf.ones_like(tf.shape(ex_)),
                                             indices=tf.constant([[1]]),
                                             updates=tf.constant([self.nL]))
        ex_dup = tf.tile(ex_, repmat)
        ey_dup = tf.tile(ey_, repmat)
        pm = tf.concat([tf.zeros([1]), tf.ones([self.nL - 1]) * float('inf')], 0)
        pm = tf.tile(tf.expand_dims(pm, 0), [batch, 1])
        t = time()
        r = tf.ones(shape=(batch, N, 1), dtype=tf.float32)
        r = tf.tile(tf.expand_dims(r, 1), [1, self.nL, 1, 1])

        # tensor = tf.squeeze(u, axis=2)
        tensor = tf.zeros(shape=(batch, N), dtype=dtype)
        updates = 0.5 * tf.ones([batch * tf.shape(A)[0]], dtype=dtype)
        f_dec = tf.expand_dims(tf.tensor_scatter_nd_update(tensor, info_indices, updates), axis=2)
        f_dec = tf.tile(tf.expand_dims(f_dec, 1), [1, self.nL, 1, 1])


        uhat_list, xhat, llr_u, llr_uy, pm, new_order = self.decode_list(ex_dup, ey_dup, f_dec, pm,
                                                                         f_dec.shape[2], r, sample=True)
        # self.crc_dec = self.decoder._polar_dec._crc_decoder
        # uhat = self.choose_codeword_crc(uhat_list, pm, A)
        if self.crc is None:
            uhat = self.choose_codeword_pm(uhat_list, pm)
        else:
            uhat = self.choose_codeword_crc(uhat_list, pm, A)
        bits_hat = tf.gather(uhat, A, axis=1)
        return bits_hat


    #@tf.function
    def forward_eval(self, batch, N, info_indices, frozen_indices, A, Ac, ebno=None):
        # generate the information bits
        bits_info = tf.cast(tf.random.uniform((batch, self.info_size), minval=0, maxval=2, dtype=tf.int32), dtype)

        if self.crc is not None and self.mode == 'scl':
            bits = self.crc_enc(bits_info)
        else:
            bits = bits_info
        # encode
        x  = self.encoder(bits)[..., None]
        y = self.channel.sample_channel_outputs(x, ebno)

        # decode and compute the errors
        ey = self.Ey(y)
        bits_hat = self.list_decoder_with_5g(ey, batch, N, info_indices, A)[...,0]
        info_bit_errors = tf.squeeze(tf.cast(tf.where(tf.equal(bits_hat[..., None], bits_info[..., None]), 0, 1), dtype),
                                     axis=-1)
        # bits_hat = self.decoder(ey[..., 0])
        # info_bit_errors = tf.squeeze(tf.cast(tf.where(tf.equal(bits_hat[..., None], bits_info[..., None]), 0, 1), dtype),
        #                              axis=-1)

        return tf.reduce_mean(info_bit_errors, axis=1), 0, bits_info, bits_hat

    def eval(self, Ns, mc_length=100000, code_rate=0.25, batch=100, tol=100, load_nsc_path=None, design_path=None,
             design_load=False, mc_design=10e7):
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
            decoder_name = load_nsc_path
        else:
            decoder_name = 'sc'

        bers, fers = list(), list()
        for n in Ns:
            log_dict = {}
            #print(n)
            t = time()
            # if design_path is None:
            #     try:
            #         if design_load:
            #             design_name = f"{self.construction_name(n, decoder_name)}:latest"
            #             sorted_bit_channels = self.load_design(design_name)
            #             print(f"Design loaded: {design_name}")
            #         else:
            #             raise Exception("design_load flags is False")
            #     except Exception as e:
            #         print(f"An error occurred: {e}")
            #         Hu, Huy, sorted_bit_channels = self.polar_code_design(n, batch, mc_design, tol=tol)
            #         self.save_design(n, sorted_bit_channels, decoder_name)
            #         log_dict.update(self.design2dict(n, Hu, Huy))
            #         log_dict.update({'mi': tf.reduce_mean(Hu * np.log(2) - Huy * np.log(2)).numpy() / np.log(2)})
            # else:
            #     sorted_bit_channels = self.load_design(design_path)
            design_time = time() - t
            k = int(code_rate * (2 ** n))

            t = time()
            err = self.polar_code_err_prob(n, mc_length, batch, None, k)
            mc_time = time() - t
            ber = np.mean(err)
            fer = np.mean(err > 0)
            bers.append(ber)
            fers.append(fer)
            print(f"n: {n: 2d} design time: {design_time: 4.1f} "
                  f"code rate: {code_rate: 5.4f} #of mc-blocks: {mc_length} mc time: {mc_time: 4.1f} "
                  f"ber: {ber: 4.3e} fer: {fer: 4.3e}")
            log_dict.update({"n": n,
                             "ber": ber,
                             "fer": fer,
                             "code_rate": code_rate})
            wandb.log(log_dict)

        if len(Ns) != 1:
            x_values = np.array(Ns)
            y_values = np.array(bers)
            z_values = np.array(fers)
            data = [[x, y, z] for (x, y, z) in zip(x_values, y_values, z_values)]
            table = wandb.Table(data=data, columns=["n", "ber", "fer"])
            wandb.log({f"ber": table})

    def choose_information_and_frozen_sets(self, sorted_bit_channels, k):
        if self.mode=='5g' or self.mode=='LDPC' or self.mode=='turbo':
            Ac, A = generate_5g_ranking(k, self.channel.N)
            # Ac = list(Ac)
            # Ac.append(A[-1])
            # A = A[:-1]
        else:
            Ac = self.f
            A = self.A
        return A, Ac

    def polar_code_err_prob(self, n, mc_err, batch, sorted_bit_channels, k, num_target_block_errors=100):
        A, Ac = self.choose_information_and_frozen_sets(sorted_bit_channels, k)
        X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(A, tf.int32))
        info_indices = tf.stack([tf.reshape(tf.transpose(X, perm=[1, 0]), -1),
                                 tf.reshape(tf.transpose(Y, perm=[1, 0]), -1)], axis=1)
        X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(Ac, tf.int32))
        frozen_indices = tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1)], axis=1)

        mc_err = (mc_err // batch + 1) * batch
        err = np.zeros(shape=0)
        t = time()
        biterrd = tf.zeros([2 ** n], dtype=dtype)
        count = 0
        block_errors = 0
        for i in range(0, mc_err, batch):
            bit_errors, errors, u, u_hat = self.forward_eval(batch, 2 ** n, info_indices, frozen_indices, A, Ac)
            biterrd += np.sum(errors, axis=0)
            err = np.concatenate((err, bit_errors))
            count+=batch
            # froze_bit_errors = tf.gather(params=errors,
            #                              indices=Ac,
            #                              axis=1)
            # if np.sum(froze_bit_errors) > 0:
            #     print('frozen bit errors', np.sum(froze_bit_errors))
            #     print('frozen bit arg', np.where(froze_bit_errors > 0)[0].shape[0]/batch)
            block_errors += np.sum(bit_errors > 0)

            if time()-t > 60:
                ber = np.mean(err)
                fer = np.mean(err > 0)
                print(
                    f'iter: {i / mc_err * 100 :5.3f}% | ber: {ber : 5.3e} fer {fer : 5.3e}| block errors: {block_errors}')
                t = time()
                if block_errors >= num_target_block_errors:
                    print('block errors reached')
                    break
        biterrd /= count # plt.figure();plt.semilogy(np.cumsum(np.sort(biterrd))/np.arange(1, 2**n+1)); plt.show()
        bercumsums = np.cumsum(np.sort(biterrd)) / np.arange(1, 2 ** n + 1)
        # print('# zero error bits:', np.sum(biterrd==0))
        # print('# design eq to zero bits:', np.sum([i in A for i in np.argsort(np.array(biterrd))[0:307]]))
        wandb.define_metric("bercumsum_decode", step_metric="bit_num_decode")
        for i, bercumsum in enumerate(bercumsums):
            wandb.log({"bercumsum_decode": bercumsum, "bit_num_decode": i})
        return err

    def get_parameters(self, decoder_name, Ns, mc_length=100000, code_rate=0.25, batch=100, tol=100, load_nsc_path=None, design_path=None,
             design_load=False, mc_design=10e7, ebno_db=None, design5G=False):
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
            decoder_name = load_nsc_path
        else:
            decoder_name = 'sc'
        for n in Ns:
            k = int(code_rate * (2 ** n))
            sorted_bit_channels = None
            A, Ac = self.choose_information_and_frozen_sets(sorted_bit_channels, k)
            X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(A, tf.int32))
            info_indices = tf.stack([tf.reshape(tf.transpose(X, perm=[1, 0]), -1),
                                     tf.reshape(tf.transpose(Y, perm=[1, 0]), -1)], axis=1)
            X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(Ac, tf.int32))
            frozen_indices = tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1)], axis=1)
        return batch, n, info_indices, frozen_indices, A, A

class PolarSCL5GDecoder(SCListDecoder):
    def __init__(self, channel, batch=100, list_num=4, Ns=(10,), code_rate=0.3, crc=None, crc_oracle=None, mode='sc', link_channel='uplink', *args, **kwargs):
        SCListDecoder.__init__(self, channel, batch=batch, list_num=list_num, crc=crc, crc_oracle=None, *args, **kwargs)
        n = int(2 ** Ns[0])
        k = int(n * code_rate)
        self.info_size = k
        self.nL = list_num
        self.mode = mode
        self.crc = crc
        if self.crc is not None and mode != '5g':
            self.crc_enc = CRCEncoder(crc_degree=crc)
            self.crc_dec = CRCDecoder(crc_encoder=self.crc_enc)
            k += self.crc_enc.crc_length
            self.crc_oracle = crc_oracle
        if mode == "LDPC":
            self.encoder = sn.fec.ldpc.LDPC5GEncoder(k, n, )
            # The decoder provides hard-decisions on the information bits
            self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True, num_iter=20)

        elif mode == '5g':  # scl + crc
            self.encoder = MyPolar5GEncoder(k=k, n=self.channel.E*self.channel.BPS, verbose=True, list_size=list_num, channel_type=link_channel)
            #print('pilots,',channel.rg.num_pilot_symbols)
            #print('E,',int(n-channel.rg.num_pilot_symbols*channel.BPS))
            print('E,',self.channel.E*self.channel.BPS)
            # self.channel.CODERATE = k/(self.channel.E*self.channel.BPS)
            dec_type = 'SCL'
            self.decoder = Polar5GDecoder(self.encoder, dec_type=dec_type, list_size=list_num)
            if list_num == 1:
                self.decoder._k_crc = -k
        elif mode == '5g_design':  # scl + crc
            self.encoder = MyPolar5GEncoder(k=k, n=self.channel.E * self.channel.BPS, verbose=True, list_size=list_num,
                                            channel_type=link_channel)
            self.encoder_design = MyPolar5GEncoder(k=self.channel.E * self.channel.BPS, n=self.channel.E * self.channel.BPS, verbose=True,
                                            channel_type=link_channel)
            self.k = k
            self.link_channel = link_channel
            # print('pilots,',channel.rg.num_pilot_symbols)
            # print('E,',int(n-channel.rg.num_pilot_symbols*channel.BPS))
            print('E,', self.channel.E * self.channel.BPS)
            # self.channel.CODERATE = k/(self.channel.E*self.channel.BPS)
            dec_type = 'SCL'
            from models.my5gpolar import  Polar5GDecoder_design
            self.decoder_design = Polar5GDecoder_design(self.encoder, dec_type="SC")
            self.decoder = Polar5GDecoder(self.encoder, dec_type=dec_type, list_size=list_num)
            if list_num == 1:
                self.decoder._k_crc = -k

        elif mode == 'scl':  # scl + crc
            self.f, self.A = generate_5g_ranking(k, n)
            # myPC = PolarCode(n, k)
            # myPC.construction_type = 'ga'
            # design_SNR = -5.0
            # Construct(myPC, design_SNR)
            # self.f = myPC.reliabilities[:n-k]
            # self.A = myPC.reliabilities[n-k:]
            self.encoder = PolarEncoder(self.f, n)
            self.decoder = PolarSCLDecoder(self.f, n, list_size=list_num, crc_degree=crc)


        elif mode == 'sc':  # sc
            # design_name = f"{self.construction_name(10, 'sc')}:latest"
            # self.f = self.load_design(design_name)
            # self.f = self.f[k:]
            # print(f"Design loaded: {design_name}")
            self.f, self.A = generate_5g_ranking(k, n)
            self.encoder = PolarEncoder(self.f, n)
            self.decoder = PolarSCDecoder(self.f, n)

        elif mode == 'rm':
            from sionna.fec.polar.utils import generate_rm_code
            f, _, _, _, _ = generate_rm_code(4, 10)  # equals k=64 and n=128 code
            enc = PolarEncoder(f, n)
            dec = PolarSCLDecoder(f, n, list_size=list_num)
        elif mode == 'turbo':
            from sionna.fec.turbo import TurboEncoder, TurboDecoder

            self.encoder = TurboEncoder(rate=k/n, constraint_length=4,
                               terminate=False)  # no termination used due to the rate loss
            self.decoder = TurboDecoder(self.encoder, num_iter=list_num)

        self.list_num = list_num
    #@tf.function
    def forward_eval(self, batch, N, info_indices, frozen_indices, A, Ac, ebno=None):
        # GA = False
        # if GA:
        #     myPC = PolarCode(N, A.shape[0])
        #     myPC.construction_type = 'ga'
        #     design_SNR =  ebno
        #     Construct(myPC, design_SNR)
        #     self.f = myPC.reliabilities[:N - A.shape[0]]
        #     self.A = myPC.reliabilities[N - A.shape[0]:]
        #     self.encoder = PolarEncoder(self.f, N)
        #     self.decoder = PolarSCLDecoder(self.f, N, list_size= self.list_num, crc_degree=self.crc)

        # generate the information bits
        bits_info = tf.cast(tf.random.uniform((batch, self.info_size), minval=0, maxval=2, dtype=tf.int32), dtype)


        if self.crc is not None and self.mode == 'scl':
            bits = self.crc_enc(bits_info)
        else:
            bits = bits_info
        # encode
        x = self.encoder(bits)[..., None]
        self.channel.CODERATE = self.info_size/x.shape[1]
        y = self.channel.sample_channel_outputs(x, ebno)

        # decode and compute the errors
        ey = self.Ey(y)

        bits_hat = self.decoder(ey[..., 0])
        if self.crc is not None and self.mode == 'scl':
            bits_hat = bits_hat[:, :-self.crc_enc.crc_length]
        info_bit_errors = tf.squeeze(tf.cast(tf.where(tf.equal(bits_hat[..., None], bits_info[..., None]), 0, 1), dtype),
                                     axis=-1)
        #tf.reduce_mean(tf.reduce_mean(info_bit_errors, axis=1))
        return tf.reduce_mean(info_bit_errors, axis=1), 0, bits_info, bits_hat

    def eval(self, Ns, mc_length=100000, code_rate=0.25, batch=100, tol=100, load_nsc_path=None, design_path=None,
             design_load=False, mc_design=10e7):
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
            decoder_name = load_nsc_path
        else:
            decoder_name = 'sc'

        bers, fers = list(), list()
        for n in Ns:
            log_dict = {}
            #print(n)
            t = time()
            # if design_path is None:
            #     try:
            #         if design_load:
            #             design_name = f"{self.construction_name(n, decoder_name)}:latest"
            #             sorted_bit_channels = self.load_design(design_name)
            #             print(f"Design loaded: {design_name}")
            #         else:
            #             raise Exception("design_load flags is False")
            #     except Exception as e:
            #         print(f"An error occurred: {e}")
            #         Hu, Huy, sorted_bit_channels = self.polar_code_design(n, batch, mc_design, tol=tol)
            #         self.save_design(n, sorted_bit_channels, decoder_name)
            #         log_dict.update(self.design2dict(n, Hu, Huy))
            #         log_dict.update({'mi': tf.reduce_mean(Hu * np.log(2) - Huy * np.log(2)).numpy() / np.log(2)})
            # else:
            #     sorted_bit_channels = self.load_design(design_path)

            design_time = time() - t
            k = int(code_rate * (2 ** n))

            t = time()
            err = self.polar_code_err_prob(n, mc_length, batch, None, k)
            mc_time = time() - t
            ber = np.mean(err)
            fer = np.mean(err > 0)
            bers.append(ber)
            fers.append(fer)
            print(f"n: {n: 2d} design time: {design_time: 4.1f} "
                  f"code rate: {code_rate: 5.4f} #of mc-blocks: {mc_length} mc time: {mc_time: 4.1f} "
                  f"ber: {ber: 4.3e} fer: {fer: 4.3e}")
            log_dict.update({"n": n,
                             "ber": ber,
                             "fer": fer,
                             "code_rate": code_rate})
            wandb.log(log_dict)

        if len(Ns) != 1:
            x_values = np.array(Ns)
            y_values = np.array(bers)
            z_values = np.array(fers)
            data = [[x, y, z] for (x, y, z) in zip(x_values, y_values, z_values)]
            table = wandb.Table(data=data, columns=["n", "ber", "fer"])
            wandb.log({f"ber": table})

    def choose_information_and_frozen_sets_5g(self, sorted_bit_channels, k):
        if self.mode=='5g' or self.mode=='LDPC' or self.mode=='turbo':
            Ac = None
            A = None
        else:
            Ac = self.f
            A = self.A
        return A, Ac

    def polar_code_err_prob(self, n, mc_err, batch, sorted_bit_channels, k, num_target_block_errors=100):
        A, Ac = self.choose_information_and_frozen_sets(sorted_bit_channels, k)
        # Ac, A = generate_5g_ranking(k, 2**n)

        # X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(A, tf.int32))
        info_indices = None
        #                          tf.reshape(tf.transpose(Y, perm=[1, 0]), -1)], axis=1)
        # X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(Ac, tf.int32))
        frozen_indices = None

        mc_err = (mc_err // batch + 1) * batch
        err = np.zeros(shape=0)
        t = time()
        biterrd = tf.zeros([2 ** n], dtype=dtype)
        count = 0
        block_errors = 0
        for i in range(0, mc_err, batch):
            bit_errors, errors, u, u_hat = self.forward_eval(batch, 2 ** n, info_indices, frozen_indices, A, Ac)
            biterrd += np.sum(errors, axis=0)
            err = np.concatenate((err, bit_errors))
            count+=batch
            # froze_bit_errors = tf.gather(params=errors,
            #                              indices=Ac,
            #                              axis=1)
            # if np.sum(froze_bit_errors) > 0:
            #     print('frozen bit errors', np.sum(froze_bit_errors))
            #     print('frozen bit arg', np.where(froze_bit_errors > 0)[0].shape[0]/batch)
            block_errors += np.sum(bit_errors > 0)

            if time()-t > 60:
                ber = np.mean(err)
                fer = np.mean(err > 0)
                print(
                    f'iter: {i / mc_err * 100 :5.3f}% | ber: {ber : 5.3e} fer {fer : 5.3e}| block errors: {block_errors}')
                t = time()
                if block_errors >= num_target_block_errors:
                    print('block errors reached')
                    break
        biterrd /= count # plt.figure();plt.semilogy(np.cumsum(np.sort(biterrd))/np.arange(1, 2**n+1)); plt.show()
        bercumsums = np.cumsum(np.sort(biterrd)) / np.arange(1, 2 ** n + 1)
        # print('# zero error bits:', np.sum(biterrd==0))
        # print('# design eq to zero bits:', np.sum([i in A for i in np.argsort(np.array(biterrd))[0:307]]))
        wandb.define_metric("bercumsum_decode", step_metric="bit_num_decode")
        for i, bercumsum in enumerate(bercumsums):
            wandb.log({"bercumsum_decode": bercumsum, "bit_num_decode": i})
        return err

    def forward_design(self, batch, N, ebno_db=None):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        ex_enc = self.Ex_enc.call(batch_N_shape)

        # generate shared randomness
        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)

        # create frozen bits for encoding. encoded bits need to be 0.5.
        f_enc = 0.5 * tf.ones(shape=(batch, N, 1))
        u, x, llr_u1 = self.encode(ex_enc, f_enc, N, r, sample=True)
        u = np.float32(tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32) > 0.5)
        x = self.encoder_design(u[...,0])[...,None]
        llr_u = tf.squeeze(tf.where(tf.equal(u, 1.0), llr_u1, -llr_u1), axis=2)
        h_u = tf.reduce_sum(-tf.math.log(tf.math.sigmoid(llr_u)), axis=0)
        # 5G encoder mode
        # if self.encoder5g:
        #     x = self.encoder5g(u[...,0])[...,None]
        x_before_rm = x
        x, rep_ind = self.NeuralRateMatching(x, N, self.channel_design.E*self.channel_design.BPS)
        bits_size = x.shape[1]
        y = self.channel_design.sample_channel_outputs(x, ebno_db)
        # decode and compute the errors
        ey = self.Ey(y)
        llrs = self.decoder_design(ey[..., 0], u[..., 0])
        pass
        pred = np.int32(llrs < 0)
        # u_2, loss_array, pred, norm_array = self.fast_ce( ey, x_before_rm[:,:N])
        errors = tf.cast(tf.not_equal(u[..., 0], pred), tf.float32)
        errors = tf.reduce_sum(errors, axis=0)
        loss_array =  self.loss_fn(u, -llrs[..., None])

        h_uy = tf.reduce_sum(loss_array, axis=0)
        return h_u, h_uy, errors

    def get_parameters(self, decoder_name, Ns, mc_length=100000, code_rate=0.25, batch=100, tol=100, load_nsc_path=None, design_path=None,
             design_load=False, mc_design=10e7, ebno_db=None, design5G=False):
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
            decoder_name = load_nsc_path
        else:
            decoder_name = 'sc'
        for n in Ns:
            k = int(code_rate * (2 ** n))
            sorted_bit_channels = None
            if self.mode == '5g_design':
                Hu, Huy, sorted_bit_channels = self.polar_code_design(n, batch, mc_design, tol=tol, ebno_db=ebno_db)
                A, Ac = self.choose_information_and_frozen_sets(sorted_bit_channels, k)
                self.encoder = MyPolar5GEncoder(k=self.k, n=self.channel.E*self.channel.BPS, verbose=True, list_size=self.list_num, channel_type=self.link_channel, ch_ranking=sorted_bit_channels)
                self.decoder = Polar5GDecoder(self.encoder, dec_type="SCL", list_size=self.list_num)
                # self.encoder = MyPolar5GEncoder(k=self.k, n=self.channel.E*self.channel.BPS, verbose=True ,channel_type=self.link_channel, ch_ranking=sorted_bit_channels, list_size=1)
                # self.decoder = Polar5GDecoder(self.encoder, dec_type="SC")#, list_size=self.list_num)

            else:
                A, Ac = self.choose_information_and_frozen_sets_5g(sorted_bit_channels, k)
            info_indices = None
            frozen_indices = None
        return batch, n, info_indices, frozen_indices, A, Ac
