import numpy as np
from sionna.phy.channel import AWGN
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.phy.mapping import Constellation, Mapper, Demapper




class GilbertElliottChannel:
    def __init__(self, n0_g, n0_b, p_gb, p_bg, num_bits_per_symbol=1, seed=None):
        """
        n0_g: noise veriance in GOOD state
        n0_b: noise veriance in BAD state
        p_gb : P(G -> B)
        p_bg : P(B -> G)
        e_good : error probability in the GOOD state
        e_bad  : error probability in the BAD state
        """
        self.p_gb = p_gb
        self.p_bg = p_bg
        self.n0_g = n0_g
        self.n0_b = n0_b
        self._awgn = AWGN(dtype='float32')
        self.save_name = "GE_channel"
        self.rng = np.random.default_rng(seed)
        self.BPS = num_bits_per_symbol
        if self.BPS == 1:
            bpsk_constellation = Constellation("custom", 1,
                                                          points=tf.convert_to_tensor([1, -1]))
            self.mapper = Mapper(constellation_type="custom",
                                            num_bits_per_symbol=self.BPS,
                                            constellation=bpsk_constellation)
        else:
            self.mapper =Mapper("qam", self.BPS)


        self.cardinality_x = self.cardinality_s = 2
        self.cardinality_y = 2
        if self.BPS == 1:
            self.demapper = Demapper("app", "custom", self.BPS, constellation=bpsk_constellation)
        else:
            self.demapper = Demapper("app", "qam", self.BPS)


    def generate_states(self, batch_size, n):
        """Generate hidden Markov chain states for each example in batch: 0=GOOD, 1=BAD"""
        s = np.zeros((batch_size, n), dtype=int)
        for b in range(batch_size):
            for i in range(1, n):
                if s[b, i - 1] == 0:  # GOOD
                    s[b, i] = 1 if self.rng.random() < self.p_gb else 0
                else:  # BAD
                    s[b, i] = 0 if self.rng.random() < self.p_bg else 1
        return s

    def llr(self, y):
        y_real_part = y[..., 0]
        y_imag_part = y[..., 1]
        y_complex = tf.complex(y_real_part, y_imag_part)
        llr_good = self.demapper(y_complex, self.n0_g)
        llr_bad = self.demapper(y_complex, self.n0_b)
        states = self.states
        good_mask = (states == 0)
        llr = tf.where(good_mask, llr_good, llr_bad)
        llr = llr[:, :, tf.newaxis]
        return llr


    #@tf.function
    def sample_channel_outputs(self, c, ebno_db=None):
        n = c.shape[1]
        batch_size = c.shape[0]
        states = self.generate_states(batch_size, n)
        self.states = states
        good_mask = (states == 0)
        #bad_mask = (s == 1)
        #y = np.zeros_like(c)
        s = self.mapper(c)
        y_good = self._awgn(s, self.n0_g)
        y_bad = self._awgn(s, self.n0_b)
        # Use tf.where to select based on mask
        y = tf.where(good_mask[...,None], y_good, y_bad)[...,0]
        # Convert complex to real representation
        y = y[:,None,None]
        y_complex = tf.reshape(y, tf.concat((y.shape[:3], [-1]), axis=0))
        y_complex = tf.squeeze(y_complex)[..., tf.newaxis]
        y_real = tf.concat([tf.math.real(y_complex), tf.math.imag(y_complex)], axis=-1)
        return y_real


if __name__ == "__main__":
    # Example usage
    n0_g = 0.5
    n0_b = 2.0
    p_gb = 1/16
    p_bg = 1/16
    channel = GilbertElliottChannel(n0_g, n0_b, p_gb, p_bg)

    # Generate random input symbols
    y = channel.sample_channel_outputs(tf.zeros((200,1024,1),dtype=tf.int32))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(tf.math.real(y[0,:,0]), label='Real Part')
    plt.plot(tf.math.imag(y[0,:,0]), label='Imaginary Part')
    plt.title('Channel Output over Gilbert-Elliott Channel')
    plt.xlabel('Symbol Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
