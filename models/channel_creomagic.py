import os
from absl.flags import get_help_width
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import time_to_ofdm_channel, TimeChannel
from tensorflow import newaxis
from sionna.phy.mimo import StreamManagement
from tensorflow.signal import ifftshift
import sionna
from sionna.phy.signal import ifft
from sionna.phy.channel.utils import cir_to_time_channel, time_lag_discrete_time_channel
from sionna.phy.channel import GenerateTimeChannel, ApplyTimeChannel
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, ZFEqualizer,  OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.signal import ifft
from sionna.phy import Block, PI
from sionna.phy.utils import expand_to_rank
from sionna.phy.signal import fft
from tensorflow.signal import ifftshift
import os
from scipy.linalg import circulant
from scipy.linalg import bandwidth
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import time_to_ofdm_channel, TimeChannel
from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator, LMMSEEqualizer
from tensorflow import newaxis
from tensorflow.signal import fftshift
from sionna.phy.mimo import StreamManagement
from tensorflow.signal import ifftshift
import sionna
from sionna.phy.mimo import lmmse_equalizer
from sionna.phy.mimo import MaximumLikelihoodDetector
from sionna.phy import Block
from sionna.phy.utils import flatten_last_dims
from sionna.phy.signal import ifft
from sionna.phy import Block, PI
from sionna.phy.utils import expand_to_rank
from sionna.phy.signal import fft
from sionna.phy.channel.utils import cir_to_time_channel, time_lag_discrete_time_channel
from sionna.phy.channel import GenerateTimeChannel, ApplyTimeChannel
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, ZFEqualizer,  OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers


if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

import tensorflow as tf
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducibility
sionna.phy.config.seed = 42
# Load the required Sionna components
from sionna.phy import Block
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec.polar.utils import generate_5g_ranking, generate_rm_code
from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder
from sionna.phy.fec.linear import OSDecoder
from sionna.phy.utils import count_block_errors, ebnodb2no, PlotBER
from sionna.phy.channel import AWGN
import matplotlib.pyplot as plt
import numpy as np
import time # for throughput measurements
from datetime import datetime
import tensorflow as tf

class MyDemapper(Block):
    # pylint: disable=line-too-long
    r"""
    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    for a tensor of received symbols

    Prior knowledge on the bits can be optionally provided.

    This class defines a block implementing different demapping
    functions. All demapping functions are fully differentiable when soft-decisions
    are computed.

    Parameters
    ----------
    demapping_method : "app" | "maxlog"
        Demapping method

    constellation_type : "qam" | "pam" | "custom"
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : `int`
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : `None` (default) | :class:`~sionna.phy.mapping.Constellation`
        If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool, (default `False`)
        If `True`, the demapper provides hard-decided bits instead of soft-values.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    y : [...,n], `tf.complex`
        Received symbols

    no : Scalar or [...,n], `tf.float`
        The noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.

    prior : `None` (default) | [num_bits_per_symbol] or [...,num_bits_per_symbol], `tf.float`
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.

    Output
    ------
    : [...,n*num_bits_per_symbol], `tf.float`
        LLRs or hard-decisions for every bit

    Note
    ----
    With the "app" demapping method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point and is set to :math:`\mathbf{0}` if no prior knowledge is assumed to be available,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" demapping method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)\\
                &= \max_{c\in\mathcal{C}_{i,0}}
                    \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
                 \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
                .
        \end{align}
    """
    def __init__(self,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        # Create constellation object
        self._constellation = Constellation.check_or_create(
                                constellation_type=constellation_type,
                                num_bits_per_symbol=num_bits_per_symbol,
                                constellation=constellation,
                                precision=precision)

        num_bits_per_symbol = self._constellation.num_bits_per_symbol

        # self._logits2llrs = SymbolLogits2LLRs(demapping_method,
        #                                       num_bits_per_symbol,
        #                                       hard_out=hard_out,
        #                                       precision=precision,
        #                                       **kwargs)

        self._no_threshold = tf.cast(np.finfo(self.rdtype.as_numpy_dtype).tiny,
                                     self.rdtype)

    @property
    def constellation(self):
        """
        :class:`~sionna.phy.mapping.Constellation` : Constellation used by the
            Demapper
        """
        return self._constellation

    def call(self, r, no, prior=None):

        # Reshape constellation points to [1,...1,num_points]
        points_shape = [1]*r.shape.rank + self.constellation.points.shape
        points = tf.reshape(self.constellation.points, points_shape)
        # Compute squared distances from y to all points
        # shape [...,n,num_points]
        # squared_dist = tf.pow(tf.abs(tf.expand_dims(y, axis=-1) - points), 2)
        #
        # # Add a dummy dimension for broadcasting. This is not needed when no
        # # is a scalar, but also does not do any harm.
        # no = tf.expand_dims(no, axis=-1)
        # # Deal with zero or very small values.
        # no = tf.math.maximum(no, self._no_threshold)
        #
        S0 = tf.constant([-1.0], dtype=tf.complex64)  # bit=0
        S1 = tf.constant([+1.0], dtype=tf.complex64)  # bit=1
        d0 = tf.abs(r - S0) ** 2  # distance to -1
        d1 = tf.abs(r - S1) ** 2  # distance to +1
        sigma2 = 1
        L = - (d0 - d1) / sigma2

        # # Compute exponents
        # # exponents = -squared_dist/no
        # #
        # # llr = self._logits2llrs(exponents, prior)
        #
        # # Reshape LLRs to [...,n*num_bits_per_symbol]
        # out_shape = tf.concat([tf.shape(y)[:-1],
        #                        [y.shape[-1] * \
        #                         self.constellation.num_bits_per_symbol]], 0)
        # llr_reshaped = tf.reshape(llr, out_shape)
        # x = np.real(r)
        # y = np.imag(r)
        # l =  - (((x-np.real(self.constellation.points[1]))**2 + (x-np.imag(self.constellation.points[1]))**2 ) - ((y-np.real(self.constellation.points[0]))**2 + (y-np.imag(self.constellation.points[0]))**2 ))


        # y_expanded = tf.expand_dims(r, axis=-1)
        #
        # res = abs(y_expanded-points)**2
        # llrs = -(res[...,1] - res[...,0])
        return L

class creomagic_channel(Block):
    """System model for channel coding BER simulations.

    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to
    initialize the model, or it can run in uncoded mode.

    Parameters
    ----------
        k: int
            number of information bits per codeword.

        n: int
            codeword length.

        num_bits_per_symbol: int
            number of bits per QAM symbol.

        encoder: Sionna Block or None
            A Sionna Block that encodes information bit tensors. Set to None for uncoded.

        decoder: Sionna Block or None
            A Sionna Block layer that decodes llr tensors. Set to None for uncoded.

        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".

        sim_esno: bool
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.

        cw_estimates: bool
            A boolean defaults to False. If true, codewords instead of information estimates are returned.
        uncoded: bool
            A boolean defaults to False. If true, no encoding/decoding is performed.
    """
    def __init__(self,
                 num_bits_per_symbol=1,
                 demapping_method="app", # "app" or "maxlog"
                 channel="awgn",
                 frame_size=64,
                 delay_spread=1000e-9, # in seconds
                 min_speed=0, # in m/s
                 max_speed=0, # in m/s
                ):
        super().__init__()
        self.frame_size = frame_size
        # store values internally

        # number of bit per QAM symbol
        self.num_bits_per_symbol = num_bits_per_symbol

        # initialize mapper and demapper for constellation object
        if self.num_bits_per_symbol == 1:
            self.constellation = Constellation("custom", 1,
                                                          points=tf.convert_to_tensor([1, -1]))
        else:
            self.constellation = Constellation("qam",
                                    num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation,
                                 hard_out=False)

        # the channel can be replaced by more sophisticated models
        self.channel_model = channel
        self.delay_spread = delay_spread
        tdl = TDL(model="A",
                  delay_spread=delay_spread,
                  carrier_frequency=3.5e9,
                  min_speed=min_speed,
                  max_speed=max_speed,)
        self.tdl = tdl
        bandwidth = 15e3*128#2e6 # 2/4/8/16 MHz
        self.bandwidth = bandwidth
        num_time_samples = self.frame_size
        self.l_min, self.l_max = time_lag_discrete_time_channel(self.bandwidth,
                                                                3e-06)
        self.l_min = 0
        self.l_tot = int(self.l_max - self.l_min + 1)
        self.channel = TimeChannel(
            tdl,
            bandwidth,
            num_time_samples,
            l_min=self.l_min,
            l_max=self.l_max,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True)

        self.channel_noise = AWGN()
        self.num_time_steps = self.frame_size + self.l_tot - 1
        ## apply cha
        self._apply_channel = ApplyTimeChannel(self.frame_size + self.zc_size ,
                                               self.l_tot,
                                               precision='single')



    #@tf.function() # enable graph mode for increased throughputs
    def sample_channel_outputs(self, c, ebno_db):
        batch_size = c.shape[0]
        no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=1,
                           coderate=1)
        s_frame = self.mapper(c) # map bits to symbols
        if self.channel_model == "tdl":
            a, tau = self.tdl(batch_size,  self.num_time_steps, self.bandwidth)
            h_time = cir_to_time_channel(self.bandwidth, a, tau, self.l_min,
                                         self.l_max, True)
            y_c = self._apply_channel(s_frame , h_time) # apply channel
        else:
            y_c = s_frame
        y_c = self.channel_noise(y_c, no) # add noise
        y = y_c[..., :self.frame_size]  # remove channel delay extension
        # reshape to block size + cp_length
        y_complex = tf.reshape(y, tf.concat((y.shape[:3], [-1]), axis=0))
        y_complex = tf.squeeze(y_complex)[..., tf.newaxis]
        y_real = tf.concat([tf.math.real(y_complex), tf.math.imag(y_complex)], axis=-1)
        return y_real

if __name__ == "__main__":
    ch = creomagic_channel(num_bits_per_symbol=1,frame_size=1024)
    ch(tf.zeros((16,1024),dtype=tf.int32),10.0)