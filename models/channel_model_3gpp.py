import sionna as sn
import tensorflow as tf
import numpy as np
from sionna.phy.channel.tr38901 import TDL
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.python.keras import initializers

from models.sc_models import InterleavingMethod
from sionna.phy.channel import AWGN, TimeChannel, time_to_ofdm_channel
from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator, LSChannelEstimator, LMMSEEqualizer
from sionna.phy.ofdm import PilotPattern
from sionna.phy.mapping import QAMSource
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import OFDMChannel
from sionna.phy.mapping import Constellation, Mapper, Demapper
import wandb
from models.Rapp import HPA_Rapp
from DigiCommPy.equalizers import zeroForcing  # import MMSE equalizer class
from sionna.phy.ofdm import ResourceGridMapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import ebnodb2no
from sionna.phy.ofdm import tdl_freq_cov_mat, tdl_time_cov_mat
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
                            LMMSEInterpolator, LinearDetector, KBestDetector, \
                            EPDetector, MMSEPICDetector
#tf.keras.backend.set_floatx('float32')
# Define the number of UT and BS antennas
from sionna.phy.mapping import Mapper, Constellation, BinarySource

from tqdm import tqdm
NUM_UT = 1
NUM_BS = 1
NUM_UT_ANT = 1
NUM_BS_ANT = 1
dtype = tf.keras.backend.floatx()


def complex_normal_test(shape, mean=1, var=1.0, dtype=tf.complex64):
    r"""Generates a tensor of complex normal random variables.

    Input
    -----
    shape : tf.shape, or list
        The desired shape.

    var : float
        The total variance., i.e., each complex dimension has
        variance ``var/2``.

    dtype: tf.complex
        The desired dtype. Defaults to `tf.complex64`.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor of complex normal random variables.
    """
    # Half the variance for each dimension
    var_dim = tf.cast(var, dtype.real_dtype)/tf.cast(2, dtype.real_dtype)
    stddev = tf.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    xr = tf.random.normal(shape, stddev=stddev,  mean=mean, dtype=dtype.real_dtype)
    xi = tf.random.normal(shape, stddev=stddev,  mean=mean, dtype=dtype.real_dtype)
    x = tf.complex(xr, xi)
    return x

def create_noise(x, no):
    from sionna.phy.utils import expand_to_rank

    noise = complex_normal_test(tf.shape(x), mean=1, dtype=x.dtype)

    # Add extra dimensions for broadcasting
    no = expand_to_rank(no, tf.rank(x), axis=-1)

    # Apply variance scaling
    noise *= tf.cast(tf.sqrt(no), noise.dtype)
    return noise

def insert_zeros(a, step):
    b = []
    for i in range(len(a)):
        if i % step == 0:
            b.append(0)
        b.append(a[i])
    return b

def bpsk_modulation(bits):
    """
    Perform BPSK modulation on a sequence of bits.

    Args:
        bits (tf.Tensor): A tensor of shape (n,) containing binary bits (0s and 1s).

    Returns:
        tf.Tensor: A tensor of shape (n,) containing BPSK modulated values (+1s and -1s).
    """
    # Ensure the input is a TensorFlow tensor
    # bits = tf.convert_to_tensor(bits, dtype=tf.float32)

    # Perform BPSK modulation: map 0 -> +1 and 1 -> -1
    bpsk_signal = 1 - 2 * bits

    return bpsk_signal

class myPilotPatternFullOFDM(PilotPattern):
    """Simple orthogonal pilot pattern with Kronecker structure.

    This function generates an instance of :class:`~sionna.ofdm.PilotPattern`
    that allocates non-overlapping pilot sequences for all transmitters and
    streams on specified OFDM symbols. As the same pilot sequences are reused
    across those OFDM symbols, the resulting pilot pattern has a frequency-time
    Kronecker structure. This structure enables a very efficient implementation
    of the LMMSE channel estimator. Each pilot sequence is constructed from
    randomly drawn QPSK constellation points.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of a :class:`~sionna.ofdm.ResourceGrid`.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension.
        Defaults to `True`.

    seed : int
        Seed for the generation of the pilot sequence. Different seed values
        lead to different sequences. Defaults to 0.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Note
    ----
    It is required that the ``resource_grid``'s property
    ``num_effective_subcarriers`` is an
    integer multiple of ``num_tx * num_streams_per_tx``. This condition is
    required to ensure that all transmitters and streams get
    non-overlapping pilot sequences. For a large number of streams and/or
    transmitters, the pilot pattern becomes very sparse in the frequency
    domain.

    Examples
    --------
    >>> rg = ResourceGrid(num_ofdm_symbols=14,
    ...                   fft_size=64,
    ...                   subcarrier_spacing = 30e3,
    ...                   num_tx=4,
    ...                   num_streams_per_tx=2,
    ...                   pilot_pattern = "kronecker",
    ...                   pilot_ofdm_symbol_indices = [2, 11])
    >>> rg.pilot_pattern.show();

    .. image:: ../figures/kronecker_pilot_pattern.png

    """

    def __init__(self,
                 num_ofdm_symbols,
                 num_effective_subcarriers,
                 pilot_ofdm_symbol_indices,
                 normalize=True,
                 seed=0,
                 step=2,
                 dtype=tf.complex64):
        num_tx = 1
        num_streams_per_tx = 1

        # Create a pilot mask
        mask = np.ones([num_tx,
                         num_streams_per_tx,
                         14,
                         1024])
        mask[0, :, np.array(pilot_ofdm_symbol_indices)+4, :num_effective_subcarriers] = 0


        mask[0, :, pilot_ofdm_symbol_indices, ::step] = 1
        num_pilot_symbols = int(np.sum(mask[0, 0]))
        # Define pilot sequences
        pilots = np.zeros([num_tx,
                           num_streams_per_tx,
                           num_pilot_symbols], np.complex64)
        qam_source = QAMSource(2, seed=seed, dtype=dtype)
        pilots[0, 0, :] = qam_source([1, 1, num_pilot_symbols])

        # Create a PilotPattern instance
        # pp = PilotPattern(mask, pilots)

        super().__init__(mask, pilots, trainable=False,
                         normalize=normalize, dtype=dtype)

class myPilotPattern(PilotPattern):
    """Simple orthogonal pilot pattern with Kronecker structure.

    This function generates an instance of :class:`~sionna.ofdm.PilotPattern`
    that allocates non-overlapping pilot sequences for all transmitters and
    streams on specified OFDM symbols. As the same pilot sequences are reused
    across those OFDM symbols, the resulting pilot pattern has a frequency-time
    Kronecker structure. This structure enables a very efficient implementation
    of the LMMSE channel estimator. Each pilot sequence is constructed from
    randomly drawn QPSK constellation points.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of a :class:`~sionna.ofdm.ResourceGrid`.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension.
        Defaults to `True`.

    seed : int
        Seed for the generation of the pilot sequence. Different seed values
        lead to different sequences. Defaults to 0.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Note
    ----
    It is required that the ``resource_grid``'s property
    ``num_effective_subcarriers`` is an
    integer multiple of ``num_tx * num_streams_per_tx``. This condition is
    required to ensure that all transmitters and streams get
    non-overlapping pilot sequences. For a large number of streams and/or
    transmitters, the pilot pattern becomes very sparse in the frequency
    domain.

    Examples
    --------
    >>> rg = ResourceGrid(num_ofdm_symbols=14,
    ...                   fft_size=64,
    ...                   subcarrier_spacing = 30e3,
    ...                   num_tx=4,
    ...                   num_streams_per_tx=2,
    ...                   pilot_pattern = "kronecker",
    ...                   pilot_ofdm_symbol_indices = [2, 11])
    >>> rg.pilot_pattern.show();

    .. image:: ../figures/kronecker_pilot_pattern.png

    """
    def __init__(self,
                 num_ofdm_symbols,
                 num_effective_subcarriers,
                 pilot_ofdm_symbol_indices,
                 normalize=True,
                 seed=0,
                 step=2,
                 dtype=tf.complex64,
                 s0_mode=False):
        num_tx = 1
        num_streams_per_tx = 1

        # Create a pilot mask
        mask = np.zeros([num_tx,
                         num_streams_per_tx,
                         num_ofdm_symbols,
                         num_effective_subcarriers])

        if s0_mode:
            mask[0, :, pilot_ofdm_symbol_indices, :step] = 1
        else:
            mask[0, :, pilot_ofdm_symbol_indices, ::step] = 1
        num_pilot_symbols = int(np.sum(mask[0, 0]))
        # Define pilot sequences
        pilots = np.zeros([num_tx,
                           num_streams_per_tx,
                           num_pilot_symbols], np.complex64)
        qam_source = QAMSource(2, seed=seed, dtype=dtype)
        pilots[0, 0, :] = qam_source([1, 1, num_pilot_symbols])

        # Create a PilotPattern instance
        #pp = PilotPattern(mask, pilots)
        super().__init__(mask=mask, pilots=pilots,
                         normalize=normalize)

    @property
    def mask(self):
        """Get the pilot mask."""
        return self._mask

    @mask.setter
    def mask(self, value):
        """Set the pilot mask."""
        self._mask = tf.constant(value, dtype=tf.int32)

class OFDMSystem(Model):  # Inherits from Keras Model

    def __init__(self, snrdb=(6.0, 6.0),
                 fft_size=None,
                 pilot_ofdm_symbol_indices=[],
                 pilots_step=1,
                 num_ofdm_symbols=8,
                 pcsi=False,
                 bps=1,
                 interleaving=False,
                 sionna=None,
                 domain='freq',
                 snrtoebno=False,
                 code_rate=0.5,
                 subcarrier_spacing=15e3,
                 cyclic_prefix_length=100,
                 batch=100,
                 ibo=0,
                 hpa_apply=False,
                 save_model=False,
                 siso=False,
                 print_details=False,
                 channel_type='TDL',
                 channel_mode='A',
                 delay_spread=100e-6,
                 carrier_frequency=3.5e9,
                 doppler_speed_min=0,
                 doppler_speed_max=0,
                 lmmse_channel_estimator=False,
                 estimate_covariance=False,):

        super().__init__()
        # init params
        self.save_name = f"5g{snrdb[0]}"
        self.snrtoebno = snrtoebno
        self.snrdb = snrdb
        self.domain = domain
        pilot_ofdm_symbol_indices = [int(x) for x in pilot_ofdm_symbol_indices if x.isdigit()]
        self.N = (num_ofdm_symbols  -
                  len(pilot_ofdm_symbol_indices)) * fft_size
        self.pcsi = pcsi
        self.BPS = bps
        self.cardinality_x = self.cardinality_s = 2
        self.cardinality_y = 2
        self.CODERATE_orig = code_rate

        # interleaving - bits reverse indices
        self.interleaving = interleaving
        if self.interleaving:
            m = self.N.bit_length() - 1  # number of bits needed to represent indices
            indices = np.arange(self.N)
            if sionna is None:
                self.reversed_indices = np.array([int(f'{i:0{m}b}'[::-1], 2) for i in indices])
            else:
                self.reversed_indices = indices
            #self.reversed_indices = indices
            #self.reversed_indices  = np.arange(self.N)
            self.inter = InterleavingMethod(self.N, True)

        cyclic_prefix_length, Tcp = self.compute_cyclic_prefix(subcarrier_spacing,
                                                               cyclic_prefix_length, fft_size, domain)
        pilot_pattern =  self.get_pilot_pattern(pilots_step,
                          pilot_ofdm_symbol_indices,
                          num_ofdm_symbols, fft_size)
        #subcarrier_spacing = subcarrier_spacing * 64 / fft_size
        self.rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                                       fft_size=fft_size,
                                       subcarrier_spacing=subcarrier_spacing,
                                       cyclic_prefix_length=cyclic_prefix_length,
                                       pilot_pattern=pilot_pattern,
                                       pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        self.E  = int(num_ofdm_symbols * fft_size - self.rg.num_pilot_symbols)
        self.CODERATE = self.N * code_rate / (self.E * self.BPS)
        self.rg.show()
        plt.savefig('resource_grid.png')  # Save the plot as an image file
        wandb.log({f"ofdm_frame": wandb.Image('resource_grid.png')})

        # The mapper maps blocks of information bits to constellation symbols
        if self.BPS == 1:
            bpsk_constellation = Constellation("custom", 1,
                                                          points=tf.convert_to_tensor([1, -1]))
            self.mapper = Mapper(constellation_type="custom",
                                            num_bits_per_symbol=self.BPS,
                                            constellation=bpsk_constellation)
        else:
            self.mapper =Mapper("qam", self.BPS)


        # OFDM modulator
        self.rg_mapper = ResourceGridMapper(self.rg)
        self.OFDMModulator = OFDMModulator(0)

        self.siso = siso
        # Create a channel model
        self.channel, self._modulator, self._demodulator = self.initialize_channel(
            channel_type=channel_type,
            channel_mode=channel_mode,
            delay_spread=delay_spread,
            carrier_frequency=carrier_frequency,
            doppler_speed_min=doppler_speed_min,
            doppler_speed_max=doppler_speed_max,
            domain=self.domain,
            rg=self.rg,
            siso=self.siso,
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size)

        # channel estimate
        # The LS channel estimator will provide channel estimates and error variances
        pilot_pattern2 = myPilotPattern(
            num_ofdm_symbols=num_ofdm_symbols,
            num_effective_subcarriers=fft_size,
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            step=2,
        )
        self.lmmse_channel_estimator = lmmse_channel_estimator
        if len(pilot_ofdm_symbol_indices) != 0:
            if lmmse_channel_estimator:
                self.estimate_covariance=estimate_covariance

                if self.estimate_covariance:
                    self.ls_est_orig = LSChannelEstimator(self.rg, interpolation_type="lin")
                    #self.ls_est_orig._pilot_pattern.mask = pilot_pattern2.mask
                    freq_cov_mat, time_cov_mat, SPACE_COV_MAT = self.cov_est(10,10)
                else:
                    ofdm_symbol_duration = self.rg.ofdm_symbol_duration
                    channel_mode_est = channel_mode
                    # channel_mode_est = "C"
                    FREQ_COV_MAT = tdl_freq_cov_mat(channel_mode_est, subcarrier_spacing, fft_size, delay_spread, precision=None)
                    TIME_COV_MAT = tdl_time_cov_mat(model=channel_mode_est, speed=doppler_speed_max, carrier_frequency=carrier_frequency,
                                                    ofdm_symbol_duration=ofdm_symbol_duration,
                                                    num_ofdm_symbols=num_ofdm_symbols, precision=None)
                    SPACE_COV_MAT = None
                    freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
                    time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
                lmmse_int_freq_first = LMMSEInterpolator(self.rg.pilot_pattern, time_cov_mat, freq_cov_mat,
                                                         order="f-t")
                self.ls_est = LSChannelEstimator(self.rg, interpolator=lmmse_int_freq_first)
            else:
                 self.ls_est = LSChannelEstimator(self.rg, interpolation_type="lin")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        sm = StreamManagement(np.array([[1]]), 1)
        self.lmmse_equ = LMMSEEqualizer(self.rg, sm)


        # The demapper produces LLR for all coded bits
        if self.BPS == 1:
            self.demapper = Demapper("app", "custom", self.BPS, constellation=bpsk_constellation)
        else:
            self.demapper = Demapper("app", "qam", self.BPS)

        num_ofdm_symbols_h = num_ofdm_symbols
        self.h_freq = self.add_weight(name="state", shape=(batch, 1, 1, 1, 1, num_ofdm_symbols_h, fft_size), dtype=tf.complex64, trainable=False,
                                     initializer=initializers.constant(0.0))

        if self.snrdb[0] == self.snrdb[1]  or save_model==False:
            self.no = self.add_weight(name="state", shape=(), dtype=dtype, trainable=False,
                                         initializer=initializers.constant(0.0))
            self.no = tf.Variable(initial_value=0.0, trainable=False, dtype=dtype)
        else:
            self.no = self.add_weight(name="state", shape=(batch,1), dtype=dtype, trainable=False,
                                         initializer=initializers.constant(0.0))
        #self.ls_est._pilot_pattern.mask = pilot_pattern2.mask

        self.shape_y = [batch, 1, 1, num_ofdm_symbols, fft_size]
        self.qam_source = QAMSource(2)
        self.channel_type = channel_type
        # hpa
        p, g, Vsat = 2, 1, 1
        self.HPA = HPA_Rapp(p, g, Vsat)
        self.hpa_apply = hpa_apply
        if self.hpa_apply:
            self.ibo_lin = self.HPA.InputBackOFF(ibo)  # Input Backoff
        else:
            self.ibo_lin = 1
        self._awgn = AWGN(dtype='complex64')
        if self.siso:
            if sionna:
                self.siso_eq = True
            else:
                self.siso_eq = False
        if print_details:
            print('OFDM System initialized with the following parameters:')
            print('OFDM symbols:', num_ofdm_symbols)
            print('FFT size:', fft_size)
            print('Subcarrier spacing:', subcarrier_spacing)
            print('CP [sec]:', Tcp)
            print('CP length:', self.rg.cyclic_prefix_length)
            print('T:', self.rg.ofdm_symbol_duration)
            print('Pilot number:', self.rg.num_pilot_symbols)
        self.ebno_db_per = 1000

    def myawgn(self, x, no):
        y = self.awgn(x, no)
        h_freq = tf.ones((y.shape[0],1,1,1,1,self.rg.num_ofdm_symbols,self.rg.fft_size), dtype=tf.complex64)
        return y, h_freq

    def initialize_channel(self, channel_type, channel_mode,
                           delay_spread,
                           carrier_frequency,
                           doppler_speed_min,
                           doppler_speed_max,
                           domain, rg, siso,
                           num_ofdm_symbols, fft_size):
        """
        Initialize the channel based on the configuration and domain.

        Args:
            cfg (dict): Configuration dictionary containing channel parameters.
            domain (str): Domain type ('freq' or 'time').
            rg (ResourceGrid): Resource grid instance.
            siso (bool): Flag indicating if the system is SISO.
            num_ofdm_symbols (int): Number of OFDM symbols.
            fft_size (int): FFT size.

        Returns:
            tuple: A tuple containing the initialized channel, modulator, and demodulator.
        """
        if channel_type == 'TDL' or  channel_type == 'CDL' :
            if channel_type == 'CDL':
                pass
            else:
                tdl = TDL(model=channel_mode,
                          delay_spread=delay_spread,
                          carrier_frequency=carrier_frequency,
                          min_speed=doppler_speed_min,
                          max_speed=doppler_speed_max)
                tdl2= TDL(model=channel_mode,
                          delay_spread=delay_spread,
                          carrier_frequency=carrier_frequency,
                          min_speed=doppler_speed_min,
                          max_speed=30)

            if domain == 'freq':
                # Frequency domain channel
                channel = OFDMChannel(tdl,
                                                 rg,
                                                 add_awgn=True,
                                                 normalize_channel=True,
                                                 return_channel=True)
                modulator = lambda x: x
                demodulator = lambda x: x

            elif domain == 'time':
                num_time_samples = rg.num_time_samples
                l_min = None
                l_max = None

                if siso:
                    num_time_samples = num_ofdm_symbols * fft_size
                #     l_min = 0
                # l_min = 0
                self.channel2 = TimeChannel(
                    tdl2,
                    rg.bandwidth,
                    num_time_samples,
                    l_min=l_min,
                    l_max=l_max,
                    add_awgn=True,
                    normalize_channel=True,
                    return_channel=True)
                channel = TimeChannel(
                    tdl,
                    rg.bandwidth,
                    num_time_samples,
                    l_min=l_min,
                    l_max=l_max,
                    add_awgn=True,
                    normalize_channel=True,
                    return_channel=True)
                modulator = OFDMModulator(rg.cyclic_prefix_length)
                demodulator = OFDMDemodulator(rg.fft_size, channel._l_min, rg.cyclic_prefix_length)
        elif channel_type == 'AWGN':
            self.awgn = AWGN(dtype='complex64')
            channel = self.myawgn
            modulator = lambda x: x
            demodulator = lambda x: x
            if domain == 'time':
                raise ValueError("AWGN channel only supports 'freq' domain.")
        return channel, modulator, demodulator

    def get_pilot_pattern(self, pilots_step,
                          pilot_ofdm_symbol_indices,
                          num_ofdm_symbols, fft_size):
        """
        Determine the pilot pattern and pilot OFDM symbol indices based on configuration.

        Returns:
            tuple: A tuple containing `pilot_pattern` and `pilot_ofdm_symbol_indices`.
        """
        if len(pilot_ofdm_symbol_indices) == 0:
            pilot_pattern = 'empty'
        else:
            pilot_pattern = myPilotPattern(
                num_ofdm_symbols=num_ofdm_symbols,
                num_effective_subcarriers=fft_size,
                pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
                step=pilots_step,
            )

        return pilot_pattern

    def compute_cyclic_prefix(self, subcarrier_spacing, cyclic_prefix_length, fft_size, domain):
        """
        Compute the cyclic prefix length based on subcarrier spacing and configuration.

        """
        # Determine Tcp based on subcarrier spacing
        if domain=='freq':
            return 0, 0

        if subcarrier_spacing == 15e3:
            Tcp = 4.69e-6  # [s]
        elif subcarrier_spacing == 30e3:
            Tcp = 2.34e-6
        elif subcarrier_spacing == 60e3:
            Tcp = 1.17e-6
        elif subcarrier_spacing == 120e3:
            Tcp = 0.57e-6
        else:
            Tcp = 0.29e-6

        # Compute cyclic prefix length
        if int(cyclic_prefix_length) == 100:
            cyclic_prefix_length = round(Tcp * subcarrier_spacing * fft_size)
        else:
            cyclic_prefix_length = int(cyclic_prefix_length)
        Tcp = (cyclic_prefix_length / fft_size) / subcarrier_spacing
        return cyclic_prefix_length, Tcp

    #@tf.function  # Graph execution to speed things up
    def llr(self, y):
        y_real_part = y[..., 0]
        y_imag_part = y[..., 1]
        no = self.no
        y_complex = tf.complex(y_real_part, y_imag_part)
        if self.siso:
            y = y_complex[:,None,None]
            llr = self.demapper([y, no])
        else:
            y = tf.reshape(y_complex, self.shape_y)

            if self.channel_type == 'AWGN':
                x_hat, no_eff = self.lmmse_equ(y, self.h_freq, 0., no)
            else:
                if self.pcsi:
                    h_hat, err_var = self.h_freq, 0.
                else:
                    h_hat, err_var = self.ls_est(y, no)
                x_hat, no_eff = self.lmmse_equ(y, h_hat, err_var, no)
            llr = self.demapper(x_hat, no_eff)
        llr = tf.squeeze(llr, axis=(1, 2))[:, :, tf.newaxis]
        if self.interleaving:
            llr = self.inter.remove_interleaving(llr)
            llr = tf.gather(llr, self.reversed_indices, axis=1)
        return llr

    def generate_ebno(self, ebno_db=None, batch=1):
        if ebno_db is None:
            if self.snrdb[0] == self.snrdb[1]:
                ebno_db = tf.random.uniform(shape=(), minval=self.snrdb[0], maxval=self.snrdb[1],
                                            dtype=tf.float32)
            else:
                ebno_db = tf.random.uniform(shape=(batch, 1), minval=self.snrdb[0], maxval=self.snrdb[1],
                                            dtype=tf.float32)
        else:
            ebno_db = tf.random.uniform(shape=(), minval=ebno_db, maxval=ebno_db, dtype=tf.float32)
        self.ebno_db = ebno_db
        if self.snrtoebno:
            no = ebnodb2no(ebno_db, num_bits_per_symbol=self.BPS, coderate=self.CODERATE, resource_grid=self.rg,)
        else:
            no = ebnodb2no(ebno_db,
                               num_bits_per_symbol=1,
                               coderate=1)
            self.rg.num_data_symbols
        self.no.assign(no)
        return no


    def create_interleaving_indices(self, length):
        indices = []
        for i in range(0, length, 4):
            indices.extend([i, i + 2, i + 1, i + 3])
        return indices

    #@tf.function  # Graph execution to speed things up
    def estimate_covariance_matrices(self, batch=100,num_it=100):
        self._binary_source = BinarySource()

        freq_cov_mat = tf.zeros([self.rg.fft_size, self.rg.fft_size], tf.complex64)
        time_cov_mat = tf.zeros([self.rg.num_ofdm_symbols, self.rg.num_ofdm_symbols], tf.complex64)
        space_cov_mat = tf.zeros([1, 1], tf.complex64)
        for _ in tqdm(tf.range(num_it)):
            ebno_db = tf.random.uniform(shape=(batch, 1), minval=5, maxval=10,
                                        dtype=tf.float32)
            no = ebnodb2no(ebno_db, num_bits_per_symbol=self.BPS, coderate=self.CODERATE, resource_grid=self.rg,)
            codewords = self._binary_source([batch, 1, 1, self.rg.num_data_symbols])
            x_mapper = self.mapper(codewords)
            x_rg = self.rg_mapper(x_mapper)
            x_ofdm = self._modulator(x_rg)
            y_complex, h_freq = self.channel(x_ofdm, 100)
            y_complex = self._demodulator(y_complex)
            h_hat = y_complex/ x_rg
            h_hat, err_var = self.ls_est_orig(y_complex, no)

            h_freq_prefect = time_to_ofdm_channel(h_freq, self.rg, self.channel._l_min)
            mse = tf.reduce_mean(tf.square(tf.abs(tf.squeeze(h_hat) - tf.squeeze(h_freq_prefect))))

            h_samples = h_hat[:, 0, :, 0, 0]

            #################################
            # Estimate frequency covariance
            #################################
            # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
            h_samples_ = tf.transpose(h_samples, [0, 1, 3, 2])
            # [batch size, num_rx_ant, fft_size, fft_size]
            freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
            # [fft_size, fft_size]
            freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1))
            # [fft_size, fft_size]
            freq_cov_mat += freq_cov_mat_

            ################################
            # Estimate time covariance
            ################################
            # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
            time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
            # [num_ofdm_symbols, num_ofdm_symbols]
            time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0, 1))
            # [num_ofdm_symbols, num_ofdm_symbols]
            time_cov_mat += time_cov_mat_

            ###############################
            # Estimate spatial covariance
            ###############################
            # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
            h_samples_ = tf.transpose(h_samples, [0, 2, 1, 3])
            # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
            space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
            # [num_rx_ant, num_rx_ant]
            space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0, 1))
            # [num_rx_ant, num_rx_ant]
            space_cov_mat += space_cov_mat_

        freq_cov_mat /= tf.complex(tf.cast(self.rg.num_ofdm_symbols * num_it, tf.float32), 0.0)
        time_cov_mat /= tf.complex(tf.cast(self.rg.fft_size * num_it, tf.float32), 0.0)
        space_cov_mat /= tf.complex(tf.cast(self.rg.fft_size * num_it, tf.float32), 0.0)
        return freq_cov_mat, time_cov_mat, space_cov_mat

    def cov_est(self, num_it=1000, batch_size=100, no=None):
        FFT_SIZE = self.rg.fft_size
        NUM_OFDM_SYMBOLS = self.rg.num_ofdm_symbols
        freq_cov_mat2 = tf.zeros([FFT_SIZE, FFT_SIZE], tf.complex64)
        time_cov_mat2 = tf.zeros([NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS], tf.complex64)
        freq_cov_mat = tf.zeros([FFT_SIZE, FFT_SIZE], tf.complex64)
        time_cov_mat = tf.zeros([NUM_OFDM_SYMBOLS, NUM_OFDM_SYMBOLS], tf.complex64)
        space_cov_mat = tf.zeros([1, 1], tf.complex64)
        for _ in tqdm(tf.range(num_it)):
            snr_db = tf.random.uniform(shape=(batch_size, 1), minval=-10, maxval=10,
                                            dtype=tf.float32)
            x = BinarySource()([batch_size, 1, 1, self.rg.num_data_symbols * self.BPS])
            x = self.mapper(x)
            #x = self.qam_source([batch_size, 1, 1, self.rg.num_data_symbols])
            x_rg = self.rg_mapper(x)
            x_rg = self._modulator(x_rg)
            ##################################
            # Channel
            ##################################
            if no is None:
                no = ebnodb2no(1,
                                           num_bits_per_symbol=1,
                                           coderate=1)
            # no = ebnodb2no(100,
            #                                num_bits_per_symbol=1,
            #                                coderate=1)
            # no = tf.pow(10.0, -snr_db/10.0)
            #CHANNEL_MODEL.set_topology(*topology)
            y_rg, h_time = self.channel(x_rg, no)
            y_rg = self._demodulator(y_rg)
            ###################################
            # Channel estimationgfgg
            ###################################
            h_freq = time_to_ofdm_channel(h_time, self.rg, self.channel._l_min)
            h_hat, var_eff  = self.ls_est_orig(y_rg,no)
            h_samples = h_hat[:, 0, :, 0, 0]
            #h_samples = y_rg[:,0]
            #h_samples = sample_channel(channel_sampler,batch_size)
            h_samples2 = h_freq[:, 0, :, 0, 0]

            #################################
            # Estimate frequency covariance
            #################################
            # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
            mse = tf.reduce_mean(tf.square(tf.abs(h_freq - h_hat)))
            h_samples_ = tf.transpose(h_samples, [0, 1, 3, 2])
            # [batch size, num_rx_ant, fft_size, fft_size]
            freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
            # [fft_size, fft_size]
            freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1))
            # [fft_size, fft_size]
            freq_cov_mat += freq_cov_mat_

            ### best ###
            # h_samples_2 = tf.transpose(h_samples2, [0, 1, 3, 2])
            # # [batch size, num_rx_ant, fft_size, fft_size]
            # freq_cov_mat_2 = tf.matmul(h_samples_2, h_samples_2, adjoint_b=True)
            # # [fft_size, fft_size]
            # freq_cov_mat_2 = tf.reduce_mean(freq_cov_mat_2, axis=(0, 1))
            # # [fft_size, fft_size]
            # freq_cov_mat2 += freq_cov_mat_2


            ################################
            # Estimate time covariance
            ################################
            # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
            time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
            # [num_ofdm_symbols, num_ofdm_symbols]
            time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0, 1))
            # [num_ofdm_symbols, num_ofdm_symbols]
            time_cov_mat += time_cov_mat_

            ### best ###
            # time_cov_mat_2 = tf.matmul(h_samples2, h_samples2, adjoint_b=True)
            # # [num_ofdm_symbols, num_ofdm_symbols]
            # time_cov_mat_2 = tf.reduce_mean(time_cov_mat_2, axis=(0, 1))
            # # [num_ofdm_symbols, num_ofdm_symbols]
            # time_cov_mat2 += time_cov_mat_2

            ###############################
            # Estimate spatial covariance
            ###############################
            # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
            h_samples_ = tf.transpose(h_samples, [0, 2, 1, 3])
            # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
            space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
            # [num_rx_ant, num_rx_ant]
            space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0, 1))
            # [num_rx_ant, num_rx_ant]
            space_cov_mat += space_cov_mat_
        # freq_cov_mat -= tf.dtypes.complex(no,no) * tf.eye(freq_cov_mat.shape[0], dtype=freq_cov_mat.dtype)
        # time_cov_mat -= tf.dtypes.complex(no,no) * tf.eye(time_cov_mat.shape[0], dtype=time_cov_mat.dtype)
        freq_cov_mat /= tf.complex(tf.cast(NUM_OFDM_SYMBOLS * num_it, tf.float32), 0.0)
        time_cov_mat /= tf.complex(tf.cast(FFT_SIZE * num_it, tf.float32), 0.0)
        space_cov_mat /= tf.complex(tf.cast(FFT_SIZE * num_it, tf.float32), 0.0)
        # freq_cov_mat2 /= tf.complex(tf.cast(NUM_OFDM_SYMBOLS * num_it, tf.float32), 0.0)
        # time_cov_mat2 /= tf.complex(tf.cast(FFT_SIZE * num_it, tf.float32), 0.0)

        # FREQ_COV_MAT = tdl_freq_cov_mat("A", 15e3, 128, 100e-9, precision=None)
        # np.mean(abs(FREQ_COV_MAT - freq_cov_mat))
        #freq_cov_mat =  freq_cov_mat - tf.dtypes.complex(no,tf.zeros_like(no)) * tf.eye(freq_cov_mat.shape[0], dtype=freq_cov_mat.dtype)

        # freq_cov_mat -= tf.cast(no, time_cov_mat.dtype) * tf.eye(freq_cov_mat.shape[0], dtype=freq_cov_mat.dtype)
        # time_cov_mat -= tf.cast(no, time_cov_mat.dtype) * tf.eye(time_cov_mat.shape[0], dtype=time_cov_mat.dtype)

        # abs(np.diag(freq_cov_mat) - np.diag(freq_cov_mat2))
        # abs(np.diag(time_cov_mat) - np.diag(time_cov_mat2))
        # np.diag(space_cov_mat) - np.diag(space_cov_mat2)

        # 2. Define a helper to subtract noise and clip the diagonal
        def denoise_matrix(mat, noise_val):
            # Subtract noise floor from the diagonal
            noise_identity = tf.complex(noise_val/2, 0.0) * tf.eye(mat.shape[0], dtype=mat.dtype)
            clean_mat = mat - noise_identity
            
            # Extract, clip to epsilon (1e-9), and re-insert the diagonal
            diag = tf.linalg.diag_part(clean_mat)
            diag_clipped = tf.complex(tf.maximum(tf.math.real(diag), 1e-9), 0.0)
            return tf.linalg.set_diag(clean_mat, diag_clipped)

        # 3. Apply to both matrices
        freq_cov_mat = denoise_matrix(freq_cov_mat, no)
        time_cov_mat = denoise_matrix(time_cov_mat, no)
        # freq_cov_mat = (1 - tf.eye(freq_cov_mat.shape[0], dtype=freq_cov_mat.dtype))*freq_cov_mat + tf.eye(freq_cov_mat.shape[0], dtype=freq_cov_mat.dtype)*freq_cov_mat2
        # time_cov_mat = (1 - tf.eye(time_cov_mat.shape[0], dtype=time_cov_mat.dtype))*time_cov_mat + tf.eye(time_cov_mat.shape[0], dtype=time_cov_mat.dtype)*time_cov_mat2

        return freq_cov_mat, time_cov_mat, space_cov_mat


    def sample_channel_outputs(self, codewords, ebno_db=None):
        no = self.generate_ebno(ebno_db, codewords.shape[0])
        if not(ebno_db is  None) and self.lmmse_channel_estimator and self.estimate_covariance:
            if ebno_db != self.ebno_db_per :
                self.ls_est_orig = LSChannelEstimator(self.rg, interpolation_type="lin")
                # self.ls_est_orig._pilot_pattern.mask = pilot_pattern2.mask
                freq_cov_mat, time_cov_mat, SPACE_COV_MAT = self.cov_est(no=no,batch_size=codewords.shape[0])
                lmmse_int_freq_first = LMMSEInterpolator(self.rg.pilot_pattern, time_cov_mat, freq_cov_mat,
                                                         order="f-t")
                self.ls_est = LSChannelEstimator(self.rg, interpolator=lmmse_int_freq_first)
                self.ebno_db_per = ebno_db
            else:
                self.ebno_db_per = ebno_db

        if self.interleaving:
            codewords = tf.gather(codewords, self.reversed_indices, axis=1)
            codewords = self.inter.apply_interleaving(codewords)

        codewords = tf.squeeze(codewords, axis=2)
        codewords = codewords[:, tf.newaxis, tf.newaxis, :]
        x_mapper = self.mapper(codewords)
        x_rg = self.rg_mapper(x_mapper)
        x_ofdm = self._modulator(x_rg)

        ### amplifer ###
        if self.hpa_apply:
            x_ofdm = x_ofdm  * self.ibo_lin
            x_ofdm, _ = self.HPA(x_ofdm[:,0])
            x_ofdm = x_ofdm[:, None]

        # Channel
        if self.siso:
            y_complex, h_freq = self.channel(x_mapper, no)
            if self.siso_eq:
                d_k_all = np.ones(x_mapper.shape, dtype=np.complex64)
                for i in range(y_complex.shape[0]):
                    r_k = y_complex[i, 0, 0]
                    h_c = h_freq[i, 0, 0, 0, 0, 0, :]
                    zf_eq = zeroForcing(100)  # initialize ZF equalizer (object) of length nTaps
                    zf_eq.design(h_c)  # Design ZF equalizer
                    optDelay = zf_eq.opt_delay  # get the optimum delay of the equalizer
                    d_k = zf_eq.equalize(r_k)
                    d_k = d_k[optDelay:optDelay+x_mapper.shape[-1]]
                    d_k_all[i] = d_k
                y_complex = tf.convert_to_tensor(d_k_all)
            else:
                y_complex = y_complex[...,abs(self.channel._l_min):y_complex.shape[-1] - self.channel._l_max]
        else:
            y_complex, h_freq = self.channel(x_ofdm, no)
            y_complex = self._demodulator(y_complex)
            y_complex = y_complex / self.ibo_lin
        if self.domain == 'time':
            h_freq = time_to_ofdm_channel(h_freq, self.rg, self.channel._l_min)
        self.h_freq.assign(h_freq)
        y_complex = tf.reshape(y_complex, tf.concat((y_complex.shape[:3], [-1]), axis=0))
        y_complex = tf.squeeze(y_complex)[..., tf.newaxis]
        y_real = tf.concat([tf.math.real(y_complex), tf.math.imag(y_complex)], axis=-1)
        return y_real
