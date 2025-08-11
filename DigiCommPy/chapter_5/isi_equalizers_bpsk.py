"""
Script: DigiCommPy.chapter_5.isi_equalizers_bpsk.py
Demonstration of Eb/N0 Vs SER for baseband BPSK modulation scheme over different ISI channels with MMSE and ZF equalizers

@author: Mathuranathan Viswanathan
Created on Aug 29, 2019
"""
import numpy as np
import matplotlib.pyplot as plt #for plotting functions
import sys
import tensorflow as tf
sys.path.append('/gpfs0/bgu-haimp/users/romhi/PycharmProjects/data-driven-polar-codes/DigiCommPy')
from modem import PSKModem #import PSKModem
from channels import awgn
from equalizers import zeroForcing, MMSEEQ #import MMSE equalizer class
from errorRates import ser_awgn #for theoretical BERs
from scipy.signal import freqz
from numpy.random import randn

#---------Input Fields------------------------
N=1000 # Number of bits to transmit
EbN0dBs = np.arange(start=0,stop=10,step=2) #  Eb/N0 range in dB for simulation
M=2 # 2-PSK
h_c = [-0.02389316-0.01777963j,  0.02836074+0.02116175j,
       -0.03489243-0.02613989j,  0.04535964+0.03420187j,
       -0.06490824-0.04953889j,  0.11496145+0.09047072j,
       -0.66972864-0.6519444j , -0.07946528-0.16111273j,
        0.10378019+0.0713807j , -0.05966787-0.0423784j ,
        0.04257138+0.03050578j, -0.03318014-0.02389409j,
        0.02720626+0.01965505j, -0.02306324-0.01669958j,
        0.02001859+0.01451935j, -0.01768569-0.01284398j,
        0.01584061+0.01151587j, -0.01434467-0.0104371j ,
        0.01310716+0.00954331j]
nTaps = 100 # Desired number of taps for equalizer filter
SER_zf = np.zeros(len(EbN0dBs)); SER_mmse = np.zeros(len(EbN0dBs))
#-----------------Transmitter---------------------
inputSymbols=np.random.randint(low=0,high=2,size=N) #uniform random symbols 0s & 1s
modem = PSKModem(M)
def bpsk_modulation(input_vector):
    # Map 0 to 1+0j and 1 to -1+0j
    return np.array([1+0j if bit == 0 else -1+0j for bit in input_vector])
modulatedSyms = bpsk_modulation(inputSymbols)
#modulatedSyms = modem.modulate(inputSymbols)
import sionna as sn
from sionna.channel.tr38901 import TDL
from sionna.channel import AWGN, RayleighBlockFading, OFDMChannel, TimeChannel, time_lag_discrete_time_channel

tdl = TDL(model='C',
          delay_spread=300e-9,
          carrier_frequency=3.5e9,
          min_speed=0,
          max_speed=0)

channel = TimeChannel(
    tdl,
    15e3*128,
    N,
    add_awgn=True,
    normalize_channel=True,
    return_channel=True)
# sn.utils.ebnodb2no(ebno_db,
#                   num_bits_per_symbol=1,
#                   coderate=1)
s = tf.convert_to_tensor(modulatedSyms[None,None,None,], dtype=tf.complex64)
#h_c = h_c[::-1]
# 1x_2 = np.convolve(s[0,0,0],h_c,'same') # apply channel effect on transmitted symbols
# h_c=randn(19)+1j*randn(19) # random complex system

# x = np.convolve(s[0,0,0],h_c) # apply channel effect on transmitted symbols
from tqdm import tqdm
for i,EbN0dB in enumerate(EbN0dBs):
    for j in tqdm(range(1000)):
        #receivedSyms = awgn(x,EbN0dB) #add awgn noise
        no = sn.utils.ebnodb2no(EbN0dB,
                      num_bits_per_symbol=1,
                      coderate=1)
        #x, h_c = channel((s,0))
        x, h_c = channel((s,no))
        #x, h_c = channel(s)

        h_c = h_c[0, 0, 0, 0, 0, 0]
        x = x[0, 0, 0]
        #x2 = np.convolve(s[0, 0, 0], h_c)  # apply channel effect on transmitted symbols
        # Omega, H_c = freqz(h_c)  # frequency response of the channel
        # fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)
        # ax2.stem(abs(h_c), use_line_collection=True)  # time domain
        # ax3.plot(Omega, 20 * np.log10(abs(H_c) / max(abs(H_c))));
        # fig2.show()
        #receivedSyms = awgn(x,EbN0dB) #add awgn noise

        #from DigiCommPy.channels import awgn
        receivedSyms = x #awgn(x, 5)  + AWGN(no)  # add noise

        # DELAY OPTIMIZED MMSE equalizer
        mmse_eq = MMSEEQ(nTaps) #initialize MMSE equalizer (object) of length nTaps
        mmse_eq.design(np.array(h_c),EbN0dB) #Design MMSE equalizer
        optDelay = mmse_eq.opt_delay #get the optimum delay of the equalizer
        #filter received symbols through the designed equalizer
        equalizedSamples = mmse_eq.equalize(np.array(receivedSyms))
        y_mmse = equalizedSamples[optDelay:optDelay+N] # samples from optDelay position

        # from equalizers import LMSEQ
        #
        # lms_eq = LMSEQ(19)  # initialize the LMS filter object
        # lms_eq.design(0.01, np.array(s[0,0,0,:1000]), np.array(x[:1000]))  #
        #
        # equalizedSamples = lms_eq.equalize(np.array(receivedSyms))
        # y_mmse = equalizedSamples[abs(channel._l_min):N + abs(channel._l_min)] # samples from optDelay position

        # DELAY OPTIMIZED ZF equalizer
        zf_eq = zeroForcing(nTaps) #initialize ZF equalizer (object) of length nTaps
        zf_eq.design(h_c) #Design ZF equalizer
        optDelay = zf_eq.opt_delay #get the optimum delay of the equalizer
        #filter received symbols through the designed equalizer
        equalizedSamples = zf_eq.equalize(receivedSyms)
        y_zf = equalizedSamples[optDelay:optDelay+N] # samples from optDelay position

        # Optimum Detection in the receiver - Euclidean distance Method
        estimatedSyms_mmse = modem.demodulate(y_mmse)
        estimatedSyms_zf = modem.demodulate(y_zf)
        # SER when filtered thro MMSE eq.
        SER_mmse[i] +=sum((inputSymbols != estimatedSyms_mmse))/N
        # SER when filtered thro ZF eq.
        SER_zf[i] +=sum((inputSymbols != estimatedSyms_zf))/N
    SER_mmse[i] /= 1000
    # SER when filtered thro ZF eq.
    SER_zf[i] /= 1000


fig1, ax1 = plt.subplots(nrows=1,ncols = 1)
ax1.semilogy(EbN0dBs,SER_zf,'g',label='ZF Equalizer');
#ax1.semilogy(EbN0dBs,SER_mmse,'r',label='MMSE equalizer')
#ax1.semilogy(EbN0dBs,SER_theory,'k',label='No interference')
ax1.set_title('Probability of Symbol Error for BPSK signals');
ax1.set_xlabel('$E_b/N_0$(dB)');ax1.set_ylabel('Probability of Symbol Error-$P_s$')
ax1.legend(); ax1.set_ylim(bottom=10**-4, top=1);fig1.show()

# compute and plot channel characteristics
Omega, H_c  = freqz(h_c) #frequency response of the channel
fig2, (ax2,ax3) = plt.subplots(nrows=1,ncols = 2)
ax2.stem(abs(h_c),use_line_collection=True) # time domain
ax3.plot(Omega,20*np.log10(abs(H_c)/max(abs(H_c))));fig2.show()