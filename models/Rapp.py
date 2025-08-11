import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Layer


class HPA_Rapp(Layer):
    def __init__(self, p, g, Vsat, **kwargs):
        super(HPA_Rapp, self).__init__(**kwargs)
        self.p = p
        self.g = g
        self.Vsat = Vsat

    def alpha_calc_complex(self, x, gain):  # ORIGINAL GOOD
        numi = tf.multiply(tf.abs(x) ** 2, gain)
        numi = tf.reduce_mean(numi, axis=2)
        denumi = tf.reduce_mean(tf.abs(x) ** 2, axis=2)
        alpha = numi / denumi
        return alpha

    def InputBackOFF(self, IBO = 3):
        # Back-off
        # input backoff set point, dB
        ipPwrAtPkOut = 10 * np.log10(abs(self.Vsat) ** 2) + 30
        PxdBm = 30 #(10 * torch.log10(torch.mean(abs(X_t) ** 2, axis=2)) + 30).squeeze()
        # dB power gain #  ipPwrAtPkOut- sat dbm, PxdBm - signal power, IBO - back in dbm
        gaindB = ipPwrAtPkOut - PxdBm - IBO
        BO_lin_add = 10 ** (gaindB / 20)  # linear voltage gain
        return BO_lin_add

    def call(self, sig):
        """
        High Power Amplifier - Rapp Model
        g - is the amplitude gain of the amplifier
        p - is the smoothness factor
        Vsat - is the output saturation level
        sig - is the input signal
        """
        p = self.p
        g = self.g
        Vsat = self.Vsat

        Ain = tf.norm(sig, axis=1, keepdims=True)  # amplitude of the signal
        norm_sig = tf.math.l2_normalize(sig, axis=1)  # normalize signal

        Aout = g * Ain * (1 + (g * Ain / Vsat) ** (2 * p)) ** (-1 / (2 * p))  # calculate Aout
        amp_fac = tf.identity(Ain)

        nonzero_mask = tf.not_equal(Ain, 0)
        amp_fac = tf.where(nonzero_mask, Aout / Ain, Ain)

        #alpha = self.alpha_calc_complex(sig, amp_fac)
        alpha = 0
        return Aout * norm_sig, alpha



# Function to plot Pin vs Pout
def plot_pin_vs_pout(model, pin_values):
    pin_values = np.array(pin_values, dtype=np.float32)
    pout_values = []

    for pin in pin_values:
        pout, _ = model(pin_values)
        pout_values.append(tf.norm(pout).numpy())

    plt.figure()
    plt.plot(pin_values[0,0], pout[0,0], label='Rapp Model')
    plt.xlabel("Input Power (Pin)")
    plt.ylabel("Output Power (Pout)")
    plt.title("Pin vs Pout Curve")
    plt.legend()
    plt.grid()
    plt.show()


# Example Usage
if __name__ == "__main__":
    p, g, Vsat = 15, 1, 1  # Example parameters
    model = HPA_Rapp(p, g, Vsat)
    pin_values = np.reshape(np.linspace(0, 2, 100),(1, 1, -1))  # Generate input power values
    plot_pin_vs_pout(model, pin_values)