import pandas as pd
import tensorflow as tf
import wandb
import numpy as np
import numpy as np
from sionna.phy.utils.plotting import PlotBER
import matplotlib.pyplot as plt
from models.channel_models import ChannelMemory, Channel
from models.channel_model_3gpp import OFDMSystem
from models.polar_models import PolarSCL5GDecoder, SCTrellisDecoder
from utils.utils import gpu_init, parse_args, read_configs, choose_channel, print_config, set_wandb
from models.polar_models import system_model
tf.keras.backend.set_floatx('float32')
pd.set_option("expand_frame_repr", False)
import time
import math


def closest_power_of_2_exponent(n):
    """
    Returns the exponent of the closest power of 2 for the given input `n`.
    For example, for input 96, it returns 7 because 2**7 is closest to 96.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")

    # Calculate the closest exponent
    exponent = math.ceil(math.log2(n))
    return exponent

def run_gpu_for_one_minute():
    # בדיקה אם יש GPU זמין
    gpus = tf.config.list_physical_devices('GPU')
    @tf.function
    def heavy_gpu_task():
        x = tf.random.normal([1024, 1024])
        for _ in range(100):
            x = tf.matmul(x, x)
        return x

    # מריץ את החישוב במשך דקה
    start = time.time()
    while time.time() - start < 120:
        _ = heavy_gpu_task()


args = parse_args()
gpu_init()

config = read_configs(args)
print_config(config)
set_wandb(config)


#code_rates = np.arange(0.05, 0.7, 0.15)
# if (config['subcarrier_num'] * 8)  == 1024:
#     code_rates = np.arange(0.05, 0.7, 0.05)
# else:
#     code_rates = ((np.arange(0.05, 0.7, 0.05) * 1024) /
#                   (2**config['Ns'][0]))
#     code_rates = code_rates[code_rates <= 0.91]

k = config['k']
#code_rates = np.arange(0.05, 0.7, 0.05)
REs_per_frames = np.arange(128+64, 512+64, 64)
REs_per_frames = np.array([256,384,512])
for REs_per_frame in REs_per_frames:
    config['subcarrier_num']  = REs_per_frame // config['num_ofdm_symbols']
    #run_gpu_for_one_minute()
    channel = choose_channel(config)
    config['Ns'] = [closest_power_of_2_exponent(channel[0].E)]
    PolarClass = PolarSCL5GDecoder
    code_rate = k/2**config['Ns'][0]
    polar = PolarClass(channel=channel,
                       batch=config["batch"],
                       Ns=config["Ns"],
                       crc=config["crc"],
                       code_rate=code_rate,
                       list_num=config["list_num"],
                       mode=config["sionna"],
                       link_channel=config['link_channel'])


    model = system_model(polar,
               mc_length=config["mc_length"],
               code_rate=code_rate,
               batch=config["batch"],
               tol=config["tol"],
               Ns=config["Ns"],
               design_path=config['design_path'],
               design_load=config['design_load'],
               mc_design=config['mc_design'])



    ber_plot128 = PlotBER()
    ebno_db = np.arange(config['snr_low'], config['snr_high'], 0.5) # sim SNR range


    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot128.simulate(model, # the function have defined previously
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=config['group'], # legend string for plotting
                         max_mc_iter=config["mc_length"]//config["batch"], # run 100 Monte Carlo runs per SNR point
                         num_target_block_errors=50, # continue with next SNR point after 1000 bit errors
                         batch_size=config["batch"], # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         target_bler=5e-4, # target BLER for early stopping
                         forward_keyboard_interrupt=True); # should be True in a loop

    # and show the figure
    ber_plot128(ylim=(1e-4, 1)) # we set the ylim to 1e-5 as otherwise more extensive simulations would be required for accurate curves.
    ber = ber_plot128.ber[0]
    bler = ber_plot128.ber[1]
    snr = ber_plot128.snr[0]

    # Log (snr, ber) and (snr, bler) to wandb

    # Log (snr, ber) and (snr, bler) to wandb
    for i in range(len(snr)):
        wandb.log({f"snr": snr[i], f"ber_{REs_per_frame}": ber[i], f"bler_{REs_per_frame}": bler[i]})
    #T = (1 + channel.rg.cyclic_prefix_length/config['subcarrier_num'])/ config['subcarrier_spacing']  # Symbol duration in seconds
    T = (1 / config['subcarrier_spacing'] + 4.69e-6) * config["num_ofdm_symbols"]
    print('T:', T)
    thoughtput = (1 - bler) * (k / T)  # Throughput in bits per second
    thoughtputber = (1 - ber) * (k / T)
    for i in range(len(snr)):
        wandb.log({f"throughput_{REs_per_frame}": thoughtput[i], f"snr": snr[i]})
        wandb.log({f"throughputber_{REs_per_frame}": thoughtputber[i], f"snr": snr[i]})

        wandb.log({f"bler_{REs_per_frame}": bler[i], f"snr": snr[i]})
        wandb.log({f"ber_{REs_per_frame}": ber[i], f"snr": snr[i]})

    try:
        wandb.log({"snr_bler": snr[np.where(bler <= 1e-3)[0][0]], "REs": REs_per_frame})
        wandb.log({"snr_ber": snr[np.where(ber <= 1e-3)[0][0]], "REs": REs_per_frame})
        print(f'\n\nCode rate: {code_rate}, SNR: {snr[np.where(bler <= 1e-3)[0][0]]}\n\n')
    except:
        wandb.log({"snr_bler": 40, "REs": REs_per_frame})
        wandb.log({"snr_ber": 40, "REs": REs_per_frame})
        #break

# for key, value in res_dict.items():
#     print(f'Code rate: {key}, SNR: {value}')
#     wandb.log({"snr_rate": value, "rate": key})
plt.show()
