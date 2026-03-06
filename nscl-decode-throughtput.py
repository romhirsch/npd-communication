import pandas as pd
import tensorflow as tf

from models.polar_models import SCNeuralListDecoder
from utils.utils import (gpu_init, parse_args, read_configs, choose_channel,
                         print_config, set_wandb)
import numpy as np
from sionna.phy.utils.plotting import PlotBER
import matplotlib.pyplot as plt
import wandb
from models.polar_models import system_model
tf.keras.backend.set_floatx('float32')
pd.set_option("expand_frame_repr", False)
print(tf.__version__)
args = parse_args()
gpu_init()
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
    config['subcarrier_num']  = REs_per_frame // 8

    #run_gpu_for_one_minute()
    channel = choose_channel(config)
    config['Ns'] = [closest_power_of_2_exponent(channel[0].E)]
    code_rate = k/2**config['Ns'][0]
    config["code_rate"] = code_rate
    channel[0].CODERATE = code_rate
    channel[1].CODERATE = code_rate
    polar = SCNeuralListDecoder(channel=channel,
                                list_num=config['list_num'],
                                crc=config['crc'],
                                crc_oracle=config['crc_oracle'],
                                embedding_size=config['embedding_size_polar'],  # 8
                                hidden_size=config['hidden_polar'],
                                layers_per_op=config['layers_per_op'],
                                activation=config['activation'],
                                batch=config["batch"],
                                trained_block_norm=int(config["Ns"][0]+1),
                                eyN0=config['eyN0'],)



    model = system_model(polar,mc_length=config["mc_length"],
               code_rate=code_rate,
               batch=config["batch"],
               tol=config["tol"],
               Ns=config["Ns"],
               design_path=config['design_path'],
               design_load=config['design_load'],
               mc_design=config['mc_design'],
                load_nsc_path=config["nsc_path"],
                         )



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
                         target_bler=1e-4, # target BLER for early stopping
                         forward_keyboard_interrupt=True); # should be True in a loop


    # and show the figure
    ber_plot128(ylim=(1e-4, 1))  #
    ber = ber_plot128.ber[0]
    bler = ber_plot128.ber[1]
    snr = ber_plot128.snr[0]

    # Log (snr, ber) and (snr, bler) to wandb

    for i in range(len(snr)):
        wandb.log({f"snr": snr[i], f"ber_{REs_per_frame}": ber[i], f"bler_{REs_per_frame}": bler[i]})
    T = (1)/ config['subcarrier_spacing'] * config["num_ofdm_symbols"] # Symbol duration in seconds
    thoughtput = (1 - bler) * (k / T)  # Throughput in bits per second
    thoughtputber = (1 - ber)* (k / T)

    # Select rows from 0 to k
    # df = pd.DataFrame(polar.sorted_arg_errors)
    # subset_df = df.iloc[:k + 1]
    # value_counts = subset_df.values.flatten()
    # counts = pd.Series(value_counts).value_counts()
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
# for key, value in res_dict.items():
#     print(f'Code rate: {key}, SNR: {value}')
#     wandb.log({"snr_rate": value, "rate": key})
plt.show()
