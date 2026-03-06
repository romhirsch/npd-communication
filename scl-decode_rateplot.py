import pandas as pd
import tensorflow as tf
#from statsmodels.sandbox.distributions.sppatch import expect
from models.polar_models import SCNeuralListDecoder
from utils.utils import (gpu_init, parse_args, read_configs, choose_channel,
                         print_config, set_wandb)
import numpy as np
from sionna.phy.utils.plotting import PlotBER
import matplotlib.pyplot as plt
import wandb
from models.polar_models import system_model
from models.polar_models import SCDecoder, SCTrellisDecoder, SCListDecoder
import time

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

tf.keras.backend.set_floatx('float32')
pd.set_option("expand_frame_repr", False)
print(tf.__version__)
args = parse_args()
gpu_init()

config = read_configs(args)
print_config(config)
set_wandb(config)

#code_rates = np.arange(0.05, 0.7, 0.15)
channel = choose_channel(config)

PolarClass = SCListDecoder
code_rates = config['code_rates']

for code_rate in code_rates:

    code_rate = round(code_rate, 2)
    #run_gpu_for_one_minute()

    polar = PolarClass(channel=channel,
                       batch=config["batch"],
                       Ns=config["Ns"],
                       crc=config["crc"],
                       code_rate=code_rate,
                       list_num=config["list_num"],
                       mode=config["sionna"],
                       link_channel=config['link_channel'],
                       eyN0=config['eyN0'])


    model = system_model(polar,mc_length=config["mc_length"],
               code_rate=code_rate,
               batch=config["batch"],
               tol=config["tol"],
               Ns=config["Ns"],
               design_path=config['design_path'],
               design_load=config['design_load'],
               mc_design=config['mc_design'],
               design5G=config['design5G'],)



    ber_plot128 = PlotBER()
    ebno_db = np.arange(config['snr_low'], config['snr_high'], 0.5) # sim SNR range

    # the first argument must be a callable (function) that yields u and u_hat for batch_size and ebno
    ber_plot128.simulate(model, # the function have defined previously
                         ebno_dbs=ebno_db, # SNR to simulate
                         legend=config['group'], # legend string for plotting
                         max_mc_iter=config["mc_length"]//config["batch"], # run 100 Monte Carlo runs per SNR point
                         num_target_block_errors=500, # continue with next SNR point after 1000 bit errors
                         batch_size=config["batch"], # batch-size per Monte Carlo run
                         soft_estimates=False, # the model returns hard-estimates
                         early_stop=True, # stop simulation if no error has been detected at current SNR point
                         show_fig=False, # we show the figure after all results are simulated
                         add_bler=True, # in case BLER is also interesting
                         target_bler=1e-3, # target BLER for early stopping
                         forward_keyboard_interrupt=True); # should be True in a loop


    # and show the figure
    ber_plot128(ylim=(1e-4, 1)) # we set the ylim to 1e-5 as otherwise more extensive simulations would be required for accurate curves.
    ber = ber_plot128.ber[0]
    bler = ber_plot128.ber[1]
    snr = ber_plot128.snr[0]

    # Log (snr, ber) and (snr, bler) to wandb

    # Log (snr, ber) and (snr, bler) to wandb
    for i in range(len(snr)):
        wandb.log({f"snr": snr[i], f"ber_{code_rate}": ber[i], f"bler{code_rate}": bler[i]})
    try:
        wandb.log({"snr_rate": snr[np.where(bler <= 1e-3)[0][0]], "rate": code_rate})
        wandb.log({"snr_rate_ber": snr[np.where(ber <= 1e-3)[0][0]], "rate_ber": code_rate})
        print(f'\n\nCode rate: {code_rate}, SNR: {snr[np.where(bler <= 1e-3)[0][0]]}\n\n')
    except:
        #wandb.log({"snr_rate": 50, "rate": code_rate})
        break

# for key, value in res_dict.items():
#     print(f'Code rate: {key}, SNR: {value}')
#     wandb.log({"snr_rate": value, "rate": key})
plt.show()
