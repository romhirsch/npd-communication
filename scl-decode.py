import pandas as pd
import tensorflow as tf
from utils.utils import gpu_init
from models.polar_models import SCListDecoder
from utils.utils import parse_args, read_configs, choose_channel, print_config, set_wandb
tf.keras.backend.set_floatx('float32')
pd.set_option("expand_frame_repr", False)

args = parse_args()
gpu_init()

config = read_configs(args)
print_config(config)
set_wandb(config)

channel = choose_channel(config)
PolarClass = SCListDecoder


polar = PolarClass(channel=channel,
                   list_num=config['list_num'],
                   crc=config['crc'],
                   batch=config["batch"],
                   sn=True)

polar.eval(mc_length=config["mc_length"],
           code_rate=config["code_rate"],
           batch=config["batch"],
           tol=config["tol"],
           Ns=config["Ns"],
           design_path=config['design_path'],
           design_load=config['design_load'],
           mc_design=config['mc_design'],)

