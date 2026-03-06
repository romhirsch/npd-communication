import pandas as pd
import tensorflow as tf
from models.channel_models import ChannelMemory, Channel
from models.channel_model_3gpp import OFDMSystem
from models.polar_models import SCDecoder, SCTrellisDecoder
from utils.utils import gpu_init, parse_args, read_configs, choose_channel, print_config, set_wandb
tf.keras.backend.set_floatx('float32')
pd.set_option("expand_frame_repr", False)

args = parse_args()
gpu_init()

config = read_configs(args)
print_config(config)
set_wandb(config)

channel = choose_channel(config)

PolarClass = SCTrellisDecoder


polar = PolarClass(channel=channel,
                   batch=config["batch"])

polar.eval(mc_length=config["mc_length"],
           code_rate=config["code_rate"],
           batch=config["batch"],
           tol=config["tol"],
           Ns=config["Ns"],
           design_path=config['design_path'],
           design_load=config['design_load'],
           mc_design=config['mc_design'],)

