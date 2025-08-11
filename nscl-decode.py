import pandas as pd
import tensorflow as tf
from models.polar_models import SCNeuralListDecoder
from utils.utils import (gpu_init, parse_args, read_configs, choose_channel,
                         print_config, set_wandb)

tf.keras.backend.set_floatx('float32')
pd.set_option("expand_frame_repr", False)
print(tf.__version__)
args = parse_args()
gpu_init()

config = read_configs(args)
print_config(config)
set_wandb(config)

channel = choose_channel(config)

polar = SCNeuralListDecoder(channel=channel,
                            list_num=config['list_num'],
                            crc=config['crc'],
                            crc_oracle=config['crc_oracle'],
                            embedding_size=config['embedding_size_polar'],  # 8
                            hidden_size=config['hidden_polar'],
                            layers_per_op=config['layers_per_op'],
                            activation=config['activation'],
                            batch=config["batch"],
                            trained_block_norm=int(config["Ns"][0]+1))

polar.eval(mc_length=config["mc_length"],
           code_rate=config["code_rate"],
           batch=config["batch"],
           tol=config["tol"],
           Ns=config["Ns"],
           load_nsc_path=config["nsc_path"],
           design_path=config['design_path'],
           design_load=config['design_load'],
           mc_design=config['mc_design'],)


