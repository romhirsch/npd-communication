import tensorflow as tf
from models.polar_models import SCNeuralDecoder
from utils.utils import (parse_args, read_configs, choose_channel,
                         print_config, set_wandb, gpu_init)
tf.keras.backend.set_floatx('float32')

args = parse_args()
gpu_init()

config = read_configs(args)
print_config(config)
set_wandb(config)

channel = choose_channel(config)

polar = SCNeuralDecoder(channel=channel,
                        embedding_size=config["embedding_size_polar"],
                        hidden_size=config["hidden_polar"],
                        layers_per_op=config["layers_per_op"],
                        batch=config["batch"],
                        trained_block_norm=int(config["Ns"][0]+1),)

polar.eval(mc_length=config["mc_length"],
           code_rate=config["code_rate"],
           batch=config["batch"],
           tol=config["tol"],
           Ns=config["Ns"],
           load_nsc_path=config["nsc_path"],
           design_path=config['design_path'],
           design_load=config['design_load'],
           mc_design=config['mc_design'],)

