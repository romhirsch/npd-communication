import tensorflow as tf
import pandas as pd
from models.polar_models import NeuralPolarDecoder
from utils.utils import gpu_init, choose_channel, print_config, parse_args, read_configs, set_wandb
tf.keras.backend.set_floatx('float32')
tf.get_logger().setLevel('ERROR')
pd.set_option("expand_frame_repr", False)
print(tf.__version__)
args = parse_args()
gpu_init()

config = read_configs(args)
print_config(config)
set_wandb(config)
#tf.config.run_functions_eagerly(True)

channel = choose_channel(config, train=True)

polar = NeuralPolarDecoder(channel=channel,
                           embedding_size=config['embedding_size_polar'],  # 8
                           hidden_size=config['hidden_polar'],  # 100
                           layers_per_op=config['layers_per_op'],  # 2
                           input_distribution=config['input_distribution'],  # sct
                           lr=config['lr'],  # 1e-3
                           optimizer=config['optimizer'],  # adam
                           batch=config['batch'],
                           pred_decay=config['pred_decay'],
                           lr_decay=config['lr_decay'],
                           trainEyOnly=config['trainEyOnly'],
                           eyN0=config['eyN0'],
                           layers_per_op_emb2llr=config['layers_per_op_emb2llr'],
                           emb2llr_snr=config['emb2llr_snr'],
                           llr_clip=config['llr_clip'])

if config['optimize_inputs']:
    polar.optimize(train_block_length=config['train_block_length'],
                   train_batch=config['batch'],
                   num_iters=config['num_iters'],
                   save_model=config['save_model'],
                   load_nsc_path=config['nsc_path'],
                   logging_freq=config['logging_freq'],
                   saving_freq=config['saving_freq'])
else:
    polar.train(train_block_length=config['train_block_length'],    # 1024
                train_batch=config['batch'],                        # 10
                num_iters=config['num_iters'],                      # 10^6
                save_model=args.save_model,                         # --save_model to activate
                train_ex=True,                                     # train the embedding ex?
                load_nsc_path=config['nsc_path'],                   # nsc_path from wandb
                logging_freq=config['logging_freq'],                # 100
                saving_freq=config['saving_freq'],
                logging_llr_hist=config["logging_llr_hist"],
                full_bu_depth=config['full_bu_depth'],
                resume_train=args.resume_train,
                train_mode=config['train_mode'])

