import os
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from models.channel_models import AWGN, BSC, BEC, Z, Ising, Trapdoor, GE, ISI, MovingAverageAGN
from models.channel_model_3gpp import OFDMSystem
from models.channel_single_carrier import SingleCarrier
from models.channel_GE import GilbertElliottChannel
import math


def is_log_power_of_two(N):
    """
    Check if log2(N) is a power of 2.

    Args:
        N (int): The input number.

    Returns:
        bool: True if log2(N) is a power of 2, False otherwise.
    """
    if N <= 0:
        return False  # log2 is undefined for non-positive numbers

    log_value = math.log2(N)
    if log_value % 1 == 0:
        return True
    else:
        return False


def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--project', default=None, type=str, help='experiment name')
    argparser.add_argument('--entity', default=None, type=str, help='experiment name')
    argparser.add_argument('--group', default=None, type=str, help='experiment name')
    argparser.add_argument('--nsc_path', default=None, type=str, help='experiment name')
    argparser.add_argument('--design_path', default=None, type=str, help='experiment name')
    argparser.add_argument('--Ns', default=None, type=str, help='experiment name')
    argparser.add_argument('--channel_name', default=None, type=str, help='experiment name')
    argparser.add_argument('--snr', default=None, type=float, help='experiment name')
    argparser.add_argument('--snr_low', default=None, type=float, help='experiment name')
    argparser.add_argument('--snr_high', default=None, type=float, help='experiment name')
    argparser.add_argument('--isi_length', default=None, type=int, help='experiment name')
    argparser.add_argument('--mc_length', default=None, type=int, help='experiment name')
    argparser.add_argument('--tol', default=None, type=int, help='experiment name')
    argparser.add_argument('--batch', default=None, type=int, help='experiment name')
    argparser.add_argument('--code_rate', default=None, type=float, help='experiment name')
    argparser.add_argument('--activation', default=None, type=str, help='experiment name')
    argparser.add_argument('--num_iters', default=None, type=int, help='experiment name')
    argparser.add_argument('--saving_freq', default=None, type=int, help='experiment name')
    argparser.add_argument('--logging_freq', default=None, type=int, help='experiment name')
    argparser.add_argument('--layers_per_op', default=None, type=int, help='experiment name')
    argparser.add_argument('--lr', default=None, type=float, help='experiment name')
    argparser.add_argument('--optimizer', default=None, type=str, help='experiment name')
    argparser.add_argument('--train_block_length', default=None, type=int, help='experiment name')
    argparser.add_argument('--input_state_size', default=None, type=int, help='experiment name')
    argparser.add_argument('--input_distribution', default=None, type=str, help='experiment name')
    argparser.add_argument('--wandb_mode', default=None, type=str, help='experiment name')
    argparser.add_argument('--embedding_size_polar', default=None, type=int, help='experiment name')
    argparser.add_argument('--hidden_polar', default=None, type=int, help='experiment name')
    argparser.add_argument('--list_num', default=None, type=int, help='experiment name')
    argparser.add_argument('--crc', default=None, type=str, help='experiment name')
    argparser.add_argument('--save_model', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--optimize_inputs', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--crc_oracle', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--NUM_BITS_PER_SYMBOL', default=None, type=int, help='experiment name')
    argparser.add_argument('--num_ofdm_symbols', default=None, type=int, help='experiment name')
    argparser.add_argument('--subcarrier_num', default=None, type=int, help='experiment name')
    argparser.add_argument('--num_pilots', default=None, type=int, help='experiment name')
    argparser.add_argument('--subcarrier_spacing', default=None, type=float, help='experiment name')
    argparser.add_argument('--channel_type', default=None, type=str, help='experiment name')
    argparser.add_argument('--channel_mode', default=None, type=str, help='experiment name')
    argparser.add_argument('--delay_spread', default=None, type=float, help='experiment name')
    argparser.add_argument('--carrier_frequency', default=None, type=float, help='experiment name')
    argparser.add_argument('--perfect_csi', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--No_pilots', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--design_load', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--sionna', default=None, type=str, help='experiment name')
    argparser.add_argument('--interleaving', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--mc_design', default=None, type=int, help='experiment name')
    argparser.add_argument('--logging_llr_hist', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--pred_decay', default=None, type=float, help='experiment name')
    argparser.add_argument('--full_bu_depth', default=None, type=float, help='experiment name')
    argparser.add_argument('--lr_decay', default=None, type=float, help='experiment name')
    argparser.add_argument('--cyclic_prefix_length', default=None, type=float, help='experiment name')
    argparser.add_argument('--domain_5g', default=None, type=str, help='experiment name')
    argparser.add_argument('--link_channel', default=None, type=str, help='experiment name')
    argparser.add_argument('--doppler_speed_min', default=None, type=float, help='experiment name')
    argparser.add_argument('--doppler_speed_max', default=None, type=float, help='experiment name')
    argparser.add_argument('--trainEyOnly', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--snrtoebno', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--pilots_step', default=None, type=int, help='experiment name')
    argparser.add_argument('--apply_hpa', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--ibo', default=None, type=int, help='experiment name')
    argparser.add_argument('--siso', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--k', default=None, type=int, help='experiment name')
    argparser.add_argument('--eyN0', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--pilot_ofdm_symbol_indices', default=None, type=list, help='experiment name')
    argparser.add_argument('--design5G', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--designGA', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--lmmse', action='store_true', help='save model as an artifact if True')
    argparser.add_argument('--estimate_covariance', action='store_true', help='enable covariance estimation for LMMSE channel estimator')
    argparser.add_argument('--code_rates', default=None, type=lambda s: [float(item) for item in s.split(',')], help='comma-separated list of code rates, e.g., 0.5,0.2,0.1')
    argparser.add_argument('--no_tf', action='store_true', help='run without tf.function (eager mode)')
    argparser.add_argument('--layers_per_op_emb2llr', default=None, type=int, help='layers_per_op specifically for emb2llr (defaults to layers_per_op if not set)')
    argparser.add_argument('--emb2llr_snr', action='store_true', help='feed log(no) as extra input to emb2llr for SNR-aware LLR magnitude calibration')
    argparser.add_argument('--resume_train', action='store_true', help='if set, load the latest saved artifact and continue training from it')
    argparser.add_argument('--train_mode', default=None, type=str,
                           help='which submodules to train: all (default) | emb2llr | ey_emb2llr | ey')
    argparser.add_argument('--llr_clip', default=None, type=float,
                           help='clip emb2llr output to [-llr_clip, llr_clip]; None disables clipping')

    args = argparser.parse_args()
    return args


def read_configs(args=None):
    configs_path = "./configs/configs.json"
    with open(configs_path) as json_file:
        config_dict = json.load(json_file)

    if args is not None:
        if getattr(args, 'no_tf', False):
            tf.config.run_functions_eagerly(True)
        for key, val in vars(args).items():
            if val is not None:
                if key in config_dict.keys():
                    config_dict[key] = val

    if config_dict["Ns"] is not None:
        config_dict["Ns"] = eval(config_dict["Ns"])
        if len(config_dict['Ns']) == 1 and config_dict['k'] is None:
            config_dict['k'] = int(config_dict['code_rate'] * 2 ** config_dict['Ns'][0])
    return config_dict


def print_config(config):
    for key, val in config.items():
        print("{:25s} | {:40s}".format(key, str(val)))
    print("-" * 70)


def choose_channel(config, train=False):
    name = config['channel_name']
    batch = config['batch']
    snr = config['snr']
    isi = config['isi_length']

    if name == "awgn":
        return AWGN(var=1.0/snr, mean=0.0)
    elif name == "bsc":
        return BSC(p=0.1)
    elif name == "bec":
        return BEC(p0=0.5, p1=0.5)
    elif name == "bec_hy":
        return BEC(p0=0.4, p1=0.8159)
    elif name == "z":
        return Z(p=0.5)
    elif name == "ising":
        return Ising(p=0.5, batch_size=batch)
    elif name == "trapdoor":
        return (Trapdoor(batch_size=batch),Trapdoor(batch_size=batch))
    elif name == "ge":
        ge = GE(batch_size=batch, rate=config["code_rate"])
        return (ge, ge)
    elif name == "isi":
        return ISI(batch_size=batch, length=isi)
    elif name == "ma_agn":
        return MovingAverageAGN(batch_size=batch)
    elif name == "5g":
        if not train:
            assert len(config['Ns']), "in 5G channels the Ns list must contain a single element"
           #assert config['Ns'][0] >= 6, "in 5G channels N should be >= 64"
        snrdb_range = (config['snr_low'], config['snr_high'])
        channel = OFDMSystem(snrdb=snrdb_range,
                 fft_size=config['subcarrier_num'],
                 pilot_ofdm_symbol_indices=config['pilot_ofdm_symbol_indices'],
                 pilots_step=config['pilots_step'],
                 num_ofdm_symbols=config['num_ofdm_symbols'],
                 pcsi=config['perfect_csi'],
                 bps=config['NUM_BITS_PER_SYMBOL'],
                 interleaving=config['interleaving'],
                 sionna=config['sionna'],
                 domain=config['domain_5g'],
                 snrtoebno=config['snrtoebno'],
                 code_rate=config['code_rate'],
                 subcarrier_spacing=config['subcarrier_spacing'],
                 cyclic_prefix_length=config['cyclic_prefix_length'],
                 batch=config['batch'],
                 ibo=config['ibo'],
                 hpa_apply=config['apply_hpa'],
                 save_model=config['save_model'],
                 siso=config['siso'],
                 channel_type=config['channel_type'],
                 channel_mode=config['channel_mode'],
                 delay_spread=config['delay_spread'],
                 carrier_frequency=config['carrier_frequency'],
                 doppler_speed_min=config['doppler_speed_min'],
                 doppler_speed_max=config['doppler_speed_max'],
                 print_details=True,
                 lmmse_channel_estimator=config['lmmse'],
                 estimate_covariance=config['estimate_covariance'],)
        if is_log_power_of_two(channel.N) or config['sionna']:
            channel_design = channel
            # channel_design = OFDMSystem(snrdb=(-10,10),
            #                      fft_size=config['subcarrier_num'],
            #                      pilot_ofdm_symbol_indices=config['pilot_ofdm_symbol_indices'],
            #                      pilots_step=config['pilots_step'],
            #                      num_ofdm_symbols=config['num_ofdm_symbols'],
            #                      pcsi=config['perfect_csi'],
            #                      bps=config['NUM_BITS_PER_SYMBOL'],
            #                      interleaving=config['interleaving'],
            #                      sionna=config['sionna'],
            #                      domain=config['domain_5g'],
            #                      snrtoebno=config['snrtoebno'],
            #                      code_rate=config['code_rate'],
            #                      subcarrier_spacing=config['subcarrier_spacing'],
            #                      cyclic_prefix_length=config['cyclic_prefix_length'],
            #                      batch=config['batch'],
            #                      ibo=config['ibo'],
            #                      hpa_apply=config['apply_hpa'],
            #                      save_model=True,
            #                      siso=config['siso'],
            #                      channel_type=config['channel_type'],
            #                      channel_mode=config['channel_mode'],
            #                      delay_spread=config['delay_spread'],
            #                      carrier_frequency=config['carrier_frequency'],
            #                      doppler_speed_min=config['doppler_speed_min'],
            #                      doppler_speed_max=config['doppler_speed_max'],
            #                      print_details=True,)
        else:
            channel_design = channel
            # channel_design = OFDMSystem(snrdb=(-10,10),
            #                      fft_size=config['subcarrier_num'],
            #                      pilot_ofdm_symbol_indices=config['pilot_ofdm_symbol_indices'],
            #                      pilots_step=config['pilots_step'],
            #                      num_ofdm_symbols=config['num_ofdm_symbols'],
            #                      pcsi=config['perfect_csi'],
            #                      bps=config['NUM_BITS_PER_SYMBOL'],
            #                      interleaving=config['interleaving'],
            #                      sionna=config['sionna'],
            #                      domain=config['domain_5g'],
            #                      snrtoebno=config['snrtoebno'],
            #                      code_rate=config['code_rate'],
            #                      subcarrier_spacing=config['subcarrier_spacing'],
            #                      cyclic_prefix_length=config['cyclic_prefix_length'],
            #                      batch=config['batch'],
            #                      ibo=config['ibo'],
            #                      hpa_apply=config['apply_hpa'],
            #                      save_model=True,
            #                      siso=config['siso'],
            #                      channel_type=config['channel_type'],
            #                      channel_mode=config['channel_mode'],
            #                      delay_spread=config['delay_spread'],
            #                      carrier_frequency=config['carrier_frequency'],
            #                      doppler_speed_min=config['doppler_speed_min'],
            #                      doppler_speed_max=config['doppler_speed_max'],
            #                      print_details=True,)
            # channel_design = OFDMSystem(snrdb=snrdb_range,
            #          #fft_size=int(2**config['Ns'][0]/config['num_ofdm_symbols']/config['NUM_BITS_PER_SYMBOL']),
            #         fft_size=config['subcarrier_num'],
            #         pilot_ofdm_symbol_indices=config['pilot_ofdm_symbol_indices'],
            #          pilots_step=config['pilots_step'],
            #          num_ofdm_symbols=config['num_ofdm_symbols'],
            #          pcsi=config['perfect_csi'],
            #          bps=config['NUM_BITS_PER_SYMBOL'],
            #          interleaving=config['interleaving'],
            #          sionna=config['sionna'],
            #          domain=config['domain_5g'],
            #          snrtoebno=config['snrtoebno'],
            #          #snrtoebno=[-5, 10],
            #          code_rate=config['code_rate'],
            #          subcarrier_spacing=config['subcarrier_spacing'],
            #          cyclic_prefix_length=config['cyclic_prefix_length'],
            #          batch=config['batch'],
            #          ibo=config['ibo'],
            #          hpa_apply=config['apply_hpa'],
            #          save_model=config['save_model'],
            #          channel_type=config['channel_type'],
            #          channel_mode=config['channel_mode'],
            #          delay_spread=config['delay_spread'],
            #          carrier_frequency=config['carrier_frequency'],
            #          doppler_speed_min=config['doppler_speed_min'],
            #          doppler_speed_max=config['doppler_speed_max'],
            #          #doppler_speed_max=16,
            #                             print_details=True)

        return (channel, channel_design)
    elif name == "sc":
        channel = SingleCarrier(num_bits_per_symbol=config['NUM_BITS_PER_SYMBOL'],
                                frame_size=int(2**config['Ns'][0]), channel=config['channel_type'],
                                channel_mode=config['channel_mode'],
                                delay_spread=config['delay_spread'],
                                carrier_frequency=config['carrier_frequency'])
        return (channel, channel)
    elif name == "ge_awgn":
        n0_g = 0.5
        n0_b = 2.0
        p_gb = 1 / 16
        p_bg = 1 / 16
        channel = GilbertElliottChannel(n0_g, n0_b, p_gb, p_bg)
        return (channel, channel)
    else:
        raise ValueError("invalid channel name")





def set_wandb(config):
    wandb.init(project=config["project"],
               entity=config["entity"],
               group=config["group"],
               mode=config['wandb_mode'])
    wandb.config.update(config)
    wandb.define_metric("iter_scl")
    wandb.define_metric("accuracy", step_metric="iter_scl")
    wandb.define_metric("ce", step_metric="iter_scl")
    wandb.define_metric("n")
    wandb.define_metric("ber", step_metric="n")
    wandb.define_metric("fer", step_metric="n")


def log2wandb(wandb, ber, bler, P, Ns):
    def log_df_as_table(df, name):
        my_table = wandb.Table(columns=df.columns.to_list(), data=df.values)
        wandb.log({name: my_table})

    df_ber = pd.DataFrame(data={"ber": ber, "bler": bler}, index=Ns)
    df_ber.index.name = "n"
    log_df_as_table(df_ber, "ber_table")

    for n, p_err in zip(Ns, P):
        df_bit_ch = pd.DataFrame(data={"error_prob": np.sort(p_err)}, index=np.arange(2 ** n))
        df_bit_ch.index.name = "channel_index"
        log_df_as_table(df_bit_ch, f"bit-channels-{n}")


def gpu_init():
    """ Allows GPU memory growth """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("GPUS have already been initialized")

