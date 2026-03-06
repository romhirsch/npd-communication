# Neural Polar Decoders for Communication

Source code for the paper **"A Study of Neural Polar Decoders for Communication"** [[1]](https://arxiv.org/abs/2510.03069).

This repository implements Neural Polar Decoders (NPDs) for end-to-end 5G communication systems. It supports OFDM and single-carrier channels, arbitrary code lengths via rate matching, higher-order modulations (BPSK/QPSK), and robustness across diverse channel conditions (TDL-A/B/C, varying Doppler and delay spread).

Based on [neural-polar-decoders](https://github.com/zivaharoni/neural-polar-decoders).

---

## Setup

```bash
conda create -n npd-env python=3.9 -y
conda activate npd-env
git clone https://github.com/romhirsch/neural-polar-decoders-communication.git
cd neural-polar-decoders-communication
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── models/                        # Core model implementations
│   ├── channel_model_3gpp.py      # 3GPP 5G OFDM channel (TDL, LMMSE, pilots)
│   ├── channel_models.py          # AWGN / memory channel wrappers
│   ├── channel_single_carrier.py  # Single-carrier channel
│   ├── channel_GE.py              # Gilbert-Elliott channel
│   ├── polar_models.py            # All polar decoder classes (SC, SCL, NPD, 5G)
│   ├── sc_models.py               # SC-specific model utilities
│   ├── system_models.py           # End-to-end system model
│   ├── input_models.py            # Input/embedding models
│   ├── layers.py                  # Custom Keras layers
│   ├── dv_models.py               # DV-based models
│   ├── my5gpolar.py               # 5G polar code utilities
│   └── Rapp.py                    # Rapp amplifier model
│
├── utils/
│   └── utils.py                   # Argument parsing, config loading, channel selection
│
├── configs/
│   ├── configs.json               # Default experiment configuration
│   └── config_3gpp.xlsx           # 3GPP parameter reference table
│
├── losses/
│   └── information_loss.py        # Custom information-theoretic loss
│
├── metrics/
│   └── info_metrics.py            # BER/BLER metrics
│
├── ops/
│   └── loss_metric_utils.py       # Loss/metric utility functions
│
├── optimizers/
│   └── lr_schedulers.py           # Learning rate schedulers
│
│
└── [main scripts — see below]
```

---

## Main Scripts

### Training

| Script | Description |
|--------|-------------|
| `nsc-train.py` | Train a Neural Polar Decoder (NPD) over a 5G/AWGN channel |

### Evaluation — Analytic Decoders

| Script | Description |
|--------|-------------|
| `sc-decode.py` | SC decoder evaluation |
| `scl-decode.py` | SCL decoder evaluation |
| `sct-decode.py` | SC trellis decoder evaluation |
| `5G-polar-decode.py` | 5G SCL (CRC-aided) polar decoder evaluation |
| `5G-decode-rateplot.py` | **Main job script** — sweeps code rates, runs 5G SCL over 3GPP channel |

### Evaluation — Neural Decoders

| Script | Description |
|--------|-------------|
| `nsc-decode.py` | SC neural decoder evaluation |
| `nscl-decode.py` | SCL neural decoder evaluation |
| `nscl-decode-rateplot.py` | Neural SCL rate-sweep evaluation |
| `nscl-decode-reperframe.py` | Neural SCL errors-per-frame evaluation |
| `nscl-decode-throughtput.py` | Neural SCL throughput evaluation |
| `nscl-decode_sim.py` | Neural SCL simulation |

### Evaluation — Sionna-based Decoders

| Script | Description |
|--------|-------------|
| `sionna-sc-decode-reperframe.py` | SC errors-per-frame via Sionna |
| `sionna-sc-decode-throughput.py` | SC throughput via Sionna |
| `nscl-sn-decode-rateplot.py` | Neural SCL with Sionna channel, rate sweep |
| `scl-sn-decode-rateplot.py` | Analytic SCL with Sionna channel, rate sweep |
| `scl-decode_rateplot.py` | SCL rate sweep (channel models) |

### Utilities

| Script | Description |
|--------|-------------|
| `plot_thoughtput.py` | Plot throughput curves from WandB results |

---

## Example Usage

### Train NPD (OFDM, subcarrier=128, N=1024)

```bash
python nsc-train.py \
--entity=data-driven-polar-codes --project=npd_communication --group=train_npd_bpsk_1024_tdlc300_nn128-1-300 --wandb_mode=online --channel_name=5g --num_iters=800000 --train_block_length=1024 --batch=100 --embedding_size_polar=128 --hidden_polar=300 --layers_per_op=1 --lr=0.001 --optimizer=adam --logging_freq=1000 --saving_freq=6400 --snr_low=-10 --snr_high=15 --input_distribution=sc --pred_decay=0 --save_model --subcarrier_spacing=15e3 --domain_5g=time --lr_decay=0.99 --num_ofdm_symbols=8 --NUM_BITS_PER_SYMBOL=1 --subcarrier_num=128 --doppler_speed_min=0 --doppler_speed_max=35 --carrier_frequency=3.5e9 --No_pilots --pilots_step=0 --eyN0 --cyclic_prefix_length=0 --delay_spread=300e-9 --channel_mode=C
```

### Evaluate Neural SCL Decoder

```bash
python nscl-decode.py \
  --project=npd_communication --group=debug --wandb_mode=online \
  --channel_name=5g --snr_low=-5 --snr_high=10 --Ns=[9] --code_rate=0.3 \
  --batch=200 --mc_length=20000 --embedding_size_polar=128 --hidden_polar=300 \
  --layers_per_op=2 --list_num=16 --num_ofdm_symbols=8 --subcarrier_num=27 \
  --subcarrier_spacing=15e3 --NUM_BITS_PER_SYMBOL=2 \
  --doppler_speed_min=0 --doppler_speed_max=8 --carrier_frequency=3.5e9 \
  --domain_5g=time --delay_spread=100e-9 --channel_mode=A \
  --crc=CRC11 --nsc_path=wandb_train:latest
```

### Evaluate 5G SCL with LMMSE (covariance estimated)

```bash
python 5G-decode-rateplot.py \
  --project=npd_communication --group=lmmse-run --channel_name=5g \
  --wandb_mode=online --snr_low=3 --snr_high=10 --Ns=[10] --code_rate=0.5 \
  --batch=100 --mc_length=100000 --sionna=5g --mc_design=100000 \
  --design_load --list_num=16 --subcarrier_num=128 --subcarrier_spacing=15e3 \
  --num_ofdm_symbols=10 --domain_5g=time \
  --doppler_speed_min=0 --doppler_speed_max=8 --carrier_frequency=3.5e9 \
  --NUM_BITS_PER_SYMBOL=1 --link_channel=uplink --channel_mode=A \
  --delay_spread=100e-9 --pilots_step=1 --pilot_ofdm_symbol_indices=2,7 \
  --lmmse --code_rates=0.5 --k=128 --estimate_covariance --snrtoebno
```

### Evaluate 5G SCL with LMMSE (analytic covariance)

```bash
python 5G-decode-rateplot.py \
  --project=npd_communication --group=lmmse-run --channel_name=5g \
  --wandb_mode=online --snr_low=3 --snr_high=10 --Ns=[10] --code_rate=0.5 \
  --batch=100 --mc_length=100000 --sionna=5g --mc_design=100000 \
  --design_load --list_num=16 --subcarrier_num=128 --subcarrier_spacing=15e3 \
  --num_ofdm_symbols=10 --domain_5g=time \
  --doppler_speed_min=0 --doppler_speed_max=8 --carrier_frequency=3.5e9 \
  --NUM_BITS_PER_SYMBOL=1 --link_channel=uplink --channel_mode=A \
  --delay_spread=100e-9 --pilots_step=1 --pilot_ofdm_symbol_indices=2,7 \
  --lmmse --code_rates=0.5 --k=128 --snrtoebno
```

### Evaluate throughput — Sionna SC with LMMSE

```bash
python sionna-sc-decode-throughput.py \
  --project=npd_communication --group=debug --channel_name=5g \
  --wandb_mode=online --snr_low=0 --snr_high=10 --Ns=[10] --code_rate=0.5 \
  --batch=100 --mc_length=100000 --sionna=5g --mc_design=100000 \
  --design_load --list_num=16 --subcarrier_num=128 --subcarrier_spacing=15e3 \
  --num_ofdm_symbols=8 --domain_5g=time \
  --doppler_speed_min=0 --doppler_speed_max=8 --carrier_frequency=3.5e9 \
  --NUM_BITS_PER_SYMBOL=1 --link_channel=uplink --channel_mode=A \
  --delay_spread=100e-9 --pilots_step=2 --pilot_ofdm_symbol_indices=2 \
  --lmmse --code_rates=0.5 --k=128 --estimate_covariance --snrtoebno
```

### Evaluate throughput — Neural SCL

```bash
python nscl-decode-throughtput.py \
  --entity=data-driven-polar-codes --project=npd_communication \
  --group=npd-th-k128-low-v62 --wandb_mode=online --channel_name=5g \
  --snr_low=-10 --snr_high=15 --Ns=[9] --code_rate=0.1 --tol=10 \
  --batch=200 --mc_length=50000 --embedding_size_polar=128 --hidden_polar=300 \
  --layers_per_op=1 --activation=elu --mc_design=500000 --list_num=16 \
  --num_ofdm_symbols=8 --subcarrier_num=64 --subcarrier_spacing=15e3 \
  --NUM_BITS_PER_SYMBOL=1 --doppler_speed_min=0 --doppler_speed_max=8 \
  --carrier_frequency=3.5e9 --domain_5g=time --delay_spread=100e-9 \
  --channel_mode=A --No_pilots --pilots_step=0 --crc=CRC11 \
  --cyclic_prefix_length=0 --eyN0 --k=128 \
  --nsc_path=train_group-train_npd_bpsk_1024_tdlc300_nn128-1-300_sc_5g-10.0_nt-1024_npd-128-1x300:v62
```


---

## Citation

```bibtex
@article{hirschStudyNeuralPolar2025,
  title   = {A Study of {Neural Polar Decoders} for {Communication}},
  author  = {Hirsch, Rom and Aharoni, Ziv and Pfister, Henry D and Permuter, Haim H},
  year    = {2025}
}

@article{aharoniDatadrivenNeuralPolar2024,
  title     = {Data-Driven {Neural Polar Decoders} for {Unknown Channels} with and without {Memory}},
  author    = {Aharoni, Ziv and Huleihel, Bashar and Pfister, Henry D and Permuter, Haim H},
  year      = {2024},
  journal   = {IEEE Transactions on Information Theory},
  publisher = {IEEE}
}

@inproceedings{aharoniCodeRateOptimization2024,
  title     = {Code {Rate Optimization} via {Neural Polar Decoders}},
  booktitle = {2024 {IEEE International Symposium on Information Theory} ({ISIT})},
  author    = {Aharoni, Ziv and Huleihel, Bashar and Pfister, Henry D. and Permuter, Haim H.},
  year      = {2024},
  pages     = {2424--2429},
  doi       = {10.1109/ISIT57864.2024.10619429}
}
```

---

## License

Apache License 2.0. See [LICENSE](./LICENSE).

---

## Contact

Rom Hirsch — PhD student, Ben-Gurion University
romhi@post.bgu.ac.il
