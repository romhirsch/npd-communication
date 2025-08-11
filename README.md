# Neural Polar Decoders for communication
This repo contains the source code of the paper "A Study of Neural Polar Decoders for Communication"[[1]](https://ieeexplore.ieee.org/document/10711969)
This repository contains the code for training and evaluating neural polar decoders (NPDs) for end-to-end communication systems.
The code include NPD adapt for OFDM and single-carrier and extended to support any code length via rate matching, higher-order modulations, and robustness
across diverse channel conditions.
This code is based on the github reposetory [neural-polar-decoders](https://github.com/zivaharoni/neural-polar-decoders)
---

## Setup

For conda:

```bash
  conda create -n npd-env python=3.9 -y
  conda activate npd-env
```
Clone the repository:
```bash
    git clone https://github.com/romhirsch/neural-polar-decoders-communication.git
    cd  neural-polar-decoders-communication 
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Analytic Polar Decoders

#### 5G polar decoder
```bash
python ./5G-polar-decode.py --entity=data-driven-polar-codes --project=npd_communication --group=sc_2p_rates_qpsk --channel_name=5g --wandb_mode=online --snr_low=2 --snr_high=2 --Ns=[8] --code_rate=0.1 --batch=200 --mc_length=100000 --sionna=sc --mc_design=100000 --design_load --list_num=16 --subcarrier_num=32 --subcarrier_spacing=15e3 --num_ofdm_symbols=10 --domain_5g=time --doppler_speed_min=0 --doppler_speed_max=4 --carrier_frequency=3.5e9 --NUM_BITS_PER_SYMBOL=1 --link_channel=uplink --channel_mode=A --delay_spread=100e-9 --pilots_step=1 --crc=CRC11 --pilot_ofdm_symbol_indices=2,7
```

### Neural Polar Decoder
possible parameters appears in utils/utils.py
#### Training example of OFDM subcarrier=64, QPSK, ofdm symbols=8, N=1024

```bash
python ./nsc-train.py --entity=data-driven-polar-codes --project=npd_communication --group=debug --wandb_mode=online --channel_name=5g --num_iters=800000 --train_block_length=1024 --batch=100 --embedding_size_polar=128 --hidden_polar=300 --layers_per_op=2 --lr=0.001 --optimizer=adam --logging_freq=1000 --saving_freq=6400 --snr_low=-5 --snr_high=15 --input_distribution=sc --pred_decay=0 --save_model --subcarrier_spacing=15e3 --domain_5g=time --lr_decay=0.999 --num_ofdm_symbols=8 --NUM_BITS_PER_SYMBOL=2 --subcarrier_num=64 --doppler_speed_min=0 --doppler_speed_max=8 --carrier_frequency=3.5e9 --No_pilots --pilots_step=0 --cyclic_prefix_length=0
```

#### Evaluation

```bash
python ./nscl-decode.py --entity=data-driven-polar-codes --project=npd_communication --group=debug  --wandb_mode=online --channel_name=5g --snr_low=-5 --snr_high=-5 --Ns=[9] --code_rate=0.3 --tol=10 --batch=200 --mc_length=20000 --embedding_size_polar=128 --hidden_polar=300 --layers_per_op=2 --activation=elu --mc_design=500000 --list_num=16 --num_ofdm_symbols=8 --subcarrier_num=27 --subcarrier_spacing=15e3 --NUM_BITS_PER_SYMBOL=2 --doppler_speed_min=0 --doppler_speed_max=8 --carrier_frequency=3.5e9 --domain_5g=time --delay_spread=100e-9 --channel_mode=A --No_pilots --pilots_step=0 --crc=CRC11 --cyclic_prefix_length=0 --nsc_path=wandb_train:latest
```


---

## Citation

This repo is part of a series of research papers on learning-based polar decoding.
Data-driven neural polar decoders are introduced in [[1]](https://ieeexplore.ieee.org/document/10711969), and the code rate optimization is presented in [[2]](https://ieeexplore.ieee.org/document/10619429).

```latex

@article{aharoniDatadrivenNeuralPolar2024,
  title = {A Study of {{Neural Polar Decoders}} for {{Communication}}},
  author = {Hirsch, Rom, Aharoni, Ziv and Pfister, Henry D and Permuter, Haim H},
  year = {2025},
  journal = {},
  publisher = {},
  keywords = {}
}


@article{aharoniDatadrivenNeuralPolar2024,
  title = {Data-Driven {{Neural Polar Decoders}} for {{Unknown Channels}} with and without {{Memory}}},
  author = {Aharoni, Ziv and Huleihel, Bashar and Pfister, Henry D and Permuter, Haim H},
  year = {2024},
  journal = {IEEE Transactions on Information Theory},
  publisher = {IEEE},
  keywords = {Artificial neural networks,Channel estimation,Channel models,Channels with memory,Computational complexity,data-driven,Decoding,Memoryless systems,neural polar decoder,polar codes,Polar codes,Power capacitors,Training,Transforms}
}

@inproceedings{aharoniCodeRateOptimization2024,
  title = {Code {{Rate Optimization}} via {{Neural Polar Decoders}}},
  booktitle = {2024 {{IEEE International Symposium}} on {{Information Theory}} ({{ISIT}})},
  author = {Aharoni, Ziv and Huleihel, Bashar and Pfister, Henry D. and Permuter, Haim H.},
  year = {2024},
  pages = {2424--2429},
  issn = {2157-8117},
  doi = {10.1109/ISIT57864.2024.10619429},
  keywords = {Channel capacity,Channel models,channels with memory,Codes,Complexity theory,data-driven,Decoding,Knowledge engineering,Memoryless systems,polar codes,Power capacitors}
}

```
---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---
## Contact

Rom Hirsch
PhD student, Ben Gurion University
romhi@post.bgu.ac.il