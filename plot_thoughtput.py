import numpy as np
import wandb
import argparse
import pandas as pd
from collections import defaultdict
from matplotlib.legend import Legend
from collections.abc import Iterable

from tqdm import tqdm
import tikzplotlib  #  Latex figs
import os
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)
import matplotlib.pyplot as plt

def group_labels():
    pass
plot_case = 11
ksize_x_plot = False
# Log in to Weights & Biases
wandb.login()
i_marker=0
plot_rate_ber = True
# Project and account details
project_name = "npd_communication"
entity_name = "data-driven-polar-codes"
# Initialize the API
api = wandb.Api()
lines_fig1 = []
lines_fig2 = []
lines_fig3 = []
filter_subcarriers = None
color_per_block = None
## BPSK list 1

plots_name = 'BPSK_TDL-A_delay100_thougthput'
#group_names = ["npd-throk32", "5gp2s2-t-k32"]
group_names = ["npd-throk102", "5gp2s1-t-k102", "npd-throk32", "5gp2s2-t-k32"]
group_names = ["npd-throk32", "5gp2s1-t-k32"]

filter_dict = {"list_num": 16, "doppler_speed_max": 8, "channel_mode": 'A', "delay_spread": 1e-7} # BPSK list 1
rates_print = [0.3]  # List of rates to print
linestyles = {'npd': '--', 'sc': '-'}  # Define linestyles for each group
xmin = -10
xmax = 10
ymin = 1e-3
ymax=1
ds_ex_label = None
# Loop through each group in the list
# Create two separate figures
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
# fig4, ax4 = plt.subplots()

for group_name in group_names:
    print(f"\nFetching runs for group: {group_name}")

    # Fetch runs for the current group
    runs = api.runs(f"{entity_name}/{project_name}", filters={"group": group_name})

    if not runs:
        print(f"No runs found for group: {group_name}")
        continue

    # Process each run
    for run in tqdm(runs):

        #Retrieve run metrics
        # if run.state != "finished":
        #     continue
        config = run.config
        con = False
        if filter_subcarriers:
            if config['subcarrier_num'] not in filter_subcarriers:
                continue
        for k, v in filter_dict.items():
            if config.get(k) != v:
                con = True
                continue
        if con:
            continue
        print(f"\nRun ID: {run.id}, Name: {run.name}")
        reframe = 256
        for RE in [192, 256, 384]:
            th = run.history(keys=["snr", f"throughput_{RE}"]).drop(columns=["_step"])
            ax3.plot(th['snr'],th[f"throughput_{RE}"]/1e6, label=config['group']+f' {RE}',  markevery=2, markersize=5, marker='o')

        ax3.set_xlabel('SNR [dB]')
        ax3.set_ylabel('throughput [Mbit/s]')



ax3.grid(True)
ax3.legend()
#ax3.set_ylim([-7, 10])

# Save the figures
# Create a folder name based on the filter dictionary

folder_name = plots_name #"_".join([f"{key}={value}" for key, value in filter_dict.items()])
# Create the directory if it doesn't exist
os.makedirs(os.path.join("results_tex",folder_name), exist_ok=True)



tikzplotlib.save(f'./results_tex/{folder_name}/rate_plot.tex', figure=fig3)
#tikzplotlib.save(f'./results_tex/{folder_name}/bler_rate_plot.tex', figure=fig4)
# Save the figures as PNG files

fig3.savefig(f'./results_tex/{folder_name}/rate_plot.png')
#fig4.savefig(f'./results_tex/{folder_name}/bler_rate_plot.png')
# Show the plots
plt.show()