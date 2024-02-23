# never_forget

Implementation of "Overcoming Catastrophic Forgetting in Neural Networks" (Kirkpatrick et al, 2017) https://arxiv.org/pdf/1612.00796.pdf

## Setup

Requires conda: https://docs.anaconda.com/free/miniconda/#quick-command-line-install

```bash
git clone https://github.com/catid/never_forget.git
cd never_forget

conda create -n nf python=3.10 -y && conda activate nf

pip install -r requirements.txt

# Download audio dataset for training
python download_dataset.py

# Verify that training works
python train.py
```

## Distributed Experiment

Set up each server:

```bash
git clone https://github.com/catid/never_forget.git
cd never_forget

conda create -n nf python=3.10 -y && conda activate nf

pip install -r requirements.txt

python command_server.py
```

On client side, edit the `hosts.txt` file to point to the servers you want to use.

```bash
git clone https://github.com/catid/never_forget.git
cd never_forget

conda create -n nf python=3.10 -y && conda activate nf

pip install -r requirements.txt

# Launch job
python command_client_grid.py

# Download results and combine them
python combine_results.py

# Graph results, producing heatmaps as .png files
python graph_results.py
```
