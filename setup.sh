#!/bin/sh
pip install -Iv torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html

pip install opencv-python

pip install stable_baselines3

pip install -e .