from torchviz import make_dot

import argparse
import math
import random
import os
from swagan import Generator

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

generator = Generator(
        256, 512, 8, channel_multiplier=2
    ).to("cuda")

sample_test = torch.randn(4, 512, device="cuda")
x, latents_test = generator([sample_test],return_latents=True)

dot=make_dot(latents_test,params=dict(generator.named_parameters()))

dot.format = 'png'
dot.render('graph_latents')
