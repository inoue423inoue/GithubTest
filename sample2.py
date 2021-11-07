import argparse
import torch
from torchvision import utils
#from model import Generator
from swagan import Generator
from tqdm import tqdm


sample_z = torch.randn(10, 20, device="cuda")
print(sample_z.size())
print(sample_z[0:][2].size())