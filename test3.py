import argparse
import torch
from torchvision import utils
#from model import Generator
from swagan import Generator
from tqdm import tqdm
from torchvision import datasets
import glob
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

            
            
if __name__ == "__main__":
    device = "cuda"

    
    a="inochan000"
    print(a[0:len(a)-3])
    