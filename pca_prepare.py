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

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()
    
    args.latent = 512
    args.n_mlp = 8
    data=np.empty((1,512))
    files = glob.glob(f"{args.path}/*")
    for file in range(len(files)):
        checkpoint = torch.load(files[file])
        for i in range(len(list(checkpoint.keys()))):
            latent = checkpoint[list(checkpoint.keys())[i]]['latent']
            #print(latent.shape) (512,)
            latent = latent.to('cpu').detach().numpy().copy()
            data=np.concatenate([data,[latent]])
    print(data.shape)
    data = np.delete(data, 0, 0)
    print(data.shape)
    #print(data[0])
    np.save('pca_human',data)
    pca=PCA()
    pca.fit(data)
    accumulated_ratio_ = np.add.accumulate(pca.explained_variance_ratio_)
    fig = plt.figure()
    plt.plot(accumulated_ratio_)
    fig.savefig("pca_img.png")
    #n_comp = 100

    #pca = PCA(n_components=n_comp)

    #pca.fit(data)
    #30次元の潜在空間に変換して圧縮
    #X_train_latent = pca.transform(data)
    #圧縮した情報から元の784次元の画像に復元
    #X_train_inv = pca.inverse_transform(X_train_latent)