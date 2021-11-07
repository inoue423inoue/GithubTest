import argparse
import torch
from torchvision import utils
#from model import Generator
from swagan import Generator
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA


@torch.no_grad()
def generate(args, g_ema, device, mean_latent,l1):

    with torch.no_grad():
        g_ema.eval()
        
        for i in tqdm(range(args.pics)):
            latent = l1
            latent[0]=latent[0]-0.1*i
            latent = pca.inverse_transform(latent)
            latent = torch.from_numpy(latent.astype(np.float32)).clone()
            latent = latent.unsqueeze(0).repeat(16, 1).unsqueeze(0).to('cuda:0')
            sample, _ = g_ema([latent], input_is_latent=True)

            utils.save_image(
                sample,
                f"output4/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            
            
if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--goal",
        type=int,
        default=123,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--n_comp",
        type=int,
        default=300,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"],strict=False)
    
    data=np.load("pca.npy")
    m=data.mean(axis=0)
    pca = PCA(n_components=512)
    

    pca.fit(data)
    data_pca = pca.transform(data)
    #圧縮した情報から元の次元の画像に復元
    #data_pca = pca.inverse_transform(data_pca)
    pca_mean=data_pca.mean(axis=0)
    

    checkpoint1 = data_pca[0]
    checkpoint2 = data_pca[1]
    
    #print(checkpoint1)
    
    #a = list(checkpoint1.keys())[0]
    #b = list(checkpoint2.keys())[0]

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent,pca_mean)
