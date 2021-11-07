import argparse
import torch
from torchvision import utils
#from model import Generator
from swagan_c import Generator
from tqdm import tqdm

@torch.no_grad()

def int_comp(a):
    x=[]
    if a==0:
        return 'cygames'
    elif a==1:
        return 'MAPPA'
    elif a==2:
        return 'pawroks'
    elif a==3:
        return 'SHAFT'
    elif a==4:
        return 'whitefox'
    elif a==5:
        return 'ジブリ'
    elif a==6:
        return 'A-1'
    elif a==7:
        return 'マッドハウス'
    else:
        return '京アニ'
def generate(args, g_ema, device, mean_latent,sample_z,one_hot,comp):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            one=torch.unsqueeze(one_hot,0).repeat(1,1).to(device)
            noises=[sample_z[i], one]
            noise=torch.cat(noises,dim=1)

            sample, _ = g_ema(
                [noise], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"proj3/test/{int_comp(comp)}_{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=5, help="number of images to be generated"
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

    args = parser.parse_args()

    args.latent = 512+9
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)
    

    g_ema.load_state_dict(checkpoint["g_ema"],strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    sample_z = torch.randn(5,args.sample, 512, device=device)
    for i in range(9):
        one_hot = torch.nn.functional.one_hot(torch.tensor(i), num_classes=9)
        generate(args, g_ema, device, mean_latent,sample_z,one_hot,i)
