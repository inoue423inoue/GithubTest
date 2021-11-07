import argparse
import torch
from torchvision import utils
#from model import Generator
from swagan import Generator
from tqdm import tqdm

@torch.no_grad()
def generate(args, g_ema, device, mean_latent,ck1,ck2,p):

    with torch.no_grad():
        g_ema.eval()
        l1 = ck1[list(ck1.keys())[1]]['latent']
        l2 = ck2[list(ck2.keys())[8]]['latent']
        for i in tqdm(range(args.pics)):
            latent = (1-i/args.pics)*l1+(i/args.pics)*l2
            latent = latent.unsqueeze(0).repeat(16, 1).unsqueeze(0).to('cuda:0')
            sample, _ = g_ema([latent], input_is_latent=True)

            utils.save_image(
                sample,
                f"output2/{str(p*args.pics+i).zfill(6)}.png",
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
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
#10274 11007
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"],strict=False)
    
    
    checkpoint1 = torch.load(args.files[0])
    checkpoint2 = torch.load(args.files[0])
        
            #a = list(checkpoint1.keys())[0]
            #b = list(checkpoint2.keys())[0]

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

        generate(args, g_ema, device, mean_latent,checkpoint1,checkpoint2,0)
