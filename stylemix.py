import argparse
import torch
from torchvision import utils
#from model import Generator
from swagan import Generator
from tqdm import tqdm

@torch.no_grad()

def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample1 = torch.randn(args.sample, args.latent, device=device)
            sample2 = torch.randn(args.sample, args.latent, device=device)


            sample_image1, latent1 = g_ema(
                [sample1], truncation=args.truncation, truncation_latent=mean_latent,return_latents=True
            )
            #print(latent1.size()) 1,16,512

            utils.save_image(
                sample_image1,
                f"output_stylemix/{str(i).zfill(3)}_0_base.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            
            sample_image2, latent2 = g_ema(
                [sample2], truncation=args.truncation, truncation_latent=mean_latent,return_latents=True
            )

            utils.save_image(
                sample_image2,
                #f"output_stylemix/{str(i).zfill(6)}.png",
                f"output_stylemix/{str(i).zfill(3)}_0_effect.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            l=[]
            for k in range(12):
                l=[]
                for j in range(12):
                    if k==j:
                        l = l+[latent2[0][j].unsqueeze(0).unsqueeze(0).to('cuda:0')*args.rate]
                    else:
                        l = l+[latent1[0][j].unsqueeze(0).unsqueeze(0).to('cuda:0')]
                #print(len(l)) 12
                x = torch.cat(l, dim=1)
                #print(x.size()) #1,12,512
                sample_imagex, _ = g_ema(
                    [x], input_is_latent=True
                )

                utils.save_image(
                    sample_imagex,
                    #f"output_stylemix/{str(i).zfill(6)}.png",
                    f"output_stylemix/{str(i).zfill(2)}_1_w{str(k)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                
            for k in range(0,12,2):
                l=[]
                for j in range(12):
                    if k==j or j==k+1:
                        l = l+[latent2[0][j].unsqueeze(0).unsqueeze(0).to('cuda:0')*args.rate]
                    else:
                        l = l+[latent1[0][j].unsqueeze(0).unsqueeze(0).to('cuda:0')]
                x = torch.cat(l, dim=1)
                #print(x.size())
                sample_imagex, _ = g_ema(
                    [x], truncation=args.truncation, input_is_latent=True
                )

                utils.save_image(
                    sample_imagex,
                    #f"output_stylemix/{str(i).zfill(6)}.png",
                    f"output_stylemix/{str(i).zfill(2)}_2_w{str(k)}_{str(k+1)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
            
            for k in range(0,12,4):
                l=[]
                for j in range(12):
                    if k==j or j==k+1 or j==k+2 or j==k+3:
                        l = l+[latent2[0][j].unsqueeze(0).unsqueeze(0).to('cuda:0')*args.rate]
                    else:
                        l = l+[latent1[0][j].unsqueeze(0).unsqueeze(0).to('cuda:0')]
                x = torch.cat(l, dim=1)
                #print(x.size())
                sample_imagex, _ = g_ema(
                    [x], truncation=args.truncation, input_is_latent=True
                )

                utils.save_image(
                    sample_imagex,
                    #f"output_stylemix/{str(i).zfill(6)}.png",
                    f"output_stylemix/{str(i).zfill(2)}_4_w{str(k)}_{str(k+3)}.png",
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
        "--rate",
        type=float,
        default=1,
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

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
