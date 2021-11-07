import argparse
import torch
from torchvision import utils
#from model import Generator
from swagan import Generator
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os
import glob

# GIFアニメーションを作成
def create_gif(in_dir, out_filename):
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    imgs = []                                                   # 画像をappendするための空配列を定義

    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i])                          # 画像ファイルを1つずつ開く
        imgs.append(img)                                        # 画像をappendで配列に格納していく

    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=0)

# GIFアニメーションを作成する関数を実行する
#create_gif(in_dir='output4', out_filename='animation2.gif')

@torch.no_grad()
def generate(args, g_ema, device, mean_latent,l1,l2,n_comp):

    with torch.no_grad():
        g_ema.eval()
        
        for i in tqdm(range(args.pics)):
            latent = (1-i/args.pics)*pca.inverse_transform(l1)+(i/args.pics)*pca.inverse_transform(l2)
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
        create_gif("output4",f"pca_result/n_comp_{n_comp}.gif")

            
            
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

    

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"],strict=False)
    n_comps=[1,3,30,100,200,300,400,512]
    
    for i in n_comps:
        data=np.load("pca_human.npy")
        pca = PCA(n_components=i)

        pca.fit(data)
        data_pca = pca.transform(data)
        #圧縮した情報から元の次元の画像に復元
        #data_pca = pca.inverse_transform(data_pca)
        

        checkpoint1 = data_pca[args.start]
        checkpoint2 = data_pca[args.goal]
        
        #print(checkpoint1)
        
        #a = list(checkpoint1.keys())[0]
        #b = list(checkpoint2.keys())[0]

        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g_ema.mean_latent(args.truncation_mean)
        else:
            mean_latent = None

        generate(args, g_ema, device, mean_latent,checkpoint1,checkpoint2,i)
