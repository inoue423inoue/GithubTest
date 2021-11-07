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


#ユークリッド距離
def euclid(hg1,hg2):
    return torch.sqrt(torch.sum((hg1-hg2)** 2))

#画像の距離のうち、大きい順と小さい順に100つずつ保持する
def big_and_small(big,small,a,b,c):
    big.append((a,b,c))
    big.sort(key=lambda pair: pair[0])
    small.append((a,b,c))
    small.sort(key=lambda pair: pair[0])
    if(len(big)>100):
        del big[0]
        del small[-1]
        
    return big,small
            
            
if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    big=[]
    small=[]
    score=0
    parser.add_argument("path", type=str, help="path to the image dataset")
    
    args = parser.parse_args()
    folders= glob.glob(f"{args.path}/*")
    
    alert=('注意:時間の都合で全ての作品のlatentを取り出せているわけではなく、\n'+
            'またlatentを取り出す画像も同作品内から20枚まで名前順に取り出したものを使用しているので、\n'
            +'ので、おおよその値しか出せていません。\n'
            +'また、スコア算出にはユークリッド距離を使用しています\n\n\n')
    f = open('proj/距離_同社.txt', 'w', encoding='UTF-8')
    #f.write(str) で書き込み
    f.write(alert)
    
    #同じ作品、同じ会社内での結果
    f.write('----------同じ作品、会社内での結果----------\n')
    for files in folders:
        #print(files)
        works=glob.glob(files+'/*.pt')
        if(len(works)>8):
            f.write(files.split('/')[-1]+' 作品数:'+str(len(works))+'\n')
            for i in range(len(works)):
                sum=0
                checkpoint = torch.load(works[i])
                num=len(list(checkpoint.keys()))
                #作品内
                if(i==0):
                    f.write('作品内\n')
                for j in range(len(list(checkpoint.keys()))):
                    for k in range(j+1,len(list(checkpoint.keys()))):
                        l1 = checkpoint[list(checkpoint.keys())[j]]['latent']
                        l2 = checkpoint[list(checkpoint.keys())[k]]['latent']
                        score=euclid(l1,l2)
                        big,small=big_and_small(big,small,score.item(),list(checkpoint.keys())[j],list(checkpoint.keys())[k])
                        sum = sum+score
                ans=sum/((num*(num-1))/2)
                f.write(works[i].split('/')[-1]+'  画像数: '+str(len(list(checkpoint.keys())))+'  平均: '+str(ans.item())+'\n')
                
            f.write('\n')
            sum_company=0
            #同会社の作品間
            for i in range(len(works)):
                if(i==0):
                    f.write('作品間\n')
                checkpoint1 = torch.load(works[i])
                for j in range(i+1,len(works)):
                    sum=0
                    checkpoint2 = torch.load(works[j])
                    for n1 in range(len(list(checkpoint1.keys()))):
                        for n2 in range(len(list(checkpoint2.keys()))):
                            l1 = checkpoint1[list(checkpoint1.keys())[n1]]['latent']
                            l2 = checkpoint2[list(checkpoint2.keys())[n2]]['latent']
                            score=euclid(l1,l2)
                            big,small=big_and_small(big,small,score.item(),list(checkpoint1.keys())[n1],list(checkpoint2.keys())[n2])
                            sum=sum+score
                    ans=sum/(len(list(checkpoint1.keys()))*len(list(checkpoint2.keys())))
                    sum_company=sum_company+ans
                    f.write(works[i].split('/')[-1]+' と '+works[j].split('/')[-1]+'  画像数: '+str(len(list(checkpoint1.keys())))+'*'+str(len(list(checkpoint2.keys())))+'  平均: '+str(ans.item())+'\n')
            ans_company=sum_company/((len(works)*(len(works)-1))/2)
            f.write('制作会社内平均(作品間の数値の全体平均): '+str(ans_company.item())+'\n')
            f.write('\n\n')
    f.close()
    f = open('proj/距離_他社.txt', 'w', encoding='UTF-8')
    f.write(alert)
    f.write('----------他社間のスコア----------\n')
    for i in range(len(folders)):
        works1=glob.glob(folders[i]+'/*.pt') 
        if(len(works1)<8):
            continue
        for j in range(i+1,len(folders)):
            works2=glob.glob(folders[j]+'/*.pt')

            if(len(works2)<8):
                continue
            f.write('\n'+folders[i].split('/')[-1]+'社 と '+folders[j].split('/')[-1]+'社\n')
            sum_company=0
            for k in range(len(works1)):
                checkpoint1 = torch.load(works1[k])
                for l in range(len(works2)):
                    checkpoint2 = torch.load(works2[l])
                    sum=0
                    for n1 in range(len(list(checkpoint1.keys()))):
                        for n2 in range(len(list(checkpoint2.keys()))):
                            l1 = checkpoint1[list(checkpoint1.keys())[n1]]['latent']
                            l2 = checkpoint2[list(checkpoint2.keys())[n2]]['latent']
                            score=euclid(l1,l2)
                            big,small=big_and_small(big,small,score.item(),list(checkpoint1.keys())[n1],list(checkpoint2.keys())[n2])
                            sum=sum+score
                    ans=sum/(len(list(checkpoint1.keys()))*len(list(checkpoint2.keys())))
                    sum_company=sum_company+ans
                    f.write(works1[k].split('/')[-1]+' と '+works2[l].split('/')[-1]+'  画像数: '+str(len(list(checkpoint1.keys())))+'*'+str(len(list(checkpoint2.keys())))+'  平均: '+str(ans.item())+'\n')
            ans_company=sum_company/(len(works1)*(len(works2)))
            f.write('制作会社間平均(作品間の数値の全体平均): '+str(ans_company.item())+'\n')
        f.write('\n')
    f.write('\n\n')
    f.close()
    
    f = open('proj/距離_ランキング.txt', 'w', encoding='UTF-8')
    f.write(alert)
    f.write('----------スコアの少なかった順に100組----------\n')
    for i,p in enumerate(small):
        f.write(str(i)+' '+p[1]+' と '+p[2]+'  スコア: '+str(p[0])+'\n')
    
    f.write('\n\n')
    f.write('----------スコアの大きかった順に100組----------\n')
    for i,p in enumerate(big):
        f.write(str(i)+' '+p[1]+' と '+p[2]+'  スコア: '+str(p[0])+'\n')
    f.close()
    