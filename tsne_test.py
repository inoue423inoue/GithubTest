import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import torch
from torchvision import utils
import glob
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


a=False
folders= glob.glob("proj4/test1_256/*")
for folder in folders:
    pts=glob.glob(folder+'/*.pt')
    #if(len(pts)<8):
    #    continue
    for pt in pts:
        checkpoint = torch.load(pt)
        spt=pt.split('/')[-1]
        if folder.split('/')[-1]=='タツノコプロ':
            continue
        print(pt)
        for i in range(len(list(checkpoint.keys()))):
            l = checkpoint[list(checkpoint.keys())[i]]['latent']
            l2 = l.to('cpu').detach().numpy().copy()

            if a==False:
                b=[l2]
                a=True
                company=[folder.split('/')[-1]]
            else:
                b=np.concatenate([b, [l2]])
                #print(b.shape)
                company.append(folder.split('/')[-1])

#tsne = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state = 0, perplexity = 20, n_iter = 1000)

X_embedded = tsne.fit_transform(b)


ddf =pd.DataFrame(X_embedded, columns = ['col1', 'col2'])
colors =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink","darkviolet","gray","greenyellow","olive","khaki"]
fig=plt.figure(figsize = (100, 100))
name=""
i=0

company_list=list(dict.fromkeys(company))
for  j,v in enumerate(company_list):
    s=company.count(v)
    if v =='京アニ':
        vv='Kyoani'
    elif v =='ジブリ':
        vv='Ghibli'
    elif v =='マッドハウス':
        vv='MadHouse'
    else:
        vv=v
    plt.scatter(ddf['col1'][i:i+s],  
                ddf['col2'][i:i+s],
                label = vv,
                color = colors[j],
                s=2000)
    i=i+s
plt.show()
plt.legend(fontsize = 100)
fig.savefig("proj4/tsne.png")