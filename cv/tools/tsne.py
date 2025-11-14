import argparse
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import sys
sys.path.append('.')

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, corr
from mdistiller.engine.cfg import CFG as cfg
import pdb
import imageio
import numpy as np
import cv2
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument("-n", "--name")
    parser.add_argument("-bs", "--batch-size", type=int, default=256)
    args = parser.parse_args()
    
    tsne = TSNE(n_components=2, init='pca')
    cfg.DATASET.TYPE = 'cifar100'
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    model, pretrain_model_path = cifar_model_dict[args.model]
    model = model(num_classes=num_classes)
    ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
    try:
        model.load_state_dict(load_checkpoint(ckpt)["model"])
    except:
        model.load_state_dict(load_checkpoint(ckpt))
    model = model.cuda()
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for idx, (image, target) in tqdm(enumerate(val_loader)):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits, feat = model(image)
            feats.append(feat['pooled_feat'].cpu().detach())
            labels.append(target.cpu())
            if len(labels) > 1000:
                break
        x_tsne = tsne.fit_transform(torch.vstack(feats))
        x_tsne = np.vstack((x_tsne.T, torch.cat(labels))).T
        df_tsne = pd.DataFrame(x_tsne, columns=['dim1', 'dim2', 'class'])
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_tsne, hue='class', x='dim1', y='dim2', palette='deep', legend=False, ax=None, s=8)
        plt.axis('off') 
        plt.savefig('tsne_'+args.name+'.jpg')
        exit()