from torch import nn
import math
import random
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
from my_parser import args

def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])


def metrics(trues, preds):
    trues = np.concatenate(trues, -1)
    preds = np.concatenate(preds, 0)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    mcc = matthews_corrcoef(trues, preds.argmax(-1))

    f1 = f1_score(trues, preds.argmax(-1))

    return acc, mcc, f1


def plot_heatmap(coefs_matrix, stock_names=None, number=1):

    coefs_matrix = coefs_matrix.detach().cpu().numpy()
    # coefs_matrix = coefs_matrix[:30, :30]
    # stock_names = stock_names[:30]

    plt.figure(figsize=(56, 40))

    ax = sns.heatmap(
        coefs_matrix,
        annot=False,
        cmap="YlOrRd",
        xticklabels=stock_names,
        yticklabels=stock_names[::-1],
        cbar=True,
        cbar_kws={"shrink": 0.8, "label": "Attention intensity", "format":""}
    )

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.title(f'{number-6}', fontsize=14)

    # 设置颜色条标签的字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(f'./data/{args.dataset}/heatmap/{number-6}.png')

    plt.close()



