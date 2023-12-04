import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_tsne(points, labels, title=None):
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = points[:, 0]
    df["comp-2"] = points[:, 1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df).set(title=title)
    plt.show()


def dim_reduction(points):
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(points)
    return pca_result


def visualization(points, labels=None):
    n_components = 2
    rng = RandomState(0)

    t_sne = TSNE(
        n_components=n_components,
        perplexity=30,
        n_iter=250,
        init="random",
        random_state=rng,
    )
    t_sne = t_sne.fit_transform(points)

    plot_tsne(t_sne, labels, "T-distributed Stochastic Neighbor Embedding")
