import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config.config import TSNE_PERPLEXITY
from utils.save_utils import save_plot

def run_tsne(embeddings, perplexity=TSNE_PERPLEXITY, seed=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    return tsne.fit_transform(embeddings)


def plot_tsne(embeddings_2d, labels, title, output_dir, filename):
    pos_idx = (labels == 1)
    neg_idx = (labels == 0)

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(embeddings_2d[neg_idx, 0], embeddings_2d[neg_idx, 1], c='tab:orange', s=40, edgecolor='k', label='Negative (0)', alpha=0.9)
    plt.scatter(embeddings_2d[pos_idx, 0], embeddings_2d[pos_idx, 1], c='tab:blue', s=20, label='Positive (1)', alpha=0.4)
    plt.legend()
    plt.title(title)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()
    save_plot(fig, filename, output_dir)

