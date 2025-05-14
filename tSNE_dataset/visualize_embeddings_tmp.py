import os
import pickle

import numpy as np
import pandas as pd
import math

from matplotlib import patches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import results_dir
from utils.model_definitions import MODELS

VAR = False

def visualize_embeddings(model_name):
    model_name = 'default_logan_ewc_2'
    model_results = '/shared/data/5_utr_embeddings_cls/default_multi_species_2kb/'
    if VAR:
        files = sorted([f for f in os.listdir(model_results) if
                        f.endswith(".pkl") and not f.startswith("tsne_") and 'var' in f],
                       key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
    else:
        files = sorted([f for f in os.listdir(model_results) if
                        f.endswith(".pkl") and not f.startswith("tsne_") and not 'var' in f],
                       key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
        files = [f.replace('_var', '') for f in files]

    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n), sharex=True, sharey=True)
    fig.suptitle(f"{model_name}", fontsize=16)

    if n == 1:
        axes = [axes]

    for idx, (ax, fpath) in enumerate(zip(axes, files)):
        layer = int(fpath.split("layer_")[-1].split(".")[0])
        tsne_path = os.path.join(model_results, f"tsne_{layer}{'_var' if VAR else ''}.pkl")

        with open(os.path.join(model_results, fpath), "rb") as f:
            data = pickle.load(f)

        embeddings = data["embeddings"]
        meta = data["meta"]
        df = pd.DataFrame(meta)

        if os.path.exists(tsne_path):
            with open(tsne_path, "rb") as f:
                tsne_results = pickle.load(f)
        else:
            tsne_results = TSNE().fit_transform(embeddings)
            with open(tsne_path, "wb") as f:
                pickle.dump(tsne_results, f)

        print(df.head())
        df["Dimension 1"] = tsne_results[:, 0]
        df["Dimension 2"] = tsne_results[:, 1]
        df["label_str"] = df["label"].map({0: "common", 1: "rare"})
        from sklearn.metrics import average_precision_score
        label_order = ["3UTR", "CDS", "intron", "intergenic", "5UTR"]
        sns.scatterplot(data=df, x="Dimension 1", y="Dimension 2",
                        hue="af", palette='viridis', s=5,
                        ax=ax, legend=(idx == n - 1))
        y_true = np.array([m["label"] for m in meta])
        scores = np.array([m["dot_product_norm"] for m in meta])
        auprc = average_precision_score(y_true, 1 - scores)  # invert if rare=1

        print("AUPRC:", auprc)
        # Annotate layer on the right side
        ax.text(1.01, 0.5, f"Layer {layer + 1}", transform=ax.transAxes,
                rotation=270, va='center', ha='left', fontsize=12)

        rect = patches.FancyBboxPatch(
            (1.00, -0.002),  # x, y (in axis coordinates)
            0.06, 1.0,  # width, height
            transform=ax.transAxes,
            boxstyle="round,pad=0.00",
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False
        )
        ax.add_patch(rect)
        ax.set_ylabel('')
        ax.tick_params(labelleft=True)
        ax.set_xlabel("Dimension 1", fontsize=12)


        # Add single legend
    handles, labels = axes[-1].get_legend_handles_labels()
    legend = axes[-1].legend(
        handles, labels,
        loc='lower left',
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        labelspacing=0.2,
        markerscale=2.0
    )
    legend.set_title(None)
    if legend:
        legend.set_title(None)
        legend.get_frame().set_linewidth(0)
        legend.get_frame().set_alpha(0)

    # Global x and y axis labels
    #fig.supxlabel("Dimension 1", fontsize=12)
    fig.supylabel("Dimension 2", fontsize=12)


    plt.subplots_adjust(hspace=0.02, bottom=0.04, top=0.95, left=0.12, right=0.92)
    plt.suptitle(f"t-SNE Embeddings – {model_name}", fontsize=14)
    plt.show()

if __name__ == "__main__":
    visualize_embeddings('')
