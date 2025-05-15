import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from matplotlib import patches
import matplotlib.pyplot as plt
import seaborn as sns

from utils.model_definitions import MODELS

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def visualize_embeddings(model_name, embeddings_dir, tsne_dir):

    try:
        embeddings_files = sorted([os.path.join(embeddings_dir, f) for f in os.listdir(embeddings_dir) if f.endswith(".pkl") ], key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
        tsne_files = sorted([os.path.join(tsne_dir, f) for f in os.listdir(tsne_dir) if f.endswith(".pkl") ], key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
    except:
        return

    files = list(zip(embeddings_files, tsne_files))

    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n), sharex=True, sharey=True)
    fig.suptitle(f"{model_name}", fontsize=16)

    if n == 1:
        axes = [axes]

    for idx, (ax, (emb_file, tsne_file)) in enumerate(zip(axes, files)):

        with open(emb_file, "rb") as f:
            data = pickle.load(f)


        embeddings = data["embeddings"]
        meta = data["meta"]
        df = pd.DataFrame(meta)

        with open(tsne_file, "rb") as f:
            tsne_results = pickle.load(f)

        df["Dimension 1"] = tsne_results[:, 0]
        df["Dimension 2"] = tsne_results[:, 1]
        df["label_str"] = df["label"].map({0: "common", 1: "rare"})

        sns.scatterplot(data=df, x="Dimension 1", y="Dimension 2",
                        hue="dot_product", palette='viridis', s=5,
                        ax=ax, legend=(idx == n - 1))

        y_true = np.array([m["label"] for m in meta])
        scores = np.array([m["dot_product"] for m in meta])
        auprc = average_precision_score(y_true, scores - 1)  # invert if rare=1

        ax.text(0.98, 0.95, f"AUPRC: {auprc:.3f}",
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        layer = int(emb_file.split("layer_")[-1].split(".")[0])
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
    emb = ['5_utr_embeddings_mean']

    for e in emb:
        for m in MODELS:
            tsne_dir = os.path.join(DATA_PATH, 'tSNE', e, m['name'])
            embeddings_dir = os.path.join(DATA_PATH, 'embeddings', e, m['name'])
            visualize_embeddings(m['name'], embeddings_dir, tsne_dir)
