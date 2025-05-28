import os
from pprint import pprint

import numpy as np
from matplotlib import gridspec, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import images_dir
from utils.model_definitions import MODELS

COLORMAP = 'viridis'

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    train_embeddings = data["train_embeddings"]
    train_meta_df = pd.DataFrame(data["train_meta"])
    test_embeddings = data["test_embeddings"]
    test_meta_df = pd.DataFrame(data["test_meta"])
    return train_embeddings, train_meta_df, test_embeddings, test_meta_df



def visualize_embedding_predictions(
        model_name,
        file_list,
        method='tsne',
        title='Embedding Visualization',
        random_state=101,
):
    class_dir = os.path.join(images_dir, 'class_by_rarity')
    figure_path = os.path.join(class_dir, f"{model_name}.png")

    if os.path.exists(figure_path):
        return

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    figsize=(15, 5 * len(file_list))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(file_list), 4, width_ratios=[1, 1, 0.3, 1], wspace=0.05, hspace=0.05)

    for i, file in enumerate(file_list):
        layer_num = int(file.split("layer_")[-1].split(".")[0])
        train_embeddings, train_meta_df, test_embeddings, test_meta_df = load_pkl(file)

        scaler = StandardScaler()
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        test_embeddings_scaled = scaler.transform(test_embeddings)

        embeddings_scaled = np.vstack([train_embeddings_scaled, test_embeddings_scaled])
        X_2d = reducer.fit_transform(embeddings_scaled)
        X_2d_test = X_2d[len(train_embeddings_scaled):]

        train_y_true = train_meta_df["label"]
        test_y_true = test_meta_df["label"]

        precision_zero_shot, recall_zero_shot, _ = precision_recall_curve(test_y_true, 1  - test_meta_df["cos_similarity"])
        ap_zero_shot = average_precision_score(test_y_true, 1  - test_meta_df["cos_similarity"])

        model = LogisticRegression(max_iter=5_000, solver='liblinear')
        model.fit(train_embeddings_scaled, train_y_true)

        y_prob = model.predict_proba(test_embeddings_scaled)[:,1]

        precision_class, recall_class, _ = precision_recall_curve(test_y_true, y_prob)
        ap_class = average_precision_score(test_y_true, y_prob)


        ax_true = fig.add_subplot(gs[i,0])
        ax_class = fig.add_subplot(gs[i,1])
        spacer = fig.add_subplot(gs[i,2])
        ax_auprc = fig.add_subplot(gs[i,3])

        spacer.axis("off")

        """
        Ground Truth
        """
        ax_true.scatter(X_2d_test[:, 0], X_2d_test[:, 1], c=test_y_true, cmap=COLORMAP, s=10)
        from matplotlib.lines import Line2D
        cmap = plt.cm.get_cmap(COLORMAP, 2)

        if i == 0:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Common', markerfacecolor=cmap(0), markersize=6),
                Line2D([0], [0], marker='o', color='w', label='Rare', markerfacecolor=cmap(1), markersize=6)
            ]

            ax_true.legend(
                handles=legend_elements,
                loc="lower right",
                frameon=False,
                fontsize=10,
                handlelength=1.5,
                labelspacing=0.2,
                markerscale=1.0
            )

        if i == 0:
            ax_true.set_title("t-SNE Colored by Labels")
        if i < len(file_list) - 1:
            ax_true.set_xticks([])
        if i == len(file_list) - 1:
            ax_true.set_xlabel("Dimension 1")

        ax_true.set_ylabel("Dimension 2")

        """
        Prediction by Classification
        """
        scatter = ax_class.scatter(
            X_2d_test[:, 0],
            X_2d_test[:, 1],
            c=y_prob,
            cmap=COLORMAP,
            s=10,
            vmin=0.0,
            vmax=1.0
        )
        ax_class.set_ylabel("")
        ax_class.set_yticks([])
        cax_gc = inset_axes(
            ax_class,
            width="5%",
            height="100%",
            loc='lower left',
            bbox_to_anchor=(1.01, 0.0, 1, 1),
            bbox_transform=ax_class.transAxes,
            borderpad=0
        )

        cbar = plt.colorbar(scatter, ax=ax_class, cax=cax_gc)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(['Common', 'Rare'])
        if i == 0:
            ax_class.set_title("t-SNE Colored by Regression")
        if i < len(file_list) - 1:
            ax_class.set_xticks([])
        if i == len(file_list) - 1:
            ax_class.set_xlabel("Dimension 1")

        ax_class.set_ylabel("")

        """
        PR Curve
        """
        auprc_cmap = plt.cm.get_cmap(COLORMAP)
        ax_auprc.plot(recall_zero_shot, precision_zero_shot, color=auprc_cmap(0.3), lw=2, label=f"auPRC Zero-Shot= {ap_zero_shot:.3f}")
        ax_auprc.plot(recall_class, precision_class, color=auprc_cmap(0.6), lw=2, label=f"auPRC Regression= {ap_class:.3f}")

        if i == 0:
            ax_auprc.set_title("Precision-Recall Curve")
        if i < len(file_list) - 1:
            ax_auprc.set_xticks([])
        if i == len(file_list) - 1:
            ax_auprc.set_xlabel("Recall")

        ax_auprc.set_ylabel("Precision")

        ax_auprc.text(1.01, 0.5, f"Layer {layer_num}", transform=ax_auprc.transAxes,
                 rotation=270, va='center', ha='left', fontsize=12)

        rect = patches.FancyBboxPatch(
            (1.00, -0.002),
            0.06, 1.0,
            transform=ax_auprc.transAxes,
            boxstyle="round,pad=0.00",
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False
        )
        ax_auprc.add_patch(rect)
        ax_auprc.legend()
        ax_auprc.grid(True)

    plt.suptitle(title, fontsize=16)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08)

    os.makedirs(class_dir, exist_ok=True)
    plt.savefig(figure_path)
    plt.show()

if __name__ == "__main__":
    embeddings_folder = '/shared/data/embeddings/5_utr_af_prediction/'

    for model in tqdm(MODELS):
        model_name = model['name']
        model_folder = os.path.join(embeddings_folder, model_name)
        try:
            files = sorted([os.path.join(model_folder, f) for f in os.listdir(model_folder)
                            if f.endswith(".pkl") and f.startswith("layer_")],
                           key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
        except:
            continue
        visualize_embedding_predictions(
            model_name=model_name,
            file_list=files,
            method='tsne',
            title=f"5'UTR SNV Rarity Classification - {model_name}"
        )