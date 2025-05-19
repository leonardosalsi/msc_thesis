import os

import numpy as np
from matplotlib import gridspec, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import f1_score
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from config import images_dir

COLORMAP = 'viridis'

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    meta_df = pd.DataFrame(data["meta"])
    train_embeddings = data["train_embeddings"]
    train_meta_df = pd.DataFrame(data["train_meta"])
    test_embeddings = data["test_embeddings"]
    test_meta_df = pd.DataFrame(data["test_meta"])
    return embeddings, meta_df, train_embeddings, train_meta_df, test_embeddings, test_meta_df

def visualize_embedding_predictions(
        model_name,
        file_list,
        method='tsne',
        title='Embedding Visualization',
        random_state=101,
):
    class_dir = os.path.join(images_dir, 'class_by_rarity')
    figure_path = os.path.join(class_dir, f"{model_name}.png")

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    figsize=(15, 5 * len(file_list))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(file_list), 3, width_ratios=[1, 1, 1], wspace=0.05, hspace=0.05)

    for i, file in enumerate(file_list):
        layer_num = int(file.split("layer_")[-1].split(".")[0])
        embeddings, meta_df, train_embeddings, train_meta_df, test_embeddings, test_meta_df = load_pkl(file)

        scaler = StandardScaler()

        embeddings_scaled = scaler.fit_transform(embeddings)
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        test_embeddings_scaled = scaler.fit_transform(test_embeddings)

        X_2d = reducer.fit_transform(embeddings_scaled)

        y_true = meta_df["label"]
        train_y_true = train_meta_df["label"]
        test_y_true = test_meta_df["label"]

        precision_zero_shot, recall_zero_shot, _ = precision_recall_curve(y_true, 1  - meta_df["cos_similarity"])
        ap_zero_shot = average_precision_score(y_true, 1  - meta_df["cos_similarity"])

        model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=300)
        model.fit(train_embeddings_scaled, train_y_true)

        y_prob_class = model.predict_proba(test_embeddings_scaled)[:, 1]

        precision_class, recall_class, _ = precision_recall_curve(test_y_true, y_prob_class)
        ap_class = average_precision_score(test_y_true, y_prob_class)

        best_f1 = 0
        best_thresh = 0
        for t in np.linspace(0.0, 1.0, 101):
            y_pred = (y_prob_class >= t).astype(int)
            f1 = f1_score(test_y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        print("BEST THRESH: ", best_thresh)

        y_pred_class = (y_prob_class >= best_thresh).astype(int)



        ax_true = fig.add_subplot(gs[i,0])
        ax_class = fig.add_subplot(gs[i,1])
        ax_auprc = fig.add_subplot(gs[i,2])

        """
        Ground Truth
        """
        ax_true.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap=COLORMAP, s=10)
        from matplotlib.lines import Line2D
        cmap = plt.cm.get_cmap(COLORMAP, 2)

        if i == 0:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Rare', markerfacecolor=cmap(1), markersize=6),
                Line2D([0], [0], marker='o', color='w', label='Common', markerfacecolor=cmap(0), markersize=6)
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
        ax_class.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred_class, cmap=COLORMAP, s=10)
        ax_class.set_ylabel("")
        ax_class.set_yticks([])

        if i == 0:
            ax_class.set_title("t-SNE Colored by Classification")
        if i < len(file_list) - 1:
            ax_class.set_xticks([])
        if i == len(file_list) - 1:
            ax_class.set_xlabel("Dimension 1")

        ax_class.set_ylabel("")

        """
        Probability by Regression
        """



        """
        PR Curve
        """
        auprc_cmap = plt.cm.get_cmap(COLORMAP)
        ax_auprc.plot(recall_zero_shot, precision_zero_shot, color=auprc_cmap(0.1), lw=2, label=f"AUPRC Zero-Shot= {ap_zero_shot:.3f}")
        ax_auprc.plot(recall_class, precision_class, color=auprc_cmap(0.1), lw=2, label=f"AUPRC Classfication= {ap_class:.3f}")

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
    exit()

if __name__ == "__main__":
    embeddings_folder = '/shared/data/embeddings/5_utr_6000'
    model_name = 'overlap_logan_ewc_5'
    model_folder = os.path.join(embeddings_folder, model_name)
    files = sorted([os.path.join(model_folder, f) for f in os.listdir(model_folder)
                    if f.endswith(".pkl") and f.startswith("layer_")],
                   key=lambda f: int(f.split("layer_")[-1].split(".")[0]))

    visualize_embedding_predictions(
        model_name=model_name,
        file_list=files,
        method='tsne',
        title=f"5'UTR SNV Rarity Classification - {model_name}"
    )