import os
from matplotlib import gridspec, patches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config import images_dir

COLORMAP = 'viridis'

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    meta_df = pd.DataFrame(data["meta"])
    return embeddings, meta_df


def visualize_embedding_predictions(
        model_name,
        file_list,
        method='tsne',
        title='Embedding Visualization',
        random_state=101,# Make room for 3 plots
):

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    figsize=(20, 5 * len(file_list))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(file_list), 4, width_ratios=[1, 1, 0.1, 1], wspace=0.05, hspace=0.05)

    for i, file in enumerate(file_list):
        layer_num = int(file.split("layer_")[-1].split(".")[0])
        embeddings, meta_df = load_pkl(file)

        y_true = meta_df["label"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, meta_df["label"])

        y_prob = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        X_2d = reducer.fit_transform(embeddings)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        ax0 = fig.add_subplot(gs[i,0])
        ax1 = fig.add_subplot(gs[i,1])
        spacer = fig.add_subplot(gs[i,2])
        ax2 = fig.add_subplot(gs[i,3])
        spacer.axis("off")

        """
        Ground Truth
        """
        ax0.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap=COLORMAP, s=10)
        from matplotlib.lines import Line2D
        cmap = plt.cm.get_cmap(COLORMAP, 2)

        if i == 0:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Common', markerfacecolor=cmap(0), markersize=6),
                Line2D([0], [0], marker='o', color='w', label='Rare', markerfacecolor=cmap(1), markersize=6)
            ]

            ax0.legend(
                handles=legend_elements,
                loc="lower right",
                frameon=False,
                fontsize=10,
                handlelength=1.5,
                labelspacing=0.2,
                markerscale=1.0
            )

        if i == 0:
            ax0.set_title("t-SNE Colored by Labels")
        if i < len(file_list) - 1:
            ax0.set_xticks([])
        if i == len(file_list) - 1:
            ax0.set_xlabel("Dimension 1")

        ax0.set_ylabel("Dimension 2")

        """
        Prediction by Regressor
        """
        ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap=COLORMAP, s=10)
        ax1.set_ylabel("")
        ax1.set_yticks([])

        if i == 0:
            ax1.set_title("t-SNE Colored by Classification")
        if i < len(file_list) - 1:
            ax1.set_xticks([])
        if i == len(file_list) - 1:
            ax1.set_xlabel("Dimension 1")

        ax1.set_ylabel("")

        """
        PR Curve
        """
        ax2.plot(recall, precision, color="blue", lw=2, label=f"AUPRC = {ap:.3f}")

        if i == 0:
            ax2.set_title("Precision-Recall Curve")
        if i < len(file_list) - 1:
            ax2.set_xticks([])
        if i == len(file_list) - 1:
            ax2.set_xlabel("Recall")

        ax2.set_ylabel("Precision")

        ax2.text(1.01, 0.5, f"Layer {layer_num}", transform=ax2.transAxes,
                 rotation=270, va='center', ha='left', fontsize=12)

        rect = patches.FancyBboxPatch(
            (1.00, -0.002),
            0.06, 1.0,
            transform=ax2.transAxes,
            boxstyle="round,pad=0.00",
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False
        )
        ax2.add_patch(rect)
        ax2.legend()
        ax2.grid(True)

    plt.suptitle(title, fontsize=16)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08)
    class_dir = os.path.join(images_dir, 'class_by_rarity')
    os.makedirs(class_dir, exist_ok=True)
    plt.savefig(os.path.join(class_dir, f"{model_name}.png"))
    plt.show()

if __name__ == "__main__":
    embeddings_folder = '/shared/data/5_utr_embeddings_mean'
    model_names = os.listdir(embeddings_folder)

    for model_name in model_names:
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