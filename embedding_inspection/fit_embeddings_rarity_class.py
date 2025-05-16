import os
from matplotlib import gridspec, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        random_state=101,
):

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    figsize=(25, 5 * len(file_list))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(file_list), 5, width_ratios=[1, 1, 1, 0.2, 1], wspace=0.05, hspace=0.05)

    for i, file in enumerate(file_list):
        layer_num = int(file.split("layer_")[-1].split(".")[0])
        embeddings, meta_df = load_pkl(file)

        scaler = StandardScaler()
        X_2d = reducer.fit_transform(embeddings)
        X_scaled = scaler.fit_transform(embeddings)

        y_true = meta_df["label"]

        model_class = LogisticRegression(max_iter=1000)
        model_class.fit(X_scaled, meta_df["label"])

        y_prob_class = model_class.predict_proba(X_scaled)[:, 1]
        y_pred_class = (y_prob_class >= 0.5).astype(int)
        precision_class, recall_class, _ = precision_recall_curve(y_true, y_pred_class)
        ap_class = average_precision_score(y_true, y_pred_class)

        model_prob = Ridge()
        model_prob.fit(X_scaled, meta_df["dot_product_norm"] - 1)
        y_pred_prob = model_prob.predict(X_scaled)
        y_pred_prob_norm = MinMaxScaler().fit_transform(y_pred_prob.reshape(-1, 1)).flatten()
        precision_prob, recall_prob, _ = precision_recall_curve(y_true, y_pred_prob)
        ap_prob = average_precision_score(y_true, y_pred_prob)

        ax_true = fig.add_subplot(gs[i,0])
        ax_class = fig.add_subplot(gs[i,1])
        ax_prob = fig.add_subplot(gs[i, 2])
        spacer = fig.add_subplot(gs[i,3])
        ax_auprc = fig.add_subplot(gs[i,4])
        spacer.axis("off")

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

        scatter1 = ax_prob.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred_prob_norm, cmap=COLORMAP, s=10)
        cax = inset_axes(ax_prob,
                         width="5%",
                         height="100%",
                         loc='lower left',
                         bbox_to_anchor=(1.01, 0.0, 1, 1),
                         bbox_transform=ax_prob.transAxes,
                         borderpad=0)

        plt.colorbar(scatter1, cax=cax, label=None)
        ax_prob.set_ylabel("")
        ax_prob.set_yticks([])
        if i == 0:
            ax_prob.set_title("t-SNE Colored by Regression")
        if i < len(file_list) - 1:
            ax_prob.set_xticks([])
        if i == len(file_list) - 1:
            ax_prob.set_xlabel("Dimension 1")

        """
        PR Curve
        """
        auprc_cmap = plt.cm.get_cmap(COLORMAP)
        ax_auprc.plot(recall_class, precision_class, color=auprc_cmap(0.1), lw=2, label=f"AUPRC Classfication= {ap_class:.3f}")
        ax_auprc.plot(recall_prob, precision_prob, color=auprc_cmap(0.9), lw=2, label=f"AUPRC Regression= {ap_prob:.3f}")
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
    class_dir = os.path.join(images_dir, 'class_by_rarity')
    os.makedirs(class_dir, exist_ok=True)
    plt.savefig(os.path.join(class_dir, f"{model_name}.png"))
    plt.show()

if __name__ == "__main__":
    embeddings_folder = '/shared/data/embeddings/5_utr_6000'
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