from pprint import pprint

from matplotlib import gridspec, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

COLORMAP = "viridis"

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    meta_df = pd.DataFrame(data["meta"])
    return embeddings, meta_df

def visualize_embedding_predictions(
    embeddings,
    y_true=None,
    y_pred=None,
    method='tsne',
    task='regression',
    title='Embedding Visualization',
    n_components=2,
    random_state=101,
    figsize=(18, 6),  # Make room for 3 plots
):
    assert y_pred is not None and y_true is not None, "You must provide both predictions and ground truth."

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    X_2d = reducer.fit_transform(embeddings)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.3, 1], wspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    spacer = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[3])

    spacer.axis("off")
    # --- Ground truth ---

    ax0.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap=COLORMAP, s=10)
    from matplotlib.lines import Line2D
    cmap = plt.cm.get_cmap(COLORMAP, 2)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Common', markerfacecolor=cmap(0), markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Rare', markerfacecolor=cmap(1), markersize=6)
    ]

    ax0.legend(
        handles=legend_elements,
        title="Ground Truth",
        loc="lower right",  # or adjust to your preference
        frameon=True,
        fontsize=9
    )

    ax0.set_title("t-SNE Colored by Ground Truth")
    ax0.set_xlabel("Component 1")
    ax0.set_ylabel("Component 2")

    # --- Prediction ---
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap=COLORMAP, s=10)

    cax = inset_axes(ax1,
                     width="5%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.01, 0.0, 1, 1),
                     bbox_transform=ax1.transAxes,
                     borderpad=0)

    plt.colorbar(scatter1, cax=cax, label=None)
    pos = ax1.get_position()
    ax1.set_ylabel("")
    ax1.set_yticks([])
    ax1.set_title("t-SNE Colored by Predicted Score")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("")

    # --- PR curve ---
    ax2.plot(recall, precision, color="blue", lw=2, label=f"AUPRC = {ap:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.text(1.01, 0.5, f"Layer 3", transform=ax2.transAxes,
            rotation=270, va='center', ha='left', fontsize=12)

    rect = patches.FancyBboxPatch(
        (1.00, -0.002),  # x, y (in axis coordinates)
        0.06, 1.0,  # width, height
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

    plt.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.12)
    plt.show()


if __name__ == "__main__":
    embeddings, meta_df = load_pkl("/shared/data/5_utr_embeddings_mean/default_logan_ewc_2/layer_0.pkl")

    y_true = meta_df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, meta_df["label"])

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auprc = average_precision_score(y_true, y_pred)

    visualize_embedding_predictions(
        embeddings=embeddings,
        y_true=y_true,
        y_pred=y_pred,
        method='tsne',
        task='classification',
        title='t-SNE of Predicted Probabilities'
    )