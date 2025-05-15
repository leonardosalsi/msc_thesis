from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Prediction ---
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap="viridis", s=10)

    # Create an inset axis for the colorbar
    cax = inset_axes(
        ax1,
        width="3%",  # width of colorbar
        height="100%",  # height of colorbar
        loc='lower left',
        bbox_to_anchor=(-0.12, 0.15, 1, 1),  # move it left with negative x
        bbox_transform=ax1.transAxes,
        borderpad=0,
    )

    plt.colorbar(scatter1, cax=cax, label=None)

    ax1.set_title("t-SNE Colored by Predicted Score")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")

    # --- Ground truth ---
    ax0 = axes[1]
    scatter0 = ax0.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="viridis", s=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Common', markerfacecolor='blue', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Rare', markerfacecolor='red', markersize=6)
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


    # --- PR curve ---
    ax2 = axes[2]
    ax2.plot(recall, precision, color="blue", lw=2, label=f"AUPRC = {ap:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    embeddings, meta_df = load_pkl("/shared/data/5_utr_embeddings_mean/default_logan_ewc_2/layer_11.pkl")

    y_true = meta_df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    model = Ridge()
    model.fit(X_scaled, meta_df["cos_similarity"])
    y_pred = model.predict(X_scaled)
    auprc = average_precision_score(y_true, y_pred)

    visualize_embedding_predictions(
        embeddings=embeddings,
        y_true=y_true,
        y_pred=y_pred,
        method='tsne',
        task='classification',
        title='t-SNE of Predicted Probabilities'
    )