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
        figsize=(15, 6),
):
    assert y_pred is not None, "You must provide predictions for visualization."

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")
    X_2d = reducer.fit_transform(embeddings)

    # Prepare PR data
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # t-SNE Scatter
    ax1 = axes[0]
    scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap="viridis", s=10)
    fig.colorbar(scatter, ax=ax1, label="Predicted Score")
    ax1.set_title("t-SNE Colored by Predicted Score")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")

    # Precision-Recall Curve
    ax2 = axes[1]
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
    embeddings, meta_df = load_pkl("/shared/data/5_utr_embeddings_mean/default_logan_ewc_2/layer_0.pkl")

    y_true = meta_df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    model = Ridge()
    model.fit(X_scaled, 1 - meta_df["dot_product_norm"])  # Or whatever your target is
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