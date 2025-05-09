import os
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from config import results_dir

def visualize_embeddings(model_name):
    model_results = os.path.join(results_dir, 'tSNE_embeddings', model_name)
    if not os.path.exists(model_results):
        raise f"Model results folder {model_results} does not exist."

    files = os.listdir(model_results)
    if len(files) == 0:
        raise f"Model results folder {model_results} is empty."

    print(files)
    return

    with open("data/mean_pooled_embeddings_layer5.pkl", "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    meta = data["meta"]

    # Optional: convert meta to DataFrame
    df = pd.DataFrame(meta)
    df["tSNE1"], df["tSNE2"] = zip(*TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(embeddings))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="tSNE1", y="tSNE2", hue="label", s=10)
    plt.title("t-SNE of Mean-Pooled Embeddings (Layer 5)")
    plt.tight_layout()
    plt.show()

    for p in perp:
    # Run t-SNE on embeddings
        pca = PCA(n_components=50)
        reduced = pca.fit_transform(cls_embeddings)
        tsne_results = TSNE(n_components=2, perplexity=p, random_state=42, init='pca').fit_transform(reduced)

        # Build dataframe for plotting
        label_order = ["3UTR", "CDS", "intron", "intergenic", "5UTR"]
        df = pd.DataFrame(tsne_results, columns=["tSNE1", "tSNE2"])
        df["Sequence"] = [m["sequence"] for m in meta]
        df["label"] = [m["label"] for m in meta]

        # Visualize
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="tSNE1", y="tSNE2", hue="label", hue_order=label_order, s=10)
        plt.title("t-SNE of Layer 5 CLS Embeddings (5′ UTR Analysis)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    pass