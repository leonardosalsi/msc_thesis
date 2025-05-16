import os
import pickle
import pandas as pd
from matplotlib import patches, gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from config import results_dir, images_dir
from utils.model_definitions import MODELS

COLORMAP = 'viridis'

def visualize_embeddings(model_name):
    class_dir = os.path.join(images_dir, 'class_by_region')
    figure_path = os.path.join(class_dir, f"{model_name}.png")

    if os.path.exists(figure_path):
        return

    embedding_dir = os.path.join(results_dir, 'embeddings', 'genomic_elements_6000', model_name)
    tsne_dir = os.path.join(results_dir, 'tSNE', 'genomic_elements_6000', model_name)

    try:
        embedding_files = sorted([os.path.join(embedding_dir, f) for f in os.listdir(embedding_dir) if f.endswith(".pkl") and f.startswith("layer") ],
                       key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
        tsne_files = sorted([os.path.join(tsne_dir, f) for f in os.listdir(tsne_dir) if
                                  f.endswith(".pkl") and f.startswith("layer")],
                                 key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
    except:
        return

    assert len(embedding_files) == len(tsne_files)
    n = len(embedding_files)

    figsize = (18, 4 * len(embedding_files))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(embedding_files), 5, width_ratios=[1, 0.1, 1, 0.1, 1], wspace=0.05, hspace=0.05)


    for i, (embedding_file, tsne_file) in enumerate(zip(embedding_files, tsne_files)):
        layer = int(embedding_file.split("layer_")[-1].split(".")[0])

        with open(embedding_file, "rb") as f:
            data = pickle.load(f)

        meta = data["meta"]
        df = pd.DataFrame(meta)

        with open(tsne_file, "rb") as f:
            X_2d = pickle.load(f)

        ax_len = fig.add_subplot(gs[i, 0])
        spacer_1 = fig.add_subplot(gs[i, 1])
        ax_gc = fig.add_subplot(gs[i, 2])
        spacer_2 = fig.add_subplot(gs[i, 3])
        ax_class = fig.add_subplot(gs[i, 4])

        spacer_1.axis("off")
        spacer_2.axis("off")

        df["Dimension 1"] = X_2d[:, 0]
        df["Dimension 2"] = X_2d[:, 1]

        """
        Color by GC-content
        """
        gc_values = df["GC"]
        sc_gc = ax_gc.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=gc_values,
            cmap="viridis", 
            s=0.1
        )

        cax_gc = inset_axes(
            ax_gc,
            width="5%",
            height="100%",
            loc='lower left',
            bbox_to_anchor=(1.01, 0.0, 1, 1),
            bbox_transform=ax_gc.transAxes,
            borderpad=0
        )

        cbar = plt.colorbar(sc_gc, ax=ax_gc, cax=cax_gc)
        cbar.set_label('')

        ax_gc.set_yticks([])
        ax_gc.set_ylabel("")
        if i == 0:
            ax_gc.set_title("t-SNE Colored by GC Content")
        if i < len(embedding_files) - 1:
            ax_gc.set_xticks([])
            ax_gc.set_xlabel("")
        if i == len(embedding_files) - 1:
            ax_gc.set_xlabel("Dimension 1")

        """
        Color by Region Length
        """

        reg_len = df["reg_len"]
        sc_len = ax_len.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=reg_len,
            cmap="viridis", 
            s=0.1
        )

        cax_len = inset_axes(
            ax_len,
            width="5%",
            height="100%",
            loc='lower left',
            bbox_to_anchor=(1.01, 0.0, 1, 1),
            bbox_transform=ax_len.transAxes,
            borderpad=0
        )

        cbar = plt.colorbar(sc_len, ax=ax_len, cax=cax_len)
        cbar.set_label('')

        ax_len.set_ylabel("Dimension 2")
        if i == 0:
            ax_len.set_title("t-SNE Colored by Region Length")
        if i < len(embedding_files) - 1:
            ax_len.set_xticks([])
            ax_len.set_xlabel("")
        if i == len(embedding_files) - 1:
            ax_len.set_xlabel("Dimension 1")

        """
        Region Classification
        """

        class_colormap = 'tab10'
        label_order = ["3UTR", "CDS", "intron", "intergenic", "5UTR"]
        sns.scatterplot(
            data=df,
            x="Dimension 1",
            y="Dimension 2",
            hue="label",
            hue_order=label_order,
            s=5,
            ax=ax_class,
            palette=sns.color_palette(class_colormap, n_colors=len(label_order)),
            legend=(i == len(embedding_files) - 1)
        )

        if i == len(embedding_files) - 1:
            handles, labels = ax_class.get_legend_handles_labels()
            ax_class.legend(
                handles, labels,
                loc='lower left',
                frameon=False,
                fontsize=10,
                handlelength=1.5,
                labelspacing=0.2,
                markerscale=2.0
            )

        ax_class.set_ylabel("")
        ax_class.set_yticks([])
        if i == 0:
            ax_class.set_title("t-SNE Colored by Genomic Region")
        if i < len(embedding_files) - 1:
            ax_class.set_xticks([])
            ax_class.set_xlabel("")
        if i == len(embedding_files) - 1:
            ax_class.set_xlabel("Dimension 1")

        ax_class.text(1.01, 0.5, f"Layer {layer + 1}", transform=ax_class.transAxes,
                      rotation=270, va='center', ha='left', fontsize=12)

        rect = patches.FancyBboxPatch(
            (1.00, -0.002),
            0.06, 1.0,
            transform=ax_class.transAxes,
            boxstyle="round,pad=0.00",
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False
        )
        ax_class.add_patch(rect)

    plt.subplots_adjust(hspace=0.02, bottom=0.04, top=0.93, left=0.05, right=0.98)
    plt.suptitle(f"Genomic Region Determination - {model_name}", fontsize=16)

    os.makedirs(class_dir, exist_ok=True)
    plt.savefig(figure_path)
    plt.show()

if __name__ == "__main__":
    for model in tqdm(MODELS):
        visualize_embeddings(model['name'])