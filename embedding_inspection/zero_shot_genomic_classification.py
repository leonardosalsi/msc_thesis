import json
import os
import pickle
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import patches, gridspec, MatplotlibDeprecationWarning
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, label_binarize
from tqdm import tqdm
import seaborn as sns
from config import results_dir, images_dir, cache_dir
from utils.model_definitions import MODELS

COLORMAP = 'viridis'

def visualize_embeddings(model_name):
    class_dir = os.path.join(images_dir, 'class_by_region')
    figure_path = os.path.join(class_dir, f"{model_name}.png")

    if os.path.exists(figure_path):
        return

    embedding_dir = os.path.join(results_dir, 'embeddings', 'genomic_regions_annotated', model_name)
    tsne_dir = os.path.join(results_dir, 'tSNE', 'genomic_regions_annotated', model_name)

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

import torch.nn.functional as F

warnings.filterwarnings(
    "ignore",
    category=MatplotlibDeprecationWarning,
    message=".*get_cmap function was deprecated.*"
)

INTERMEDIATE_FILE_CACHE = os.path.join(cache_dir, f"genomic_regions_annotated")
os.makedirs(INTERMEDIATE_FILE_CACHE, exist_ok=True)

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    train_embeddings = data["train_embeddings"]
    train_meta_df = pd.DataFrame(data["train_meta"])
    test_embeddings = data["test_embeddings"]
    test_meta_df = pd.DataFrame(data["test_meta"])
    return train_embeddings, train_meta_df, test_embeddings, test_meta_df

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    else:
        return obj

def zero_shot_prediction(train_embeddings, train_labels, test_embeddings, test_labels):
    centroids = {}
    for cls in np.unique(train_labels):
        c = train_embeddings[train_labels == cls].mean(axis=0)
        centroids[cls] = c / np.linalg.norm(c)
    classes = sorted(centroids)
    cent_vecs = np.vstack([centroids[c] for c in classes])
    sims = test_embeddings.dot(cent_vecs.T)
    Y_test_bin = label_binarize(test_labels, classes=classes)
    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}
    for idx, cls in enumerate(classes):
        scores = sims[:, idx]
        precision, recall, _ = precision_recall_curve(Y_test_bin[:, idx], scores)
        ap = average_precision_score(Y_test_bin[:, idx], scores)
        ap_per_class[cls] = ap
        precision_per_class[cls] = precision
        recall_per_class[cls] = recall

    return ap_per_class, precision_per_class, recall_per_class, classes

def few_shot_prediction(train_embeddings, train_labels, test_embeddings, test_labels):
    model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    model.fit(train_embeddings, train_labels)
    y_prob = model.predict_proba(test_embeddings)
    classes = model.classes_
    y_test_bin = label_binarize(test_labels, classes=classes)
    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}

    for idx, cls in enumerate(classes):
        scores = y_prob[:, idx]
        precision, recall, _ =  precision_recall_curve(y_test_bin[:, idx], scores)
        ap = average_precision_score(y_test_bin[:, idx], scores)
        ap_per_class[cls] = ap
        precision_per_class[cls] = precision
        recall_per_class[cls] = recall

    return ap_per_class, precision_per_class, recall_per_class, classes

def evaluate_class_prediction(model_name, file_list):
    class_dir = os.path.join(images_dir, 'class_genomic_regions')
    figure_path = os.path.join(class_dir, f"{model_name}.pdf")

    if os.path.exists(figure_path):
        return

    results = {file.split("layer_")[-1].split(".")[0]: {} for file in file_list}
    all_results = []

    for i, file in enumerate(file_list):
        layer_num = int(file.split("layer_")[-1].split(".")[0])
        train_embeddings, train_meta, test_embeddings, test_meta = load_pkl(file)

        train_embeddings_normalized = normalize(train_embeddings, axis=1)
        test_embeddings_normalized = normalize(test_embeddings, axis=1)

        train_y_true = train_meta["label"]
        test_y_true = test_meta["label"]

        intermediate_results_file = os.path.join(INTERMEDIATE_FILE_CACHE, f"{model_name}_{layer_num}.json")

        if os.path.exists(intermediate_results_file):
            with open(intermediate_results_file, "r") as f:
                intermediate_results = json.load(f)
        else:
            zero_shot_res = zero_shot_prediction(train_embeddings_normalized, train_y_true, test_embeddings_normalized,
                                                 test_y_true)
            ap_per_class_zero_shot, precision_per_class_zero_shot, recall_per_class_zero_shot, classes_zero_shot = zero_shot_res

            few_shot_res = few_shot_prediction(train_embeddings_normalized, train_y_true, test_embeddings_normalized, test_y_true)
            ap_per_class_few_shot, precision_per_class_few_shot, recall_per_class_few_shot, classes_few_shot = few_shot_res
            pprint(ap_per_class_zero_shot)
            intermediate_results = {
                'zero_shot': {
                    'ap_per_class_zero_shot': ap_per_class_zero_shot,
                    'precision_per_class_zero_shot': precision_per_class_zero_shot,
                    'recall_per_class_zero_shot': recall_per_class_zero_shot,
                    'classes_zero_shot': classes_zero_shot,
                },
                'few_shot': {
                    'ap_per_class_few_shot': ap_per_class_few_shot,
                    'precision_per_class_few_shot': precision_per_class_few_shot,
                    'recall_per_class_few_shot': recall_per_class_few_shot,
                    'classes_few_shot': classes_few_shot,
                }
            }
            intermediate_results = make_json_safe(intermediate_results)
            with open(intermediate_results_file, "w") as f:
                json.dump(intermediate_results, f)

        zero_shot_res = intermediate_results["zero_shot"]
        few_shot_res = intermediate_results["few_shot"]
        precision_per_class_zero_shot = zero_shot_res["precision_per_class_zero_shot"]
        precision_per_class_few_shot = few_shot_res["precision_per_class_few_shot"]

        for a in precision_per_class_zero_shot:
            all_results += precision_per_class_zero_shot[a]
        for a in precision_per_class_few_shot:
            all_results += precision_per_class_few_shot[a]
        results[str(layer_num)] = intermediate_results

    ymin = min(all_results)
    ymax = max(all_results)
    padding = (ymax - ymin) * 0.1
    ymin, ymax = ymin - padding, ymax + padding

    figsize = (5 * len(results), 5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(results), width_ratios=[1] * len(results), wspace=0.05, hspace=0.05)

    for i, result in enumerate(results):
        layer = result

        zero_shot_res = intermediate_results["zero_shot"]
        few_shot_res = intermediate_results["few_shot"]
        ap_per_class_zero_shot = np.array(zero_shot_res["ap_per_class_zero_shot"])
        precision_per_class_zero_shot = np.array(zero_shot_res["precision_per_class_zero_shot"])
        recall_per_class_zero_shot = np.array(zero_shot_res["recall_per_class_zero_shot"])
        classes_zero_shot = np.array(zero_shot_res["classes_zero_shot"])

        ap_per_class_few_shot = np.array(few_shot_res["ap_per_class_few_shot"])
        precision_per_class_few_shot = np.array(few_shot_res["precision_per_class_few_shot"])
        recall_per_class_few_shot = np.array(few_shot_res["recall_per_class_few_shot"])
        classes_few_shot = np.array(few_shot_res["classes_few_shot"])


        exit()
        ax_auprc = fig.add_subplot(gs[i])
        ax_auprc.set_ylim(ymin, ymax)
        auprc_cmap = plt.cm.get_cmap(COLORMAP)
        ax_auprc.plot(recall_zero_shot, precision_zero_shot, color=auprc_cmap(0.3), lw=2,
                      label=f"auPRC Zero-Shot= {ap_zero_shot:.3f}")
        ax_auprc.plot(recall_few_shot, precision_few_shot, color=auprc_cmap(0.6), lw=2,
                      label=f"auPRC Regression= {ap_few_shot:.3f}")

        if i == 0:
            ax_auprc.set_ylabel("Precision", fontsize=14)
        else:
            ax_auprc.set_yticklabels([])

        ax_auprc.tick_params(left=False, bottom=False)

        rect = patches.FancyBboxPatch(
            (0.0, 1.0),
            1.0, 0.12,
            boxstyle="round,pad=0.00",
            transform=ax_auprc.transAxes,
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False
        )
        ax_auprc.text(
            0.5, 1.05, f"Layer {int(layer) + 1}",
            transform=ax_auprc.transAxes,
            ha='center',
            va='center',
            fontsize=16
        )
        ax_auprc.add_patch(rect)
        ax_auprc.legend()
        ax_auprc.grid(True)

if __name__ == "__main__":
    embeddings_folder = os.path.join(results_dir, 'embeddings', 'genomic_regions_annotated')
    for model_name in MODELS:
        model_folder = os.path.join(embeddings_folder, model_name)
        try:
            files = sorted([os.path.join(model_folder, f) for f in os.listdir(model_folder)
                            if f.endswith(".pkl") and f.startswith("layer_")],
                           key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
        except:
            continue
        evaluate_class_prediction(model_name, files)