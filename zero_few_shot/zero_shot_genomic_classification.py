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
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, label_binarize
from tqdm import tqdm
import seaborn as sns
from config import results_dir, images_dir, cache_dir
from utils.model_definitions import MODELS

COLORMAP = 'viridis'

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
        precision, recall, _ = precision_recall_curve(Y_test_bin[:, idx], scores, drop_intermediate=False)
        ap = average_precision_score(Y_test_bin[:, idx], scores)
        ap_per_class[cls] = ap
        precision_per_class[cls] = precision
        recall_per_class[cls] = recall

    return ap_per_class, precision_per_class, recall_per_class, classes

def few_shot_prediction(train_embeddings, train_labels, test_embeddings, test_labels):
    model = LogisticRegression(max_iter=5_000, solver='liblinear')
    model.fit(train_embeddings, train_labels)
    y_prob = model.predict_proba(test_embeddings)
    classes = model.classes_
    y_test_bin = label_binarize(test_labels, classes=classes)
    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}

    for idx, cls in enumerate(classes):
        scores = y_prob[:, idx]
        precision, recall, _ =  precision_recall_curve(y_test_bin[:, idx], scores, drop_intermediate=False)
        ap = average_precision_score(y_test_bin[:, idx], scores)
        ap_per_class[cls] = ap
        precision_per_class[cls] = precision
        recall_per_class[cls] = recall

    return ap_per_class, precision_per_class, recall_per_class, classes

def evaluate_class_prediction(model_name, file_list):
    class_dir = os.path.join(images_dir, 'class_genomic_regions')
    os.makedirs(class_dir, exist_ok=True)
    figure_path = os.path.join(class_dir, f"gen_region_{model_name}_tsne.pdf")
    figure_path_auprc = os.path.join(class_dir, f"gen_region_{model_name}_auprc.pdf")
    tsne_dir = os.path.join(results_dir, 'tSNE', 'genomic_regions_annotated', model_name)

    if os.path.exists(figure_path):
        return

    results = {file.split("layer_")[-1].split(".")[0]: {} for file in file_list}
    all_results = []
    last_layer = 0

    for i, file in enumerate(file_list):
        layer_num = int(file.split("layer_")[-1].split(".")[0])
        train_embeddings, train_meta, test_embeddings, test_meta = load_pkl(file)

        train_embeddings_normalized = normalize(train_embeddings, axis=1)
        test_embeddings_normalized = normalize(test_embeddings, axis=1)

        train_y_true = train_meta["label"]
        test_y_true = test_meta["label"]

        tsne_file = os.path.join(tsne_dir, f"layer_{layer_num}.pkl")
        if not os.path.exists(tsne_file):
            tsne_results = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(test_embeddings)
            with open(tsne_file, "wb") as f:
                pickle.dump(tsne_results, f)

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
        last_layer = str(layer_num)

    ymin = min(all_results)
    ymax = max(all_results)
    padding = (ymax - ymin) * 0.1
    ymin, ymax = ymin - padding, ymax + padding

    last_layer_result = results[last_layer]
    tsne_file = os.path.join(tsne_dir, f"layer_{last_layer}.pkl")
    with open(tsne_file, "rb") as f:
        X_2d = pickle.load(f)

    _, _, _, test_meta = load_pkl(file_list[-1])
    zero_shot_res = last_layer_result["zero_shot"]
    few_shot_res = last_layer_result["few_shot"]
    ap_per_class_zero_shot = zero_shot_res["ap_per_class_zero_shot"]
    precision_per_class_zero_shot = zero_shot_res["precision_per_class_zero_shot"]
    recall_per_class_zero_shot = zero_shot_res["recall_per_class_zero_shot"]

    ap_per_class_few_shot = few_shot_res["ap_per_class_few_shot"]
    precision_per_class_few_shot = few_shot_res["precision_per_class_few_shot"]
    recall_per_class_few_shot = few_shot_res["recall_per_class_few_shot"]

    figsize = (25, 8)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 0.02, 1, 0.2, 1], wspace=0.05, hspace=0.05)

    ax_class = fig.add_subplot(gs[0])
    ax_gc = fig.add_subplot(gs[2])
    ax_length = fig.add_subplot(gs[4])
    spacer_1 = fig.add_subplot(gs[3])
    spacer_1.axis('off')
    spacer_1 = fig.add_subplot(gs[1])
    spacer_1.axis('off')

    df = pd.DataFrame(test_meta)
    df["Dimension 1"] = X_2d[:, 0]
    df["Dimension 2"] = X_2d[:, 1]

    """
    Plot Actual Classes
    """
    class_colormap = 'tab10'
    label_order = ["3UTR", "CDS", "intron", "intergenic", "5UTR"]
    sc = sns.scatterplot(
        data=df,
        x="Dimension 1",
        y="Dimension 2",
        hue="label",
        hue_order=label_order,
        s=5,
        ax=ax_class,
        palette=sns.color_palette(class_colormap, n_colors=len(label_order)),
        legend=True
    )
    ax_class.set_title("Genomic Classes", fontsize=24)
    ax_class.set_xlabel("")
    ax_class.set_ylabel("Dimension 2", fontsize=20)
    handles, labels = ax_class.get_legend_handles_labels()
    ax_class.legend(handles=handles[0:], labels=labels[0:], markerscale=4, fontsize=14)
    ax_class.set_xticks(ax_class.get_xticks())
    ax_class.set_xticklabels(ax_class.get_xticklabels(), fontsize=14)
    ax_class.set_yticks(ax_class.get_yticks())
    ax_class.set_yticklabels(ax_class.get_yticklabels(), fontsize=14)
    """
    Plot By GC content
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
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('')

    ax_gc.set_yticks([])
    ax_gc.set_ylabel("")
    ax_gc.set_title("Colored by GC content", fontsize=24)
    ax_gc.set_xlabel("Dimension 1", fontsize=20)
    ax_gc.set_xticks(ax_gc.get_xticks())
    ax_gc.set_xticklabels(ax_gc.get_xticklabels(), fontsize=14)
    """
    Plot By Sequence Length
    """
    reg_len = df["reg_len"]
    sc_len = ax_length.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=reg_len,
        cmap="viridis",
        s=0.1
    )

    cax_len = inset_axes(
        ax_length,
        width="5%",
        height="100%",
        loc='lower left',
        bbox_to_anchor=(1.01, 0.0, 1, 1),
        bbox_transform=ax_length.transAxes,
        borderpad=0
    )

    cbar = plt.colorbar(sc_len, ax=ax_length, cax=cax_len)
    cbar.set_label('')
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
    ax_length.set_yticks([])
    ax_length.set_ylabel("")
    ax_length.set_title("Colored by Region Length", fontsize=24)
    ax_length.set_xlabel("")
    ax_length.set_xticks(ax_length.get_xticks())
    ax_length.set_xticklabels(ax_length.get_xticklabels(), fontsize=14)
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.10,
        top=0.95,
    )

    plt.savefig(figure_path, dpi=300)
    plt.show()
    plt.close(fig)

    figsize = (16, 8)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05, hspace=0.05)

    ax_auprc_zero = fig.add_subplot(gs[0])
    ax_auprc_few = fig.add_subplot(gs[1])

    """
    Plot Class auPRC
    """


    ax_auprc_zero.hlines(
        y=0.2, xmin=-1, xmax=2,
        linestyles='--', colors='black', alpha=1, lw=2,
        label=None
    )
    for c in label_order:
        ax_auprc_zero.plot(recall_per_class_zero_shot[c], precision_per_class_zero_shot[c], lw=1,
                  label=f"auPRC {c} = {ap_per_class_zero_shot[c]:.3f}")
    ax_auprc_zero.set_xlim([-0.02, 1.02])
    ax_auprc_zero.set_ylim([-0.02, 1.02])

    ax_auprc_zero.set_ylabel("Precision", fontsize=18)
    ax_auprc_zero.set_title("Zero-Shot auPRC", fontsize=20)
    ax_auprc_zero.set_xlabel("")
    ax_auprc_zero.grid()
    handles, labels = ax_auprc_zero.get_legend_handles_labels()
    ax_auprc_zero.legend(handles=handles, labels=labels, markerscale=4, fontsize=16)
    ax_auprc_zero.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_auprc_zero.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=14)
    ax_auprc_zero.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_auprc_zero.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=14)
    ax_auprc_few.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_auprc_few.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=14)
    ax_auprc_few.hlines(
        y=0.2, xmin=-1, xmax=2,
        linestyles='--', colors='black', alpha=1, lw=2,
        label=None
    )
    for c in label_order:
        ax_auprc_few.plot(recall_per_class_few_shot[c], precision_per_class_few_shot[c], lw=1,
                  label=f"auPRC {c} = {ap_per_class_few_shot[c]:.3f}")
    ax_auprc_few.set_xlim([-0.02, 1.02])
    ax_auprc_few.set_ylim([-0.02, 1.02])

    ax_auprc_few.set_ylabel("")
    ax_auprc_few.set_title("Regressive auPRC", fontsize=20)
    ax_auprc_few.set_xlabel("")
    ax_auprc_few.set_yticklabels([])
    ax_auprc_few.tick_params(left=False, bottom=False)
    ax_auprc_few.grid()
    handles, labels = ax_auprc_few.get_legend_handles_labels()
    ax_auprc_few.legend(handles=handles, labels=labels, markerscale=4, fontsize=16)

    fig.supxlabel("Recall",
                  x=0.5,
                  y=0.020,
                  fontsize=18)
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.10,
        top=0.95,
    )

    plt.savefig(figure_path_auprc, dpi=300)
    plt.show()
    plt.close(fig)

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