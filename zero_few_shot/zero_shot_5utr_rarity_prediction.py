import os
import warnings
from pprint import pprint

import numpy as np
from matplotlib import gridspec, patches, MatplotlibDeprecationWarning
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import images_dir, results_dir
from utils.model_definitions import MODELS

COLORMAP = 'viridis'
warnings.filterwarnings(
    "ignore",
    category=MatplotlibDeprecationWarning,
    message=".*get_cmap function was deprecated.*"
)

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    train_embeddings = data["train_embeddings"]
    train_meta_df = pd.DataFrame(data["train_meta"])
    test_embeddings = data["test_embeddings"]
    test_meta_df = pd.DataFrame(data["test_meta"])
    return train_embeddings, train_meta_df, test_embeddings, test_meta_df

def zero_shot_prediction(y_true, cos_similarity):
    precision, recall, _ = precision_recall_curve(y_true, 1 - cos_similarity)
    ap = average_precision_score(y_true, 1 - cos_similarity)
    return ap, precision, recall

def few_shot_prediction(train_embeddings, train_labels, test_embeddings, test_labels):
    model = LogisticRegression(max_iter=5_000, solver='liblinear')
    model.fit(train_embeddings, train_labels)
    y_prob = model.predict_proba(test_embeddings)[:, 1]
    precision, recall, _ = precision_recall_curve(test_labels, y_prob)
    ap = average_precision_score(test_labels, y_prob)
    return ap, precision, recall

def evaluate_af_prediction(
        model_name,
        file_list,
):
    class_dir = os.path.join(images_dir, 'class_by_rarity')
    figure_path = os.path.join(class_dir, f"{model_name}.pdf")

    if os.path.exists(figure_path):
        return

    results = {file.split("layer_")[-1].split(".")[0]: {} for file in file_list}

    all_results = []

    for i, file in enumerate(file_list):
        layer_num = int(file.split("layer_")[-1].split(".")[0])
        train_embeddings, train_meta, test_embeddings, test_meta = load_pkl(file)

        scaler = StandardScaler()
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        test_embeddings_scaled = scaler.transform(test_embeddings)

        train_y_true = train_meta["label"]
        test_y_true = test_meta["label"]
        test_cosime_similarity = test_meta["cos_similarity"]

        ap_zero_shot, precision_zero_shot, recall_zero_shot = zero_shot_prediction(test_y_true, test_cosime_similarity)
        ap_few_shot, precision_few_shot, recall_few_shot = few_shot_prediction(train_embeddings_scaled, train_y_true, test_embeddings_scaled, test_y_true)

        all_results += list(precision_zero_shot) + list(precision_few_shot)
        results[str(layer_num)]["precision_zero_shot"] = precision_zero_shot
        results[str(layer_num)]["recall_zero_shot"] = recall_zero_shot
        results[str(layer_num)]["ap_zero_shot"] = ap_zero_shot
        results[str(layer_num)]["precision_few_shot"] = precision_few_shot
        results[str(layer_num)]["recall_few_shot"] = recall_few_shot
        results[str(layer_num)]["ap_few_shot"] = ap_few_shot

    ymin = min(all_results)
    ymax = max(all_results)
    padding = (ymax - ymin) * 0.1
    ymin, ymax = ymin - padding, ymax + padding

    figsize = (5 * len(results), 5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1,len(results), width_ratios=[1]*len(results), wspace=0.05, hspace=0.05)

    for i, result in enumerate(results):
        layer = result
        precision_zero_shot = results[layer]["precision_zero_shot"]
        recall_zero_shot = results[layer]["recall_zero_shot"]
        ap_zero_shot = results[layer]["ap_zero_shot"]
        precision_few_shot = results[layer]["precision_few_shot"]
        recall_few_shot = results[layer]["recall_few_shot"]
        ap_few_shot = results[layer]["ap_few_shot"]
        print(f"{model_name}\t{layer}\t{ap_zero_shot}\t{ap_few_shot}")
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

    fig.subplots_adjust(left=0.06, right=0.96)
    fig.text(0.5, 0.02, 'Recall', va='center', rotation='horizontal', fontsize=14)
    os.makedirs(class_dir, exist_ok=True)

    plt.savefig(figure_path)
    plt.close(fig)

if __name__ == "__main__":
    embeddings_folder = os.path.join(results_dir, 'embeddings', 'genomic_regions_annotated')
    print(f"MODEL NAME\tLAYER\tZERO SHOT\tFITTED")
    for model_name in MODELS:
        model_folder = os.path.join(embeddings_folder, model_name)
        try:
            files = sorted([os.path.join(model_folder, f) for f in os.listdir(model_folder)
                            if f.endswith(".pkl") and f.startswith("layer_")],
                           key=lambda f: int(f.split("layer_")[-1].split(".")[0]))
        except:
            continue
        evaluate_af_prediction(model_name, files)