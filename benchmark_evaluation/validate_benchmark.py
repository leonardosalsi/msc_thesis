import json
import math
import os
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt, gridspec
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef

from benchmark_evaluation.groupings import get_task_alias, get_model_alias_for_downstream, DATATYPE, \
    get_for_all_compare_to_litereature, get_for_all_compare, get_for_ewc_compare, get_for_best_logan_compare, \
    get_for_context_length_compare, get_for_reference_compare
from config import results_dir, images_dir

def plot_benchmark_validity(data, savedir):
    figsize = (6, 3)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,0.1, 1], wspace=0.05, hspace=0.5)

    ax_human = fig.add_subplot(gs[0])
    spacer = fig.add_subplot(gs[1])
    ax_1000G = fig.add_subplot(gs[2])

    spacer.axis('off')

    """
    Plot validation for NT-500M-Human
    """
    data_human = data['human']

    ax_human.set_title("NT-500M-Human", loc="center")
    ax_human.set_ylim(0.4, 1)
    ax_human.set_xlim(0.4, 1)
    ax_human.grid()
    ax_human.plot([0, 1], [0, 1], "--", color="gray")
    ax_human.scatter(data_human['ref'], data_human['meas'], s=10, c='b', label='Reference')
    ax_human.set_xlabel("Reference MCC")
    ax_human.set_ylabel("Measured MCC")
    ax_human.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    R_human_var = pearsonr(data_human['ref'], data_human['meas'])[0]
    ax_human.text(
        0.55, 0.1, f"R = {R_human_var:.3f}",
        transform=ax_human.transAxes,
        fontsize=14,
        fontstyle='italic',
        verticalalignment='top'
    )

    """
    Plot validation for NT-500M-1000G
    """
    data_1000G = data['1000G']
    ax_1000G.set_title("NT-500M-1000G", loc="center")
    ax_1000G.set_ylim(0.4, 1)
    ax_1000G.set_xlim(0.4, 1)
    ax_1000G.grid()
    ax_1000G.plot([0, 1], [0, 1], "--", color="gray")
    ax_1000G.scatter(data_1000G['ref'], data_1000G['meas'], s=10, c='b', label='Reference')
    ax_1000G.set_xlabel("Reference MCC")
    ax_1000G.set_ylabel("")
    ax_1000G.set_xticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    R_1000G_var = pearsonr(data_1000G['ref'], data_1000G['meas'])[0]
    ax_1000G.text(
        0.55, 0.1, f"R = {R_1000G_var:.3f}",
        transform=ax_1000G.transAxes,
        fontsize=14,
        fontstyle='italic',
        verticalalignment='top'
    )

    plt.subplots_adjust(hspace=0.05, bottom=0.18, top=0.92, left=0.1, right=0.98)
    plt.savefig(os.path.join(savedir, f'benchmark_validation.png'))
    plt.show()

def get_result_vectors(data):
    ref_tg = []
    meas_tg = []
    ref_human = []
    meas_human = []
    for task in data:
        ref_tg.append(data[task]['ref_multi_species_untrained_500_tg']['mean'])
        meas_tg.append(data[task]['default_multi_species_untrained_500_tg']['mean'])
        ref_human.append(data[task]['ref_multi_species_untrained_500_human']['mean'])
        meas_human.append(data[task]['default_multi_species_untrained_500_human']['mean'])

    return {
        "human": {
            "ref": ref_human,
            "meas": meas_human
        },
        "1000G": {
            "ref": ref_tg,
            "meas": meas_tg
        }
    }

def evaluate_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    try:
        labels = list(map(lambda x: x['labels'], data))
        predictions = list(map(lambda x: x['predictions'], data))
        assert len(labels) == len(predictions)
        scores = []

        for label, prediction in zip(labels, predictions):
            score = matthews_corrcoef(label, prediction)
            scores.append(score)

        mean = np.mean(scores)
        std = np.std(scores)
    except:
        mean = data['mean']
        std = data['std']
    return mean, std

def prepare_data_for_visualization(file_lists):
    data = {}
    for model_name, file_list in file_lists.items():
        for file in file_list:
            task = file.split("/")[-1].replace('.json', '')
            if not task in data:
                data[task] = {}
            mean, std = evaluate_file(file)
            data[task][model_name] = {'mean': mean, 'std': std}
    return data


if __name__ == '__main__':
    compare_group = get_for_reference_compare
    data_class = DATATYPE.BENCHMARK
    savedir = os.path.join(images_dir, 'benchmark')
    os.makedirs(savedir, exist_ok=True)
    benchmark_files, filename = compare_group(data_class)
    data = prepare_data_for_visualization(benchmark_files)
    data = get_result_vectors(data)
    plot_benchmark_validity(data, savedir)



