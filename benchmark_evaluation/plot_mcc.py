import json
import math
import os
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef

from benchmark_evaluation.groupings import get_task_alias, get_model_alias_for_downstream, DATATYPE, \
    get_for_all_compare_to_litereature, get_for_all_compare, get_for_ewc_compare, get_for_best_logan_compare, \
    get_for_context_length_compare
from config import results_dir, images_dir

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

def visualize_mcc_per_task(data, colors, filename_base, model_names):
    num_tasks = len(data)
    print(num_tasks)
    print(f"Number of tasks: {num_tasks}")
    cols = 3
    rows = math.ceil(num_tasks / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(len(model_names) * 2, rows * 5), constrained_layout=True)
    axes = axes.flatten()

    for idx, (task_name, model_results) in enumerate(data.items()):
        if idx >= len(axes):  # Safety check
            break

        task_data_name = get_task_alias(task_name)
        ax = axes[idx]

        mcc_values = [model_results.get(model, {}).get('mean', 0) for model in model_names]
        std_values = [model_results.get(model, {}).get('std', 0) * 2 for model in model_names]

        x = [i * 0.21 for i in range(len(model_names))]
        bars = ax.bar(
            x,
            mcc_values,
            yerr=std_values,
            color=colors[:len(model_names)],
            width=0.2,
            capsize=5,
            error_kw={'elinewidth': 2}
        )

        for bar, value in zip(bars, mcc_values):
            if value > 0.2:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + 0.05,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=18,
                    color="white",
                    weight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() + 0.05,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=18,
                    color="black",
                    weight="bold",
                )
        ax.grid(axis='y')
        ax.set_title(task_data_name, fontsize=24, pad=10, loc="center")
        ax.set_ylim(0, 1)

        col = idx % cols
        is_last_row = idx >= (rows - 1) * cols or idx + cols >= num_tasks
        if is_last_row:
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=90, ha="center", fontsize=16)
        else:
            ax.set_xticks([])
        for label in ax.get_xticklabels():
            if label.get_text() == "NT-MS V2 (50M)":
                label.set_fontweight("bold")

    for idx in range(num_tasks, len(axes)):
        fig.delaxes(axes[idx])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(filename_base, f'mcc_per_tasks.png'))
    plt.show()


def visualize_mcc_across_tasks(data, filename_base):
    model_mcc = {}
    for task_name, model_results in data.items():
        for model_name, scores in model_results.items():
            model_mcc.setdefault(model_name, []).append(scores['mean'])

    for model_name, scores in model_mcc.items():
        model_mcc[model_name] = np.mean(scores)

    # Sort model names by mean MCC
    model_names = sorted(model_mcc, key=model_mcc.get, reverse=False)
    model_aliases = [get_model_alias_for_downstream(m) for m in model_names]
    mean_mcc = [model_mcc[m] for m in model_names]

    # Sample viridis colormap
    cmap = plt.cm.get_cmap('viridis', len(model_names))
    n_colors = len(model_names)
    colors = [cmap(0.0 + (0.85 - 0.0) * i / (n_colors - 1)) for i in range(n_colors)]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.barh(model_names, mean_mcc, color=colors, height=0.9)

    for bar, value in zip(bars, mean_mcc):
        ax.text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="black",
            weight="bold"
        )

    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean MCC", fontsize=14)
    ax.set_title("Mean MCC across Tasks", fontsize=18, pad=10, loc="center")
    ax.grid(axis='x')
 
    # Bold specific label
    for label in ax.get_yticklabels():
        if label.get_text() == "NT-MS V2 (50M)":
            label.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(os.path.join(filename_base, f'mcc_across_tasks.png'))
    plt.show()
    return model_names, colors

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

def get_ranking(data):
    tasks = list(data.keys())
    models = list(data[tasks[0]].keys())
    scores = {}
    pprint(data)
    for model in models:
        scores[model] = 0
    num_matches = 0
    for i, model_1 in enumerate(models):
        remainder = models[i:]

        for j, model_2 in enumerate(remainder):
            if j != i:
                for task in tasks:
                    if i == 0:
                        num_matches += 1
                    results = data[task]
                    result_1 = results[model_1]['mean']
                    result_2 = results[model_2]['mean']
                    if result_1 > result_2:
                        scores[model_1] += 1
                    elif result_1 < result_2:
                        scores[model_2] += 1

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    print("Matches:", num_matches)
    filename = f'/shared/img/mcc_model_ranking.txt'


    with open(filename, "w") as f:
        for i, p in enumerate(sorted_keys):
            winrate = "{:.2f}".format(scores[p] / num_matches * 100)
            f.write(f"{i + 1}: {p} [{scores[p]} wins, winrate {winrate}%]\n")

def get_mean_task_rank(data):
    tasks = list(data.keys())
    models = list(data[tasks[0]].keys())
    scores = {}
    mean_scores = {}
    for model in models:
        scores[model] = []
    for task, results in data.items():
        task_results = data[task]
        sorted_models = sorted(task_results.keys(), key=lambda _model: task_results[_model]['mean'], reverse=True)
        for _model in sorted_models:
            scores[_model].append(sorted_models.index(_model))

    for model, _scores in scores.items():
        mean_scores[model] = float(np.mean(_scores))

    sorted_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=False)
    print(sorted_scores)
    filename = f'/shared/img/mcc_model_mean_ranking.txt'


    with open(filename, "w") as f:
        for i, p in enumerate(sorted_scores):
            f.write(f"{i + 1}: {p[0]} [Mean rank {p[1]}]\n")

if __name__ == '__main__':
    savedir = os.path.join(images_dir, 'benchmark')
    os.makedirs(savedir, exist_ok=True)
    f = get_for_all_compare_to_litereature
    benchmark_files, filename = f(DATATYPE.BENCHMARK)
    data = prepare_data_for_visualization(benchmark_files)
    filename_base = os.path.join(savedir, filename)
    os.makedirs(filename_base, exist_ok=True)
    #get_mean_task_rank(data)
    model_names, colors = visualize_mcc_across_tasks(data, filename_base)
    visualize_mcc_per_task(data, colors, filename_base, model_names)


