import math
import os
import json
from downstream_tasks import TASKS, MODELS
import numpy as np
from matplotlib import pyplot as plt

data_location = "./data"
task_permutation = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 5, 1, 3, 2, 6, 7, 8, 27, 19, 20, 21, 22, 23, 24, 25, 26]
model_permutation = [1.5, 4.5, 1, 5, 3, 2, 6, 4]



def get_model_by_id(modelId):
    for model in MODELS:
        if model['modelId'] == modelId:
            return model
    return None

def get_task_by_id(taskId):
    for task in TASKS:
        if task['taskId'] == taskId:
            return task
    return None

"""
Collect and sort data per downstream task
"""
def prepare_data_for_visualization():
    _data = {}
    files = [f for f in os.listdir(data_location) if os.path.isfile(os.path.join(data_location, f))]
    for taskId in task_permutation:
        task = get_task_by_id(taskId)
        _data[task['data_alias']] = {}

    for taskId in task_permutation:
        task = get_task_by_id(taskId)
        task_files = list(filter(lambda filename: task['alias'] in filename, files))
        if task['alias'] == 'enhancers':
            task_files = list(filter(lambda filename: 'Genomic' not in filename, task_files))
            task_files = list(filter(lambda filename: 'types' not in filename, task_files))
        if len(task_files) == 0:
            continue
        for modelId in model_permutation:
            model = get_model_by_id(int(modelId))
            model_task_files = list(filter(lambda filename: model['name'] in filename, task_files))
            mode = ""
            try:
                if modelId == int(modelId):
                    file = list(filter(lambda filename: '-with-random-weights' not in filename, model_task_files))[0]
                else:
                    file = list(filter(lambda filename: '-with-random-weights' in filename, model_task_files))[0]
                    mode = " with rand. weights"
            except:
                print("=======")
                print("ERROR ON " + task['data_alias'] + "   " + model['data_alias'])
                print(model_task_files)
            file_path = os.path.join('data', file)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    content = json.loads(content)
                    _data[task['data_alias']][model['data_alias'] + mode] = {'mean': content['mean'],
                                                                             'std': content['std']}
    return _data

def visualize_mcc_per_task(data):
    num_tasks = len(data)
    print(f"Number of tasks: {num_tasks}")
    cols = 3
    rows = math.ceil(num_tasks / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5), constrained_layout=True)
    axes = axes.flatten()

    colors = plt.cm.tab10.colors

    for idx, (task_name, model_results) in enumerate(data.items()):
        if idx >= len(axes):  # Safety check
            break

        ax = axes[idx]
        model_names = list(model_results.keys())
        mcc_values = [model_results[model]['mean'] for model in model_names]
        std_values = [model_results[model]['std'] * 2 for model in model_names]

        x = np.arange(len(model_names))
        bars = ax.bar(
            x,
            mcc_values,
            yerr=std_values,
            color=colors[:len(model_names)],
            width=0.9,
            capsize=5,
            error_kw={'elinewidth': 2}
        )

        for bar, value in zip(bars, mcc_values):
            if value > 0.2:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + 0.05,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=12,
                    color="white",
                    weight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() + 0.05,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=12,
                    color="black",
                    weight="bold",
                )

        ax.set_title(task_name, fontsize=24, pad=10, loc="center")
        ax.set_ylim(0, 1)

        col = idx % cols
        is_last_row = idx >= (rows - 1) * cols or idx + cols >= num_tasks
        if is_last_row:
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=90, ha="center", fontsize=16)
        else:
            ax.set_xticks([])

    for idx in range(num_tasks, len(axes)):
        fig.delaxes(axes[idx])
    plt.savefig('img/eval_mcc.svg')

def visualize_mcc_across_tasks(data):
    model_mcc = {}
    for (task_name, model_results) in data.items():
        for (model_name, scores) in model_results.items():
            if model_name not in model_mcc:
                model_mcc[model_name] = []
            model_mcc[model_name].append(scores['mean'])


    for (model_name, scores) in model_mcc.items():
        model_mcc[model_name] = np.mean(scores)

    model_names = list(model_mcc.keys())
    mean_mcc = [model_mcc[model] for model in model_names]

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    bars = ax.bar(model_names, mean_mcc, color=colors[:len(model_names)], width=0.9)
    for bar, value in zip(bars, mean_mcc):
        if value > 0.2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + 0.05,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=12,
                color="white",
                weight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() + 0.05,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=12,
                color="black",
                weight="bold",
            )
    x = np.arange(len(model_names))
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=90, ha="center", fontsize=12)
    ax.set_title("Mean MCC across Tasks", fontsize=18, pad=10, loc="center")
    ax.set_ylim(0, 1)
    plt.savefig('img/mcc_across_tasks.svg')

def visualize_normalized_mcc_across_tasks(data):
    model_mcc = {}
    for (task_name, model_results) in data.items():
        for (model_name, scores) in model_results.items():
            if model_name not in model_mcc:
                model_mcc[model_name] = []
            model_mcc[model_name].append(scores['mean'])

    for (model_name, scores) in model_mcc.items():
        min = np.min(scores)
        max = np.max(scores)
        normalized_scores = (scores - min) / (max - min)
        model_mcc[model_name] = np.mean(normalized_scores)

    model_names = list(model_mcc.keys())
    mean_mcc = [model_mcc[model] for model in model_names]

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    bars = ax.bar(model_names, mean_mcc, color=colors[:len(model_names)], width=0.9)
    for bar, value in zip(bars, mean_mcc):
        if value > 0.2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + 0.05,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=12,
                color="white",
                weight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() + 0.05,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=12,
                color="black",
                weight="bold",
            )
    x = np.arange(len(model_names))
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=90, ha="center", fontsize=12)
    ax.set_title("Normalized mean MCC across Tasks", fontsize=18, pad=10, loc="center")
    ax.set_ylim(0, 1)
    plt.savefig('img/norm_mcc_across_tasks.svg')

if __name__ == "__main__":
    data = prepare_data_for_visualization()
    with open('mcc_data_lora.json', 'w') as f:
        json.dump(data, f, indent=4)
    visualize_mcc_per_task(data)
    visualize_mcc_across_tasks(data)
    visualize_normalized_mcc_across_tasks(data)