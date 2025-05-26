import json
import math
import os
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt, gridspec, colors, patches
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef

from downstream_evaluation.groupings import get_task_alias, get_model_alias_for_downstream, DATATYPE, \
    get_for_all_compare_to_litereature, get_for_all_compare, get_for_ewc_compare, get_for_best_logan_compare, \
    get_for_context_length_compare, get_for_reference_compare
from config import results_dir, images_dir

TASK_ORDER = [

]

def plot_benchmark_validity(data, savedir):
    figsize = (6.4, 3.4)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,0.2, 1], wspace=0.05, hspace=0.5)

    ax_human = fig.add_subplot(gs[0])
    spacer = fig.add_subplot(gs[1])
    ax_1000G = fig.add_subplot(gs[2])

    spacer.axis('off')

    """
    Plot validation for NT-500M-Human
    """
    data_human = data['human']

    ax_human.set_title("NT-HumanRef (500M)", loc="center")
    ax_human.set_ylim(0.4, 1)
    ax_human.set_xlim(0.4, 1)
    ax_human.grid(alpha=0.3)
    ax_human.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.5)
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
    ax_1000G.set_title("NT-1000G (500M)", loc="center")
    ax_1000G.set_ylim(0.4, 1)
    ax_1000G.set_xlim(0.4, 1)
    ax_1000G.grid(alpha=0.3)
    ax_1000G.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.5)
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
    plt.savefig(os.path.join(savedir, f'pearson_validation.pdf'))
    plt.show()



def plot_downstream_results(data):
    rows = 6
    cols = 3
    figsize = (10, 15)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols, width_ratios=[1, 1, 1], wspace=0.05, hspace=0.5)

    i = 0
    curr_x = 0
    curr_y = 0

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    _cmap = plt.get_cmap("viridis")
    cmap = truncate_colormap(_cmap, minval=0.0, maxval=0.6)

    for task in data:
        print(task)

    for task in data:
        task_results = data[task]
        task_alias = get_task_alias(task)

        human_ref = task_results['ref_multi_species_no_cont_500_human']
        human_meas = task_results['default_multi_species_no_cont_500_human']

        tg_ref = task_results['ref_multi_species_no_cont_500_tg']
        tg_meas = task_results['default_multi_species_no_cont_500_tg']

        ax = plt.subplot(gs[curr_y,curr_x])
        ax.set_ylim(0, 1)
        ax.set_xlim(0.4, 4.6)

        human_var_rect = patches.FancyBboxPatch(
            (0.0, 1.0),
            1.0, 0.3,
            boxstyle="round,pad=0.00",
            transform=ax.transAxes,
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False,
        )
        ax.text(
            0.5, 1.12, task,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=14
        )
        ax.add_patch(human_var_rect)

        ax.bar(1, human_ref['mean'], 0.9, color=cmap(0.0))
        ax.bar(2, human_meas['mean'], 0.9, yerr=human_meas['std']*2, capsize=2 ,color=cmap(0.33))
        ax.bar(3, tg_ref['mean'], 0.9, color=cmap(0.66))
        ax.bar(4, tg_meas['mean'], 0.9, yerr=tg_meas['std']*2, capsize=2, color=cmap(0.88))
        ax.plot([2.5, 2.5], [0, 1], "--", color="black", linewidth=1)


        if  human_ref['mean'] < 0.49:
            y_pos = human_ref['mean'] + 0.05
            color = 'black'
        else:
            y_pos = 0.1
            color = 'white'


        ax.text(
            0.15, y_pos, f"{human_ref['mean']:.3f}",
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=12,
            rotation=90,
            color=color,
            fontweight='bold'
        )

        if  human_meas['mean'] < 0.49:
            y_pos = human_meas['mean'] + 0.05
            color = 'black'
        else:
            y_pos = 0.1
            color = 'white'

        ax.text(
            0.39, y_pos, f"{human_meas['mean']:.3f}",
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=12,
            rotation=90,
            color=color,
            fontweight='bold'
        )

        if  tg_ref['mean'] < 0.49:
            y_pos = tg_ref['mean'] + 0.05
            color = 'black'
        else:
            y_pos = 0.1
            color = 'white'

        ax.text(
            0.63, y_pos, f"{tg_ref['mean']:.3f}",
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=12,
            rotation=90,
            color=color,
            fontweight='bold'
        )

        if  tg_meas['mean'] < 0.49:
            y_pos = tg_meas['mean'] + 0.05
            color = 'black'
        else:
            y_pos = 0.1
            color = 'white'

        ax.text(
            0.86, y_pos, f"{tg_meas['mean']:.3f}",
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=12,
            rotation=90,
            color=color,
            fontweight='bold'
        )

        if curr_x != 0:
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            ax.set_yticks([0,1])

        if curr_y != rows - 1:
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xticks([1,2,3,4])
            ax.set_xticklabels(["NT-HumanRef (Reference)", "NT-HumanRef (Measured)", "NT-1000G (Reference)", "NT-1000G (Measured)"],
                               rotation=90,
                               fontsize=14
                               )

        if curr_x == 2:
            curr_x = 0
            curr_y +=1
        else:
            curr_x += 1

    fig.text(0.02, 0.57, 'MCC', va='center', rotation='vertical', fontsize=14)
    plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.18 )
    plt.savefig(os.path.join(savedir, f'benchmark_validation.pdf'))
    plt.show()


def get_result_vectors(data):
    ref_tg = []
    meas_tg = []
    ref_human = []
    meas_human = []
    for task in data:
        ref_tg.append(data[task]['ref_multi_species_no_cont_500_tg']['mean'])
        meas_tg.append(data[task]['default_multi_species_no_cont_500_tg']['mean'])
        ref_human.append(data[task]['ref_multi_species_no_cont_500_human']['mean'])
        meas_human.append(data[task]['default_multi_species_no_cont_500_human']['mean'])

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

def evaluate_file(filepath, bootstrapped):
    with open(filepath, "r") as f:
        data = json.load(f)

    if bootstrapped:
        """
        Calculate Law of Total Variance (LOTV) for bootstrap results.
        """
        try:
            b_mean = np.array(data['bootstrap_means'])
            b_std = np.array(data['bootstrap_stds'])
            mean = float(np.mean(b_mean))
            std = float(np.sqrt(np.mean(b_std ** 2 + (b_mean - mean) ** 2)))
            return mean, std
        except KeyError:
            mean = data['mean']
            std = data['std']
            return mean, std
    else:
        mean = data['mean']
        std = data['std']
        return mean, std

def prepare_data_for_visualization(file_lists, bootstrapped=False):
    data = {}
    for model_name, file_list in file_lists.items():
        for file in file_list:
            task = file.split("/")[-1].replace('.json', '')
            if not task in data:
                data[task] = {}
            mean, std = evaluate_file(file, bootstrapped)
            data[task][model_name] = {'mean': mean, 'std': std}
    return data


if __name__ == '__main__':
    compare_group = get_for_reference_compare
    data_class = DATATYPE.BENCHMARK
    bootstrapped = True

    savedir = os.path.join(images_dir, 'benchmark')
    os.makedirs(savedir, exist_ok=True)
    benchmark_files, filename = compare_group(data_class)
    data = prepare_data_for_visualization(benchmark_files, bootstrapped)
    plot_downstream_results(data)
    data = get_result_vectors(data)
    plot_benchmark_validity(data, savedir)



