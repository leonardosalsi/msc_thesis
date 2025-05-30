import os
import pickle
from enum import Enum
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt, gridspec, patches
from scipy.stats import pearsonr

from config import images_dir
from downstream_evaluation.groupings import _collect_benchmark_data, DATATYPE, _collect_utr_class_data, \
    _collect_mrl_class_data
from downstream_evaluation.plot_mcc import prepare_data_for_visualization
from utils.model_definitions import TASK_GROUPS, get_task_by_name, MODELS

SAVEDIR = os.path.join(images_dir, 'downstream_evaluation')
os.makedirs(SAVEDIR, exist_ok=True)

"""
Avoid recalculation, might be implemented later.
"""

compare_logan = [
    ('NT-50M (no overlap, multispecies)', 'NT-50M (no overlap, logan, EWC 5)'),
    ('NT-50M (overlap, multispecies)', 'NT-50M (overlap, logan, EWC 5)'),
    ('NT-50M (no overlap, multispecies, 2k ctx.)', 'NT-50M (no overlap, logan, EWC 5, 2k ctx.)'),
    ('NT-50M (overlap, multispecies, 2k ctx.)', 'NT-50M (overlap, logan, EWC 5, 2k ctx.)'),
    ('NT-50M (no overlap, multispecies, contrastive CLS)', 'NT-50M (no overlap, logan, EWC 5, contrastive CLS)'),
    ('NT-50M (no overlap, multispecies, contrastive mean-pool)', 'NT-50M (no overlap, logan, EWC 5, contrastive mean-pool)'),
    ('NT-50M (overlap, multispecies, contrastive CLS)', 'NT-50M (overlap, logan, EWC 5, contrastive CLS)'),
    ('NT-50M (overlap, multispecies, contrastive mean-pool)', 'NT-50M (overlap, logan, EWC 5, contrastive mean-pool)'),
    ('NT-50M (no overlap, multispecies, GC & Shannon)', 'NT-50M (no overlap, logan, EWC 5, GC & Shannon)'),
    ('NT-50M (no overlap, multispecies, GC & Shannon)', 'NT-50M (overlap, logan, EWC 5, GC & Shannon)'),
    ('NT-50M (no overlap, multispecies, GC & Shannon, 2k ctx.)', 'NT-50M (no overlap, logan, EWC 5, GC & Shannon, 2k ctx.)'),
    ('NT-50M (overlap, multispecies, GC & Shannon, 2k ctx.)', 'NT-50M (overlap, logan, EWC 5, GC & Shannon, 2k ctx.)')
]

compare_overlap = [
    ('NT-50M (no overlap, multispecies)', 'NT-50M (overlap, multispecies)'),
    ('NT-50M (no overlap, multispecies, 2k ctx.)', 'NT-50M (overlap, multispecies, 2k ctx.)'),
    ('NT-50M (no overlap, multispecies, contrastive CLS)', 'NT-50M (overlap, multispecies, contrastive CLS)'),
    ('NT-50M (no overlap, multispecies, contrastive mean-pool)', 'NT-50M (overlap, multispecies, contrastive mean-pool)'),
    ('NT-50M (no overlap, multispecies, GC & Shannon)', 'NT-50M (overlap, multispecies, GC & Shannon)'),
    ('NT-50M (no overlap, multispecies, GC & Shannon, 2k ctx.)', 'NT-50M (overlap, multispecies, GC & Shannon, 2k ctx.)'),
    ('NT-50M (no overlap, logan, no EWC)', 'NT-50M (overlap, logan, no EWC)'),
    ('NT-50M (no overlap, logan, EWC 0.5)', 'NT-50M (overlap, logan, EWC 0.5)'),
    ('NT-50M (no overlap, logan, EWC 1)', 'NT-50M (overlap, logan, EWC 1)'),
    ('NT-50M (no overlap, logan, EWC 2)', 'NT-50M (overlap, logan, EWC 2)'),
    ('NT-50M (no overlap, logan, EWC 5)', 'NT-50M (overlap, logan, EWC 5)'),
    ('NT-50M (no overlap, logan, EWC 10)', 'NT-50M (overlap, logan, EWC 10)'),
    ('NT-50M (no overlap, logan, EWC 25)', 'NT-50M (overlap, logan, EWC 25)'),
    ('NT-50M (no overlap, logan, EWC 5, 2k ctx.)', 'NT-50M (overlap, logan, EWC 5, 2k ctx.)'),
    ('NT-50M (no overlap, logan, EWC 5, contrastive CLS)', 'NT-50M (overlap, logan, EWC 5, contrastive CLS)'),
    ('NT-50M (no overlap, logan, EWC 5, contrastive mean-pool)', 'NT-50M (overlap, logan, EWC 5, contrastive mean-pool)'),
    ('NT-50M (no overlap, logan, EWC 5, GC & Shannon)', 'NT-50M (overlap, logan, EWC 5, GC & Shannon)'),
    ('NT-50M (no overlap, logan, EWC 5, GC & Shannon, 2k ctx.)', 'NT-50M (overlap, logan, EWC 5, GC & Shannon, 2k ctx.)')
]

compare_sh_gc = [
    ('NT-50M (no overlap, multispecies)', 'NT-50M (no overlap, multispecies, GC & Shannon)'),
    ('NT-50M (no overlap, multispecies, 2k ctx.)', 'NT-50M (no overlap, multispecies, GC & Shannon, 2k ctx.)'),
    ('NT-50M (overlap, multispecies)', 'NT-50M (overlap, multispecies, GC & Shannon)'),
    ('NT-50M (overlap, multispecies, 2k ctx.)', 'NT-50M (overlap, multispecies, GC & Shannon, 2k ctx.)'),
    ('NT-50M (no overlap, logan, EWC 5)', 'NT-50M (no overlap, logan, EWC 5, GC & Shannon)'),
    ('NT-50M (no overlap, logan, EWC 5, 2k ctx.)', 'NT-50M (no overlap, logan, EWC 5, GC & Shannon, 2k ctx.)'),
    ('NT-50M (overlap, logan, EWC 5)', 'NT-50M (overlap, logan, EWC 5, GC & Shannon)'),
    ('NT-50M (overlap, logan, EWC 5, 2k ctx.)', 'NT-50M (overlap, logan, EWC 5, GC & Shannon, 2k ctx.)'),
]

compare_pca = [
    ('NT-50M (no overlap, multispecies)', 'NT-50M (no overlap, multispecies, contrastive CLS)', 'NT-50M (no overlap, multispecies, contrastive mean-pool)'),
    ('NT-50M (overlap, multispecies)', 'NT-50M (overlap, multispecies, contrastive CLS)', 'NT-50M (overlap, multispecies, contrastive mean-pool)'),
    ('NT-50M (no overlap, logan, EWC 5)', 'NT-50M (no overlap, logan, EWC 5, contrastive CLS)', 'NT-50M (no overlap, logan, EWC 5, contrastive mean-pool)'),
    ('NT-50M (overlap, logan, EWC 5)', 'NT-50M (overlap, multispecies, contrastive CLS)', 'NT-50M (overlap, multispecies, contrastive mean-pool)'),
]

compare_context = [
    ('NT-50M (no overlap, multispecies)', 'NT-50M (no overlap, multispecies, 2k ctx.)'),
    ('NT-50M (overlap, multispecies)', 'NT-50M (overlap, multispecies, 2k ctx.)'),
    ('NT-50M (no overlap, multispecies, GC & Shannon)', 'NT-50M (no overlap, multispecies, GC & Shannon, 2k ctx.)'),
    ('NT-50M (overlap, multispecies, GC & Shannon)', 'NT-50M (overlap, multispecies, GC & Shannon, 2k ctx.)'),
    ('NT-50M (no overlap, logan, EWC 5)', 'NT-50M (no overlap, logan, EWC 5, 2k ctx.)'),
    ('NT-50M (overlap, logan, EWC 5)', 'NT-50M (overlap, logan, EWC 5, 2k ctx.)'),
    ('NT-50M (no overlap, logan, EWC 5, GC & Shannon)', 'NT-50M (no overlap, logan, EWC 5, GC & Shannon, 2k ctx.)'),
    ('NT-50M (overlap, logan, EWC 5, GC & Shannon)', 'NT-50M (overlap, logan, EWC 5, GC & Shannon, 2k ctx.)'),
]

def calc_percentual_change_from_a_to_b(a, b):
    return (b - a) / a * 100

def get_normative_name(model_alias):
    for m in MODELS:
        if MODELS[m] == model_alias:
            return m
    print("Could not find model alias '{}'.".format(model_alias))

def compare_one_fold(compare, filename, datahandler):
    compare = dict(enumerate(compare))
    compare_results = {key: [] for key in compare.keys()}
    group = [get_normative_name(a) for a, b in compare.values()] + [get_normative_name(b) for a, b in compare.values()]

    if datahandler == DATATYPE.UTR_CLASS:
        benchmark_files = _collect_utr_class_data(group)
        data = prepare_data_for_visualization(benchmark_files, True)
        means = []
        for idx, c in compare.items():
            c_1 = get_normative_name(c[0])
            c_2 = get_normative_name(c[1])
            task = data['utr5_ben_pat']
            r_1 = task[c_1]['mean']
            r_2 = task[c_2]['mean']
            means.append(r_2 - r_1)
        return np.mean(np.array(means)), np.std(np.array(means))

    benchmark_files = _collect_benchmark_data(group)
    data = prepare_data_for_visualization(benchmark_files, True)

    for idx, c in compare.items():
        c_1 = get_normative_name(c[0])
        c_2 = get_normative_name(c[1])
        for task in data:
            r_1 = data[task][c_1]['mean']
            r_2 = data[task][c_2]['mean']
            diff = calc_percentual_change_from_a_to_b(r_1, r_2)
            print(f"[{task}] {c_1} -> {c_2}: {diff:.2f}%")
            compare_results[idx].append(diff)

    means = []
    stds = []

    for idx, results in compare_results.items():
        means.append(np.mean(results))
        stds.append(np.std(results))

    means = np.array(means)
    stds = np.array(stds)

    mean = float(np.mean(means))
    std = float(np.sqrt(np.mean(stds ** 2 + (means - stds) ** 2)))

    data_to_plot = [results for idx, results in compare_results.items()]
    labels = [f"C{i + 1}" for i, (a, b) in enumerate(compare.values())]

    cmap = plt.cm.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axhline(
        y=0,
        color='black',
        linestyle=':',
        linewidth=1,
        alpha=0.7
    )
    bp = ax.boxplot(
        data_to_plot,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=cmap(0.6), edgecolor=cmap(0.1), alpha=0.7),
        whiskerprops=dict(color=cmap(0.1), linewidth=1),
        capprops=dict(color=cmap(0.1), linewidth=1),
        medianprops=dict(color=cmap(0.1), linewidth=1, linestyle="--"),
        showmeans=True,
        meanline=True,
        meanprops=dict(color="crimson", linestyle="-"),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4),
    )
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% change in MCC")
    ax.text(
        0.01, 0.01,
        f"µ={mean:.3f}%, σ={std:.3f}%",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12,
        color="black"
    )

    plt.tight_layout()
    plt.savefig(os.path.join(SAVEDIR, f"compare_{filename}_mcc.pdf"))
    plt.show()

def compare_two_fold(compare, filename, datahandler):
    compare = dict(enumerate(compare))
    compare_results_cls = {key: [] for key in compare.keys()}
    compare_results_mean = {key: [] for key in compare.keys()}
    compare_results = {key: [] for key in compare.keys()}

    group = [get_normative_name(a) for a, b, c in compare.values()] + [get_normative_name(b) for a, b, c in compare.values()] + [get_normative_name(c) for a, b, c in compare.values()]
    if datahandler == DATATYPE.UTR_CLASS:
        benchmark_files = _collect_utr_class_data(group)
        data = prepare_data_for_visualization(benchmark_files, True)
        means_cls = []
        means_mean = []
        for idx, c in compare.items():
            c_1 = get_normative_name(c[0])
            c_2 = get_normative_name(c[1])
            c_3 = get_normative_name(c[2])
            task = data['utr5_ben_pat']
            r_1 = task[c_1]['mean']
            r_2 = task[c_2]['mean']
            r_3 = task[c_3]['mean']
            means_cls.append(r_2 - r_1)
            means_mean.append(r_3 - r_1)
        return (np.mean(np.array(means_cls)), np.mean(np.array(means_mean))), (np.std(np.array(means_cls)), np.std(np.array(means_mean)))

    benchmark_files = _collect_benchmark_data(group)
    data = prepare_data_for_visualization(benchmark_files, True)

    for idx, c in compare.items():
        c_1 = get_normative_name(c[0])
        c_2 = get_normative_name(c[1])
        c_3 = get_normative_name(c[2])
        for task in data:
            r_1 = data[task][c_1]['mean']
            r_2 = data[task][c_2]['mean']
            r_3 = data[task][c_3]['mean']
            diff_cls = calc_percentual_change_from_a_to_b(r_1, r_2)
            diff_mean = calc_percentual_change_from_a_to_b(r_1, r_3)
            diff = calc_percentual_change_from_a_to_b(r_2, r_3)
            compare_results_cls[idx].append(diff_cls)
            compare_results_mean[idx].append(diff_mean)
            compare_results[idx].append(diff)

    means_cls = []
    stds_cls = []

    for idx, results in compare_results_cls.items():
        means_cls.append(np.mean(results))
        stds_cls.append(np.std(results))

    means_cls = np.array(means_cls)
    stds_cls = np.array(stds_cls)

    mean_cls = float(np.mean(means_cls))
    std_cls = float(np.sqrt(np.mean(stds_cls ** 2 + (means_cls - stds_cls) ** 2)))

    means_mean = []
    stds_mean = []

    for idx, results in compare_results_mean.items():
        means_mean.append(np.mean(results))
        stds_mean.append(np.std(results))

    means_mean = np.array(means_mean)
    stds_mean = np.array(stds_mean)

    mean_mean = float(np.mean(means_mean))
    std_mean = float(np.sqrt(np.mean(stds_mean ** 2 + (means_mean - stds_mean) ** 2)))

    means = []
    stds = []

    for idx, results in compare_results.items():
        means.append(np.mean(results))
        stds.append(np.std(results))

    means = np.array(means)
    stds = np.array(stds)

    mean = float(np.mean(means))
    std = float(np.sqrt(np.mean(stds ** 2 + (means - stds) ** 2)))

    data_to_plot = [results for idx, results in compare_results.items()]
    labels = [f"C{i + 1}" for i, (a, b, c) in enumerate(compare.values())]

    n = len(compare)
    data_cls = [compare_results_cls[i] for i in range(n)]
    data_mean = [compare_results_mean[i] for i in range(n)]
    labels = [f"C{i + 1}" for i in range(n)]

    x = np.arange(n)
    pos_cls = x - 0.2
    pos_mean = x + 0.2

    cmap = plt.cm.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axhline(
        y=0,
        color='black',
        linestyle=':',
        linewidth=1,
        alpha=0.7
    )
    bp1 = ax.boxplot(
        data_cls,
        positions=pos_cls,
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=cmap(0.6), edgecolor=cmap(0.1), alpha=0.7),
        whiskerprops=dict(color=cmap(0.1), linewidth=1),
        capprops=dict(color=cmap(0.1), linewidth=1),
        medianprops=dict(color=cmap(0.1), linewidth=1, linestyle="--"),
        showmeans=True,
        meanline=True,
        meanprops=dict(color="crimson", linestyle="-"),
        showfliers=False,
        zorder=10
    )

    bp2 = ax.boxplot(
        data_mean,
        positions=pos_mean,
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=cmap(0.3), edgecolor=cmap(0.1), alpha=0.7),
        whiskerprops=dict(color=cmap(0.1), linewidth=1),
        capprops=dict(color=cmap(0.1), linewidth=1),
        medianprops=dict(color=cmap(0.1), linewidth=1, linestyle="--"),
        showmeans=True,
        meanline=True,
        meanprops=dict(color="crimson", linestyle="-"),
        showfliers=False,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% change in MCC")
    ax.text(
        0.01, 0.09,
        f"CLS: µ={mean_cls:.3f}%, σ={std_cls:.3f}%",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12,
        color="black"
    )
    ax.text(
        0.01, 0.01,
        f"Mean-pool: µ={mean_mean:.3f}%, σ={std_mean:.3f}%",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12,
        color="black"
    )

    plt.tight_layout()
    plt.savefig(os.path.join(SAVEDIR, f"compare_{filename}_mcc.pdf"))
    plt.show()

def compare_across_groups_one_fold(compare, filename, datahandler):
    if datahandler == DATATYPE.UTR_CLASS:
        return
    compare = dict(enumerate(compare))
    group = [get_normative_name(a) for a, b in compare.values()] + [get_normative_name(b) for a, b in compare.values()]

    benchmark_files = _collect_benchmark_data(group)
    data = prepare_data_for_visualization(benchmark_files, True)

    figsize = (30, 4)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.02, hspace=1)
    axes = [gs[0], gs[1], gs[2], gs[3]]

    group_results = {key: {} for key in TASK_GROUPS}
    all_results = []

    for ax_idx, gs in enumerate(axes):
        task_group = TASK_GROUPS[ax_idx]
        compare_results = {key: [] for key in compare.keys()}
        group_tasks = {d: data[d] for d in data if get_task_by_name(d)['grouping'] == task_group}
        for idx, c in compare.items():
            c_1 = get_normative_name(c[0])
            c_2 = get_normative_name(c[1])
            for task in group_tasks:
                r_1 = group_tasks[task][c_1]['mean']
                r_2 = group_tasks[task][c_2]['mean']
                diff = r_2 - r_1
                compare_results[idx].append(diff)

        means = []
        stds = []

        for idx, results in compare_results.items():
            means.append(np.mean(results))
            all_results.append(np.mean(results))
            stds.append(np.std(results))

        means = np.array(means)
        stds = np.array(stds)

        group_results[task_group] = {
            'means': means,
            'stds': stds,
        }

    ymin = min(all_results)
    ymax = max(all_results)
    print(ymin, ymax)
    padding = (ymax - ymin) * 0.05  # 10% padding
    ymin, ymax = ymin - padding, ymax + padding  # 10% padding

    for ax_idx, task_group in enumerate(TASK_GROUPS):
        ax = fig.add_subplot(axes[ax_idx])
        colors = [
            "green" if m > 0 else "red" if m < 0 else "gray"
            for m in group_results[task_group]['means']
        ]
        n = len(means)
        spacing = 3
        x = np.arange(n) * spacing
        stds_max = max(group_results[task_group]['stds'])
        ax.bar(
            x,
            group_results[task_group]['means'],
            color=colors,
            width=0.8 * spacing
        )
        ax.errorbar(
            x,
            group_results[task_group]['means'],
            yerr=group_results[task_group]['stds'],
            fmt="none",  # no marker
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i + 1}" for i in range(n)])
        ax.set_ylim(ymin - stds_max, ymax + stds_max)
        ax.set_xlim(-0.5 * spacing, (n - 1) * spacing + 0.5 * spacing)
        ax.margins(x=0.1)
        ax.axhline(
            y=0,
            color='black',
            linestyle='-',
            linewidth=1.2,
            alpha=0.7
        )
        rect = patches.FancyBboxPatch(
            (0.0, 1.0),
            1.0, 0.1,
            boxstyle="round,pad=0.00",
            transform=ax.transAxes,
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False
        )
        ax.text(
            0.5, 1.05, task_group,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=16
        )
        if ax_idx == 0:
            ax.set_ylabel('ΔMCC', fontsize=16)
        ax.add_patch(rect)
        if ax_idx > 0:
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(SAVEDIR, f"compare_{filename}_mcc_grouped.pdf"))
    plt.show()

def compare_across_groups_two_fold(compare, filename, datahandler):
    if datahandler == DATATYPE.UTR_CLASS:
        return
    compare = dict(enumerate(compare))
    group = [get_normative_name(a) for a, b, c in compare.values()] + [get_normative_name(b) for a, b, c in compare.values()] + [get_normative_name(c) for a, b, c in compare.values()]

    benchmark_files = _collect_benchmark_data(group)
    data = prepare_data_for_visualization(benchmark_files, True)
    figsize = (30, 8)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], wspace=0.02, hspace=0.04)

    axes = [(gs[0,0], gs[1,0]), (gs[0,1], gs[1,1]), (gs[0,2], gs[1,2]), (gs[0,3], gs[1,3])]

    group_results = {key: {} for key in TASK_GROUPS}
    group_results_cls = {key: {} for key in TASK_GROUPS}
    group_results_mean = {key: {} for key in TASK_GROUPS}

    all_results = []
    all_results_cls = []
    all_results_mean = []

    for ax_idx, ax_cls in enumerate(axes):
        task_group = TASK_GROUPS[ax_idx]
        compare_results_cls = {key: [] for key in compare.keys()}
        compare_results_mean = {key: [] for key in compare.keys()}
        compare_results = {key: [] for key in compare.keys()}
        group_tasks = {d: data[d] for d in data if get_task_by_name(d)['grouping'] == task_group}
        for idx, c in compare.items():
            c_1 = get_normative_name(c[0])
            c_2 = get_normative_name(c[1])
            c_3 = get_normative_name(c[2])
            for task in group_tasks:
                r_1 = group_tasks[task][c_1]['mean']
                r_2 = group_tasks[task][c_2]['mean']
                r_3 = group_tasks[task][c_3]['mean']
                diff_cls = r_2 - r_1
                diff_mean = r_3 - r_1
                diff = diff_cls - diff_mean
                compare_results_cls[idx].append(diff_cls)
                compare_results_mean[idx].append(diff_mean)
                compare_results[idx].append(diff)

        means_cls = []
        stds_cls = []

        for idx, results in compare_results_cls.items():
            means_cls.append(np.mean(results))
            all_results_cls.append(np.mean(results))
            stds_cls.append(np.std(results))

        means_cls = np.array(means_cls)
        stds_cls = np.array(stds_cls)

        group_results_cls[task_group] = {
            'means': means_cls,
            'stds': stds_cls,
        }

        means_mean = []
        stds_mean = []

        for idx, results in compare_results_mean.items():
            means_mean.append(np.mean(results))
            all_results_mean.append(np.mean(results))
            stds_mean.append(np.std(results))

        means_mean = np.array(means_mean)
        stds_mean = np.array(stds_mean)

        group_results_mean[task_group] = {
            'means': means_mean,
            'stds': stds_mean,
        }

        means = []
        stds = []

        for idx, results in compare_results.items():
            means.append(np.mean(results))
            all_results.append(np.mean(results))
            stds.append(np.std(results))

        means = np.array(means)
        stds = np.array(stds)

        group_results[task_group] = {
            'means': means,
            'stds': stds,
        }

    ymin = min(all_results)
    ymax = max(all_results)
    padding = (ymax - ymin) * 0.1
    ymin, ymax = ymin - padding, ymax + padding

    ymin_cls = min(all_results_cls)
    ymax_cls = max(all_results_cls)
    padding = (ymax_cls - ymin_cls) * 0.1
    ymin_cls, ymax_cls = ymin_cls - padding, ymax_cls + padding

    ymin_mean = min(all_results_mean)
    ymax_mean = max(all_results_mean)
    padding = (ymax_mean - ymin_mean) * 0.1
    ymin_mean, ymax_mean = ymin_mean - padding, ymax_mean + padding

    for ax_idx, task_group in enumerate(TASK_GROUPS):
        gs_gr = axes[ax_idx]
        ax_cls = fig.add_subplot(gs_gr[0])
        ax_mean = fig.add_subplot(gs_gr[1])

        colors = [
            "green" if m > 0 else "red" if m < 0 else "gray"
            for m in group_results_cls[task_group]['means']
        ]
        n = len(means)
        spacing = 3
        x = np.arange(n) * spacing

        ax_cls.bar(x, group_results_cls[task_group]['means'], color=colors, width=0.8 * spacing)
        ax_cls.errorbar(
            x,
            group_results_cls[task_group]['means'],
            yerr=group_results_cls[task_group]['stds'],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )
        stds_max = max(group_results_cls[task_group]['stds'])
        ax_cls.set_xticks(x)
        ax_cls.set_xticklabels([f"C{i + 1}" for i in range(n)])
        ax_cls.set_ylim(ymin_cls - stds_max, ymax_cls + stds_max)
        ax_cls.set_xlim(-0.5 * spacing, (n - 1) * spacing + 0.5 * spacing)
        ax_cls.margins(x=0.1)
        ax_cls.axhline(
            y=0,
            color='black',
            linestyle='-',
            linewidth=1.2,
            alpha=0.7
        )
        rect = patches.FancyBboxPatch(
            (0.0, 1.0),
            1.0, 0.12,
            boxstyle="round,pad=0.00",
            transform=ax_cls.transAxes,
            linewidth=0.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False
        )
        ax_cls.text(
            0.5, 1.05, task_group,
            transform=ax_cls.transAxes,
            ha='center',
            va='center',
            fontsize=16
        )
        ax_cls.add_patch(rect)

        ax_cls.set_xticks([])
        if ax_idx > 0:
            ax_cls.set_yticks([])

        colors = [
            "green" if m > 0 else "red" if m < 0 else "gray"
            for m in group_results_mean[task_group]['means']
        ]
        n = len(means)
        spacing = 3
        x = np.arange(n) * spacing

        ax_mean.bar(x, group_results_mean[task_group]['means'], color=colors, width=0.8 * spacing)
        ax_mean.errorbar(
            x,
            group_results_mean[task_group]['means'],
            yerr=group_results_mean[task_group]['stds'],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )
        stds_max = max(group_results_mean[task_group]['stds'])
        ax_mean.set_xticks(x)
        ax_mean.set_xticklabels([f"C{i + 1}" for i in range(n)])
        ax_mean.set_ylim(ymin_mean - stds_max, ymax_mean + stds_max)
        ax_mean.set_xlim(-0.5 * spacing, (n - 1) * spacing + 0.5 * spacing)
        ax_mean.margins(x=0.1)
        ax_mean.axhline(
            y=0,
            color='black',
            linestyle='-',
            linewidth=1.2,
            alpha=0.7
        )

        if ax_idx == len(TASK_GROUPS) - 1:
            ax_cls.text(1.0, 0.5, f"CLS", transform=ax_cls.transAxes,
                        rotation=270, va='center', ha='left', fontsize=16)

            rect = patches.FancyBboxPatch(
                (1.00, -0.002),
                0.06, 1.0,
                transform=ax_cls.transAxes,
                boxstyle="round,pad=0.00",
                linewidth=0.5,
                edgecolor='black',
                facecolor='white',
                clip_on=False
            )
            ax_cls.add_patch(rect)
            ax_mean.text(1.00, 0.5, f"Mean-Pool", transform=ax_mean.transAxes,
                        rotation=270, va='center', ha='left', fontsize=16)

            rect = patches.FancyBboxPatch(
                (1.00, -0.002),
                0.06, 1.0,
                transform=ax_mean.transAxes,
                boxstyle="round,pad=0.00",
                linewidth=0.5,
                edgecolor='black',
                facecolor='white',
                clip_on=False
            )
            ax_mean.add_patch(rect)

        if ax_idx > 0:
            ax_mean.set_yticks([])
    fig.text(0.095, 0.5, 'ΔMCC', va='center', rotation='vertical', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVEDIR, f"compare_{filename}_mcc_grouped.pdf"))
    plt.show()

def plot_5_utr(data, ylabel, filename):
    figsize = (16, 6)
    fig, ax = plt.subplots(figsize=figsize)
    labels = [x for x in data.keys()]
    x = np.arange(len(labels))
    for i, l in enumerate(labels):
        mean = data[l]['means']
        std = data[l]['stds']
        ax.bar(i, mean, label=l, color='green' if mean > 0 else 'crimson')
        ax.errorbar(
            i,
            mean,
            yerr=std,
            fmt="none",  # no marker
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )
    ax.axhline(
        y=0,
        color='black',
        linestyle='-',
        linewidth=1.2,
        alpha=0.7
    )
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=14)
    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.45,
        top=0.95,
    )
    plt.savefig(os.path.join(SAVEDIR, f"compare_{filename}.pdf"))


def collect_mrl_data(compare, type):

    def prepare_mrl_data(files):
        data = {}
        for model_name, file_list in files.items():
            for file in file_list:
                with open(file, "rb") as f:
                    model_results = pickle.load(f)
                if not model_name in model_results:
                    data[model_name] = {}
                y_pred_random_fixed = model_results['y_pred_random_fixed']
                y_true_random_fixed = model_results['y_true_random_fixed']
                y_pred_random_var = model_results['y_pred_random_var']
                y_true_random_var = model_results['y_true_random_var']
                y_pred_human_fixed = model_results['y_pred_human_fixed']
                y_true_human_fixed = model_results['y_true_human_fixed']
                y_pred_human_var = model_results['y_pred_human_var']
                y_true_human_var = model_results['y_true_human_var']

                R_random_fixed = pearsonr(y_true_random_fixed, y_pred_random_fixed)[0]
                R_random_var = pearsonr(y_true_random_var, y_pred_random_var)[0]
                R_human_fixed = pearsonr(y_true_human_fixed, y_pred_human_fixed)[0]
                R_human_var = pearsonr(y_true_human_var, y_pred_human_var)[0]

                data[model_name] = {
                    'R_random_fixed': R_random_fixed,
                    'R_random_var': R_random_var,
                    'R_human_fixed': R_human_fixed,
                    'R_human_var': R_human_var,
                }
        return data

    compare = dict(enumerate(compare))
    if type == 'pca':
        group = [get_normative_name(a) for a, b, c in compare.values()] + [get_normative_name(b) for a, b, c in compare.values()] + [get_normative_name(c) for a, b, c in compare.values()]
        benchmark_files = _collect_mrl_class_data(group)
        data = prepare_mrl_data(benchmark_files)

        R_all_cls = []
        R_human_fixed_cls = []
        R_human_var_cls = []
        R_random_fixed_cls = []
        R_random_var_cls = []
        R_all_mean = []
        R_human_fixed_mean = []
        R_human_var_mean = []
        R_random_fixed_mean = []
        R_random_var_mean = []

        for idx, c in compare.items():
            c_1 = get_normative_name(c[0])
            c_2 = get_normative_name(c[1])
            c_3 = get_normative_name(c[2])
            r_1 = data[c_1]
            r_2 = data[c_2]
            r_3 = data[c_3]

            R_human_fixed_cls.append(r_2['R_human_fixed'] - r_1['R_human_fixed'])
            R_human_var_cls.append(r_2['R_human_var'] - r_1['R_human_var'])
            R_random_fixed_cls.append(r_2['R_random_fixed'] - r_1['R_random_fixed'])
            R_random_var_cls.append(r_2['R_random_var'] - r_1['R_random_var'])
            R_all_cls.append(r_2['R_human_fixed'] - r_1['R_human_fixed'])
            R_all_cls.append(r_2['R_human_var'] - r_1['R_human_var'])
            R_all_cls.append(r_2['R_random_fixed'] - r_1['R_random_fixed'])
            R_all_cls.append(r_2['R_random_var'] - r_1['R_random_var'])

            R_human_fixed_mean.append(r_3['R_human_fixed'] - r_1['R_human_fixed'])
            R_human_var_mean.append(r_3['R_human_var'] - r_1['R_human_var'])
            R_random_fixed_mean.append(r_3['R_random_fixed'] - r_1['R_random_fixed'])
            R_random_var_mean.append(r_3['R_random_var'] - r_1['R_random_var'])
            R_all_mean.append(r_3['R_human_fixed'] - r_1['R_human_fixed'])
            R_all_mean.append(r_3['R_human_var'] - r_1['R_human_var'])
            R_all_mean.append(r_3['R_random_fixed'] - r_1['R_random_fixed'])
            R_all_mean.append(r_3['R_random_var'] - r_1['R_random_var'])

        return (np.mean(np.array(R_all_cls)), np.std(np.array(R_all_cls))), \
            (np.mean(np.array(R_human_fixed_cls)), np.std(np.array(R_human_fixed_cls))), \
            (np.mean(np.array(R_human_var_cls)), np.std(np.array(R_human_var_cls))), \
            (np.mean(np.array(R_random_fixed_cls)), np.std(np.array(R_random_fixed_cls))), \
            (np.mean(np.array(R_random_var_cls)), np.std(np.array(R_random_var_cls))), \
            (np.mean(np.array(R_all_mean)), np.std(np.array(R_all_mean))), \
            (np.mean(np.array(R_human_fixed_mean)), np.std(np.array(R_human_fixed_mean))), \
            (np.mean(np.array(R_human_var_mean)), np.std(np.array(R_human_var_mean))), \
            (np.mean(np.array(R_random_fixed_mean)), np.std(np.array(R_random_fixed_mean))), \
            (np.mean(np.array(R_random_var_mean)), np.std(np.array(R_random_var_mean))),


    else:
        group = [get_normative_name(a) for a, b in compare.values()] + [get_normative_name(b) for a, b in
                                                                        compare.values()]

        benchmark_files = _collect_mrl_class_data(group)
        data = prepare_mrl_data(benchmark_files)

        R_all = []
        R_human_fixed = []
        R_human_var = []
        R_random_fixed = []
        R_random_var = []

        for idx, c in compare.items():
            c_1 = get_normative_name(c[0])
            c_2 = get_normative_name(c[1])
            r_1 = data[c_1]
            r_2 = data[c_2]

            R_human_fixed.append(r_2['R_human_fixed'] - r_1['R_human_fixed'])
            R_human_var.append(r_2['R_human_var'] - r_1['R_human_var'])
            R_random_fixed.append(r_2['R_random_fixed'] - r_1['R_random_fixed'])
            R_random_var.append(r_2['R_random_var'] - r_1['R_random_var'])
            R_all.append(r_2['R_human_fixed'] - r_1['R_human_fixed'])
            R_all.append(r_2['R_human_var'] - r_1['R_human_var'])
            R_all.append(r_2['R_random_fixed'] - r_1['R_random_fixed'])
            R_all.append(r_2['R_random_var'] - r_1['R_random_var'])

        return (np.mean(np.array(R_all)), np.std(np.array(R_all))), \
            (np.mean(np.array(R_human_fixed)), np.std(np.array(R_human_fixed))), \
            (np.mean(np.array(R_human_var)), np.std(np.array(R_human_var))), \
            (np.mean(np.array(R_random_fixed)), np.std(np.array(R_random_fixed))), \
            (np.mean(np.array(R_random_var)), np.std(np.array(R_random_var))),

def plot_mrl(plot_res):
    print(plot_res)
    figsize = (10, 2 * len(plot_res))
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(5, 1, width_ratios=[1], wspace=0.05, hspace=0.1)

    labels = [x for x in plot_res.keys()]
    x = np.arange(len(labels))

    ax_all = fig.add_subplot(gs[0])
    ax_human_fixed = fig.add_subplot(gs[1])
    ax_human_var = fig.add_subplot(gs[2])
    ax_random_fixed = fig.add_subplot(gs[3])
    ax_random_var = fig.add_subplot(gs[4])

    ax_all.set_ylabel("")
    ax_human_fixed.set_ylabel("")
    ax_human_var.set_ylabel("ΔR")
    ax_random_fixed.set_ylabel("")
    ax_random_var.set_ylabel("")
    fig.text(0.975, 0.88, "All Data",
             va="center", rotation="vertical", fontsize=10)
    fig.text(0.975, 0.73, "Human (50bp) MPRA",
             va="center", rotation="vertical", fontsize=10)
    fig.text(0.975, 0.57, "Human (25-100bp) MPRA",
             va="center", rotation="vertical", fontsize=10)
    fig.text(0.975, 0.41, "Random (50bp) MPRA",
             va="center", rotation="vertical", fontsize=10)
    fig.text(0.975, 0.25, "Random (25-100bp) MPRA",
             va="center", rotation="vertical", fontsize=10)
    ax_all.set_xticks([])
    ax_human_fixed.set_xticks([])
    ax_human_var.set_xticks([])
    ax_random_fixed.set_xticks([])
    ax_random_var.set_xticks(x)
    ax_random_var.set_xticklabels(labels, rotation=90)

    lim = [-0.4, 0.4]
    ax_all.set_ylim(lim)
    ax_human_fixed.set_ylim(lim)
    ax_human_var.set_ylim(lim)
    ax_random_fixed.set_ylim(lim)
    ax_random_var.set_ylim(lim)
    ax_all.axhline(
        y=0,
        color='black',
        linestyle='-',
        linewidth=1.2,
        alpha=0.7
    )

    ax_human_fixed.axhline(
        y=0,
        color='black',
        linestyle='-',
        linewidth=1.2,
        alpha=0.7
    )

    ax_human_var.axhline(
        y=0,
        color='black',
        linestyle='-',
        linewidth=1.2,
        alpha=0.7
    )

    ax_random_fixed.axhline(
        y=0,
        color='black',
        linestyle='-',
        linewidth=1.2,
        alpha=0.7
    )

    ax_random_var.axhline(
        y=0,
        color='black',
        linestyle='-',
        linewidth=1.2,
        alpha=0.7
    )

    for i, model_group in enumerate(plot_res):
        ax_all.bar(i, plot_res[model_group]['R_all']['means'], color='green' if plot_res[model_group]['R_all']['means'] > 0 else 'crimson')
        ax_human_fixed.bar(i, plot_res[model_group]['R_human_fixed']['means'], color='green' if plot_res[model_group]['R_human_fixed']['means'] > 0 else 'crimson')
        ax_human_var.bar(i, plot_res[model_group]['R_human_var']['means'], color='green' if plot_res[model_group]['R_human_var']['means'] > 0 else 'crimson')
        ax_random_fixed.bar(i, plot_res[model_group]['R_random_fixed']['means'], color='green' if plot_res[model_group]['R_random_fixed']['means'] > 0 else 'crimson')
        ax_random_var.bar(i, plot_res[model_group]['R_random_var']['means'], color='green' if plot_res[model_group]['R_random_var']['means'] > 0 else 'crimson')
        ax_all.errorbar(
            i,
            plot_res[model_group]['R_all']['means'],
            yerr=plot_res[model_group]['R_all']['stds'],
            fmt="none",  # no marker
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )
        ax_human_fixed.errorbar(
            i,
            plot_res[model_group]['R_human_fixed']['means'],
            yerr=plot_res[model_group]['R_human_fixed']['stds'],
            fmt="none",  # no marker
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )
        ax_human_var.errorbar(
            i,
            plot_res[model_group]['R_human_var']['means'],
            yerr=plot_res[model_group]['R_human_var']['stds'],
            fmt="none",  # no marker
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )
        ax_random_fixed.errorbar(
            i,
            plot_res[model_group]['R_random_fixed']['means'],
            yerr=plot_res[model_group]['R_random_fixed']['stds'],
            fmt="none",  # no marker
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )
        ax_random_var.errorbar(
            i,
            plot_res[model_group]['R_random_var']['means'],
            yerr=plot_res[model_group]['R_random_var']['stds'],
            fmt="none",  # no marker
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5
        )
    plt.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.18)
    plt.savefig(os.path.join(SAVEDIR, f"compare_mrl.pdf"))

class CompareHandler(Enum):
    LOGAN = ('logan', compare_logan, "Logan")
    OVERLAP = ('overlap', compare_overlap, "Overlapping")
    SAMPLING = ('sh_gc', compare_sh_gc, "GC & Shannon")
    PCA = ('pca', compare_pca, "Contrastive")
    CONTEXT = ('context', compare_context, "2k Context")

if __name__ == '__main__':
    datahandler = DATATYPE.MRL_PRED
    plot_res = {}

    if datahandler != DATATYPE.MRL_PRED:
        for handler in CompareHandler:
            filename, compare, clearname = handler.value
            if len(compare[0]) == 2:
                compare_fn = compare_one_fold
                grp_compare_fn = compare_across_groups_one_fold
            elif len(compare[0]) == 3:
                compare_fn = compare_two_fold
                grp_compare_fn = compare_across_groups_two_fold
            else:
                raise NotImplementedError
            res = compare_fn(compare, filename, datahandler)
            grp_compare_fn(compare, filename, datahandler)
            if res is not None:
                means, stds = res
                if isinstance(means, tuple):
                    means_cls, means_mean = means
                    stds_cls, stds_mean = stds
                    plot_res['Contrastive (CLS)'] = {"means": means_cls, "stds": stds_cls}
                    plot_res['Contrastive (mean-pool)'] = {"means": means_mean, "stds": stds_mean}
                else:
                    plot_res[clearname] = {"means": means, "stds": stds}
        if plot_res != {}:
            plot_5_utr(plot_res, "ΔMCC", '5_utr')
    else:
        for handler in CompareHandler:
            filename, compare, clearname = handler.value
            res = collect_mrl_data(compare, filename)

            if len(res) == 10:
                plot_res['Contrastive (CLS)'] = {
                    "R_all": {'means': res[0][0], "stds": res[0][1]},
                    "R_human_fixed": {'means': res[1][0], "stds": res[1][1]},
                    "R_human_var": {'means': res[2][0], "stds": res[2][1]},
                    "R_random_fixed": {'means': res[3][0], "stds": res[3][1]},
                    "R_random_var": {'means': res[4][0], "stds": res[4][1]},
                }
                plot_res['Contrastive (mean-pool)'] = {
                    "R_all": {'means': res[5][0], "stds": res[5][1]},
                    "R_human_fixed": {'means': res[6][0], "stds": res[6][1]},
                    "R_human_var": {'means': res[7][0], "stds": res[7][1]},
                    "R_random_fixed": {'means': res[8][0], "stds": res[8][1]},
                    "R_random_var": {'means': res[9][0], "stds": res[9][1]},
                }
            else:
                plot_res[clearname] = {
                    "R_all": {'means': res[0][0], "stds": res[0][1]},
                    "R_human_fixed": {'means': res[1][0], "stds": res[1][1]},
                    "R_human_var": {'means': res[2][0], "stds": res[2][1]},
                    "R_random_fixed": {'means': res[3][0], "stds": res[3][1]},
                    "R_random_var": {'means': res[4][0], "stds": res[4][1]},
                }
        plot_mrl(plot_res)