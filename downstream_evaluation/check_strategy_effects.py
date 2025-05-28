import os
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt, gridspec, patches
from config import images_dir
from downstream_evaluation.groupings import _collect_benchmark_data
from downstream_evaluation.plot_mcc import prepare_data_for_visualization
from utils.model_definitions import TASK_GROUPS, get_task_by_name, MODELS

SAVEDIR = os.path.join(images_dir, 'downstream_evaluation')
os.makedirs(SAVEDIR, exist_ok=True)

"""
Avoid recalculation, might be implemented later.
"""
MCCRES = {
    'NT-50M (no continual)': 0.6629566369022641,
    'NT-100M (no continual)': 0.6688366794294678,
    'NT-250M (no continual)': 0.6822685828857774,
    'NT-500M (no continual)': 0.6890571709601603,
    'NT-50M (no overlap, multispecies)': 0.6662839020426587,
    'NT-50M (no overlap, multispecies, 2k ctx.)': 0.6675417422572667,
    'NT-50M (overlap, multispecies)': 0.6613914573274892,
    'NT-50M (overlap, multispecies, 2k ctx.)': 0.6624570634557607,
    'NT-50M (overlap, logan, no EWC)': 0.6638213120747101,
    'NT-50M (overlap, logan, EWC 0.5)': 0.6596238867948331,
    'NT-50M (overlap, logan, EWC 1)': 0.6616211280861498,
    'NT-50M (overlap, logan, EWC 2)': 0.6587640708513566,
    'NT-50M (overlap, logan, EWC 5)': 0.6614948389026848,
    'NT-50M (overlap, logan, EWC 10)': 0.6610889769046829,
    'NT-50M (overlap, logan, EWC 25)': 0.6658175478751497,
    'NT-50M (no overlap, multispecies, GC & Shannon)': 0.6660513633716081,
    'NT-50M (no overlap, multispecies, GC & Shannon, 2k ctx.)': 0.667852625929835,
    'NT-50M (overlap, multispecies, GC & Shannon)': 0.6562630444391516,
    'NT-50M (overlap, multispecies, GC & Shannon, 2k ctx.)': 0.661293194091659,
    'NT-50M (no overlap, multispecies, contrastive CLS)': 0.6726701713891627,
    'NT-50M (no overlap, multispecies, contrastive mean-pool)': 0.6592191915966387,
    'NT-50M (overlap, multispecies, contrastive CLS)': 0.6626690524094649,
    'NT-50M (overlap, multispecies, contrastive mean-pool)': 0.6598921815072202,
    'NT-50M (no overlap, logan, no EWC)': 0.6700147541151964,
    'NT-50M (no overlap, logan, EWC 0.5)': 0.6696880592281658,
    'NT-50M (no overlap, logan, EWC 1)': 0.667828340649412,
    'NT-50M (no overlap, logan, EWC 2)': 0.6676896318309892,
    'NT-50M (no overlap, logan, EWC 5)': 0.6707952652565227,
    'NT-50M (no overlap, logan, EWC 10)': 0.6700759206998628,
    'NT-50M (no overlap, logan, EWC 25)': 0.6670971334968067,
    'NT-50M (no overlap, logan, EWC 5, 2k ctx.)': 0.6619595012383651,
    'NT-50M (overlap, logan, EWC 5, contrastive CLS)': 0.6661911357899541,
    'NT-50M (overlap, logan, EWC 5, contrastive mean-pool)': 0.6640817023875969,
    'NT-50M (no overlap, logan, EWC 5, contrastive CLS)': 0.6751144701353696,
    'NT-50M (no overlap, logan, EWC 5, contrastive mean-pool)': 0.6673329424574441,
    'NT-50M (no overlap, logan, EWC 5, GC & Shannon)': 0.670079008369739,
    'NT-50M (no overlap, logan, EWC 5, GC & Shannon, 2k ctx.)': 0.6629377348882154,
    'NT-50M (overlap, logan, EWC 5, GC & Shannon)': 0.6581823078434047,
    'NT-50M (overlap, logan, EWC 5, GC & Shannon, 2k ctx.)': 0.6608261275028725,
    'NT-50M (overlap, logan, EWC 5, 2k ctx.)': 0.6569398898137684
}

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

def compare_one_fold(compare, filename):
    compare = dict(enumerate(compare))
    compare_results = {key: [] for key in compare.keys()}
    group = [get_normative_name(a) for a, b in compare.values()] + [get_normative_name(b) for a, b in compare.values()]

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

def compare_two_fold(compare, filename):
    compare = dict(enumerate(compare))
    compare_results_cls = {key: [] for key in compare.keys()}
    compare_results_mean = {key: [] for key in compare.keys()}
    compare_results = {key: [] for key in compare.keys()}

    group = [get_normative_name(a) for a, b, c in compare.values()] + [get_normative_name(b) for a, b, c in compare.values()] + [get_normative_name(c) for a, b, c in compare.values()]

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

def compare_across_groups_one_fold(compare, filename):
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
    padding = (ymax - ymin) * 0.1  # 10% padding
    ymin, ymax = ymin - padding, ymax + padding  # 10% padding

    for ax_idx, task_group in enumerate(TASK_GROUPS):
        ax = fig.add_subplot(axes[ax_idx])
        colors = [
            "green" if m > 0 else "red" if m < 0 else "gray"
            for m in group_results[task_group]['means']
        ]
        n = len(means)
        spacing = 3  # <- tweak this to taste
        x = np.arange(n) * spacing  # now ticks are at 0, 1.5, 3.0, …

        ax.bar(x, group_results[task_group]['means'], color=colors, width=0.8 * spacing)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i + 1}" for i in range(n)])
        ax.set_ylim(ymin, ymax)
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

def compare_across_groups_two_fold(compare, filename):
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
        ax_cls.set_xticks(x)
        ax_cls.set_xticklabels([f"C{i + 1}" for i in range(n)])
        ax_cls.set_ylim(ymin_cls, ymax_cls)
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
        ax_mean.set_xticks(x)
        ax_mean.set_xticklabels([f"C{i + 1}" for i in range(n)])
        ax_mean.set_ylim(ymin_mean, ymax_mean)
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


class CompareHandler(Enum):
    LOGAN = ('logan', compare_logan)
    OVERLAP = ('overlap', compare_overlap)
    SAMPLING = ('sh_gc', compare_sh_gc)
    PCA = ('pca', compare_pca)
    CONTEXT = ('context', compare_context)

if __name__ == '__main__':
    for handler in CompareHandler:
        filename, compare = handler.value
        if len(compare[0]) == 2:
            compare_fn = compare_one_fold
            grp_compare_fn = compare_across_groups_one_fold
        elif len(compare[0]) == 3:
            compare_fn = compare_two_fold
            grp_compare_fn = compare_across_groups_two_fold
        else:
            raise NotImplementedError
        compare_fn(compare, filename)
        grp_compare_fn(compare, filename)

