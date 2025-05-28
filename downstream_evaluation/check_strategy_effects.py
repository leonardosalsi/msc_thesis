from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from downstream_evaluation.groupings import MODEL_DICT, _collect_benchmark_data
from downstream_evaluation.plot_mcc import prepare_data_for_visualization

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
    ('NT-50M (no overlap, logan, EWC 5)', 'NT-50M (no overlap, multispecies)'),
    ('NT-50M (overlap, logan, EWC 5)', 'NT-50M (overlap, multispecies)'),
    ('NT-50M (no overlap, logan, EWC 5, 2k ctx.)' , 'NT-50M (no overlap, multispecies, 2k ctx.)'),
    ('NT-50M (overlap, logan, EWC 5, 2k ctx.)' , 'NT-50M (overlap, multispecies, 2k ctx.)'),
    ('NT-50M (no overlap, logan, EWC 5, contrastive CLS)', 'NT-50M (no overlap, multispecies, contrastive CLS)'),
    ('NT-50M (no overlap, logan, EWC 5, contrastive mean-pool)', 'NT-50M (no overlap, multispecies, contrastive mean-pool)'),
    ('NT-50M (overlap, logan, EWC 5, contrastive CLS)', 'NT-50M (overlap, multispecies, contrastive CLS)'),
    ('NT-50M (overlap, logan, EWC 5, contrastive mean-pool)', 'NT-50M (overlap, multispecies, contrastive mean-pool)'),
    ('NT-50M (no overlap, logan, EWC 5, GC & Shannon)', 'NT-50M (no overlap, multispecies, GC & Shannon)'),
    ('NT-50M (overlap, logan, EWC 5, GC & Shannon)', 'NT-50M (no overlap, multispecies, GC & Shannon)'),
    ('NT-50M (no overlap, logan, EWC 5, GC & Shannon, 2k ctx.)' ,'NT-50M (no overlap, multispecies, GC & Shannon, 2k ctx.)'),
    ('NT-50M (overlap, logan, EWC 5, GC & Shannon, 2k ctx.)' ,'NT-50M (overlap, multispecies, GC & Shannon, 2k ctx.)')
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

def calc_diff(a, b):
    return (a - b) / b * 100

def get_normative_name(model_alias):
    for m in MODEL_DICT:
        if MODEL_DICT[m] == model_alias:
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
            diff = calc_diff(r_1, r_2)
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
    plt.savefig(f"compare_{filename}_mcc.pdf")
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
            diff_cls = calc_diff(r_2, r_1)
            diff_mean = calc_diff(r_3, r_1)
            diff = calc_diff(r_2, r_3)
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
    plt.savefig(f"compare_{filename}_mcc.pdf")
    plt.show()

class CompareHandler(Enum):
    LOGAN = ('logan', compare_logan)
    OVERLAP = ('overlap', compare_overlap)
    SAMPLING = ('sh_gc', compare_sh_gc)
    PCA = ('pca', compare_pca)
    CONTEXT = ('context', compare_context)

if __name__ == '__main__':
    handler = CompareHandler.CONTEXT

    filename, compare = handler.value
    if len(compare[0]) == 2:
        compare_one_fold(compare, filename)
    elif len(compare[0]) == 3:
        compare_two_fold(compare, filename)
