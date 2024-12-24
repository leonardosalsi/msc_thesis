import math

import matplotlib.pyplot as plt
import numpy as np
import json

with open('values.json', 'r') as file:
    data = json.load(file)

aliases = {
    'nucleotide-transformer-v2-50m-multi-species': 'NT-Multispecies V2 (50M)',
    'nucleotide-transformer-v2-100m-multi-species': 'NT-Multispecies V2 (100M)',
    'nucleotide-transformer-v2-250m-multi-species': 'NT-Multispecies V2 (250M)',
    'nucleotide-transformer-v2-500m-multi-species': 'NT-Multispecies V2 (500M)',
}

colors = {
    'nucleotide-transformer-v2-50m-multi-species': 'firebrick',
    'nucleotide-transformer-v2-100m-multi-species': 'darkorange',
    'nucleotide-transformer-v2-250m-multi-species': 'orchid',
    'nucleotide-transformer-v2-500m-multi-species': 'darkslateblue',
}

num_subgroups = len(data)
cols = 3
rows = math.ceil(num_subgroups / cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5), constrained_layout=True)

axes = axes.flatten()

for idx, (key, models) in enumerate(data.items()):
    ax = axes[idx]
    model_names = [aliases[model["model"]] for model in models]
    mcc_values = [model["mcc"] for model in models]
    bar_color = [colors[model["model"]] for model in models]

    # Bar plot
    x = np.arange(len(model_names))
    bars = ax.bar(x, mcc_values, color=bar_color)

    for bar, value in zip(bars, mcc_values):
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

    ax.set_title(
        key,
        fontsize=24,
        pad=10,
        loc="center"
    )

    ax.set_ylim(0, 1)

    col = idx % cols
    is_last_row = idx >= (rows - 1) * cols or idx + cols >= num_subgroups

    if is_last_row:

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=90, ha="right", fontsize=16)
    else:
        ax.set_xticks([])

for idx in range(len(data), len(axes)):
    fig.delaxes(axes[idx])

plt.show()
