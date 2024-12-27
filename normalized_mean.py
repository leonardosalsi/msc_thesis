import json
import numpy as np

import matplotlib.pyplot as plt

# Load the JSON data
with open("values.json", "r") as file:
    data = json.load(file)

# Collect all tasks and their MCC values per model
models = {}
for task, entries in data.items():
    # Extract MCC values for normalization
    mcc_values = [entry["mcc"] for entry in entries]
    min_mcc = min(mcc_values)
    max_mcc = max(mcc_values)

    normalized_values = [
        (entry["mcc"] - min_mcc) / (max_mcc - min_mcc) if max_mcc != min_mcc else 0
        for entry in entries
    ]

    for entry, norm_mcc in zip(entries, normalized_values):
        model = entry["model"]
        if model not in models:
            models[model] = []
        models[model].append(norm_mcc)

normalized_means = [(model, np.mean(values)) for model, values in models.items()]
model_names, mean_values = zip(*normalized_means)

colors = [
    'firebrick',
    'darkorange',
    'orchid',
    'darkslateblue',
    "mediumvioletred",
    "saddlebrown",
]

# Create the bar chart

plt.figure(figsize=(9, 6))

plt.bar(model_names, mean_values, color=colors)


# Formatting

plt.xticks(rotation=90, ha="right", fontsize=10)

plt.yticks(fontsize=10)

plt.ylim(0, 1)

plt.xlabel("", fontsize=12)

plt.ylabel("Normalized Mean MCC", fontsize=12)

plt.title("", fontsize=14)

plt.tight_layout()



# Show the graph

plt.show()