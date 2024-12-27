import json
import numpy as np

import matplotlib.pyplot as plt

with open("values.json", "r") as file:
    data = json.load(file)

models = {}
for task, entries in data.items():
    for entry in entries:
        model = entry["model"]
        if model not in models:
            models[model] = []
        models[model].append(entry["mcc"])

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