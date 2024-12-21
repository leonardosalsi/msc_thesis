import torch
from downstream_tasks import TASKS as tasks, MODELS as model_names
from mcc import finetune_model_by_task_mcc
import random
import json

result_matrix = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU instead.")


for task in tasks:
    # Create a dictionary to hold results for this task
    task_results = {}

    for model_name in model_names:
        # Generate a random number for this model under the current task

        mcc = finetune_model_by_task_mcc(device, model_name, task)
        print(f"MCC of {model_name} on {task}: {mcc}")
        task_results[model_name] = mcc

    # Append the task and its results as a tuple (task, task_results)
    result_matrix.append({task['name']: task_results})


# Save the result_matrix to a JSON file
with open('result_matrix.json', 'w') as f:
    json.dump(result_matrix, f, indent=4)

