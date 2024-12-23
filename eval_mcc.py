import torch
import logging
from downstream_tasks import TASKS as tasks, MODELS as model_names
from mcc import finetune_model_by_task_mcc
import json
from config import LOGLEVEL

logging.basicConfig(
    filename="eval_mcc.log",
    filemode="w",                 # Overwrite log file on each run
    level=LOGLEVEL,           # Log level
    format="%(message)s"
)
logger = logging.getLogger()

console_handler = logging.StreamHandler()
console_handler.setLevel(LOGLEVEL)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)
result_matrix = []

logger.log(LOGLEVEL, "Getting device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

for task in tasks:
    task_results = {}

    for model_name in model_names:
        logger.log(LOGLEVEL, f"{model_name} on {task['repo']}=>{task['name']}")
        mcc = finetune_model_by_task_mcc(logger, device, model_name, task)
        logger.log(LOGLEVEL, f"MCC of {model_name} on {task['name']}=>{task['name']}: {mcc}")
        task_results[model_name] = mcc

    result_matrix.append({task['name']: task_results})

output_file = 'result_matrix.json'
with open(output_file, 'w') as f:
    json.dump(result_matrix, f, indent=4)
logger.info(f"Results saved to {output_file}")
