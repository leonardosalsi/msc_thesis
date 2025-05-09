import datetime
import os
import sys
import torch
from util import init_logger, LOGLEVEL
import logging as pyLogging

from utils.model_definitions import TASKS

LOGLEVEL = 22
LOGGER = init_logger()

def print_args(args, title):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if ("LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] == "0") or ("LOCAL_RANK" not in os.environ):
        LOGGER.log(LOGLEVEL, "\n" + "=" * 80)
        LOGGER.log(LOGLEVEL, f"{title} - {timestamp}".center(80))
        LOGGER.log(LOGLEVEL, "=" * 80)
        for arg, value in sorted(vars(args).items()):
            LOGGER.log(LOGLEVEL, "{:<25}: {}".format(arg, value))
        LOGGER.log(LOGLEVEL, "=" * 80 + "\n")
    return timestamp


def get_device():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        LOGGER.log(LOGLEVEL, f"Using GPU (local rank {local_rank}): {torch.cuda.get_device_name(local_rank)}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            LOGGER.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
            LOGGER.log(LOGLEVEL, f"Number of GPUs available: {torch.cuda.device_count()}")
        else:
            LOGGER.log(LOGLEVEL, "GPU not available. Using CPU instead.")

    torch.cuda.empty_cache()
    return device

def compute_metrics(eval_preds):
    return {"eval_loss": eval_preds.loss}

def check_folders(base_path):
    train_path = os.path.join(base_path, "train")
    validation_path = os.path.join(base_path, "validation")

    # Check if both directories exist
    if not (os.path.isdir(train_path) and os.path.isdir(validation_path)):
        print("Train path:", train_path)
        print("Validation path:", validation_path)
        print("Error: Required folders 'train' and 'validation' not found in the provided path.")
        sys.exit(1)

    return train_path, validation_path

def get_chunk_size_file_name(chunk_size) -> str:
    if chunk_size is not None:
        return (str(chunk_size / 1000).replace(".", "_") + "kbp").replace("_0", "")
    else:
        return ""

def get_filtered_dataset_name(chunk_size, shannon, gc) -> str:
    shannon_txt = ""
    gc_txt = ""
    if shannon is not None:
        shannon_txt = f"_sh_{shannon[0]}_{shannon[1]}"
    if gc is not None:
        gc_txt = f"_gc_{gc[0]}_{gc[1]}"
    chunk_size_file_name = get_chunk_size_file_name(chunk_size)
    return f"{chunk_size_file_name}{shannon_txt}{gc_txt}".replace(".", "_")

def init_logger():
    pyLogging.basicConfig(
        filename=f"/dev/null",
        filemode="a",
        level=LOGLEVEL,  # Log level
        format="%(message)s"
    )
    logger = pyLogging.getLogger()
    console_handler = pyLogging.StreamHandler()
    console_handler.setLevel(LOGLEVEL)
    console_handler.setFormatter(pyLogging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger

def get_task_by_id(taskId):
    for task in TASKS:
        if task['taskId'] == taskId:
            return task
    return None