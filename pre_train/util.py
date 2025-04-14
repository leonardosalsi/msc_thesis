import datetime
import os
import sys
import torch

from util import init_logger, LOGLEVEL

def print_args(args, title):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger = init_logger()
    if ("LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] == "0") or ("LOCAL_RANK" not in os.environ):
        logger.log(LOGLEVEL, "\n" + "=" * 80)
        logger.log(LOGLEVEL, f"{title} - {timestamp}".center(80))
        logger.log(LOGLEVEL, "=" * 80)
        for arg, value in sorted(vars(args).items()):
            logger.log(LOGLEVEL, "{:<25}: {}".format(arg, value))
        logger.log(LOGLEVEL, "=" * 80 + "\n")
    return timestamp


def get_device():
    logger = init_logger()
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        logger.log(LOGLEVEL, f"Using GPU (local rank {local_rank}): {torch.cuda.get_device_name(local_rank)}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            logger.log(LOGLEVEL, f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.log(LOGLEVEL, f"Number of GPUs available: {torch.cuda.device_count()}")
            logger.log(LOGLEVEL, f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        else:
            logger.log(LOGLEVEL, "GPU not available. Using CPU instead.")

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