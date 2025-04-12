import datetime
import os
import sys
import torch

def print_args(args, title):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print("\n" + "=" * 80)
    print(f"{title} - {timestamp}".center(80))
    print("=" * 80)
    for arg, value in sorted(vars(args).items()):
        print("{:<25}: {}".format(arg, value))
    print("=" * 80 + "\n")
    return timestamp

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    return device

def compute_metrics(eval_preds):
    return {"eval_loss": eval_preds.loss}

def check_folders(base_path):

    train_path = os.path.join(base_path, "train")
    validation_path = os.path.join(base_path, "validation")

    # Check if both directories exist
    if not (os.path.isdir(train_path) and os.path.isdir(validation_path)):
        print("Error: Required folders 'train' and 'validation' not found in the provided path.")
        sys.exit(1)

    return train_path, validation_path