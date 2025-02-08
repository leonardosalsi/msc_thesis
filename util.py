from downstream_tasks import MODELS, TASKS, PRETRAINED_MODELS

LOGLEVEL = 22



def get_chunk_size_file_name(chunk_size) -> str:
    return (str(chunk_size / 1000).replace(".", "_") + "kbp").replace("_0", "")

def get_filtered_dataset_name(chunk_size, shannon, gc) -> str:
    shannon_txt = ""
    gc_txt = ""
    if shannon is not None:
        shannon_txt = f"_sh_{shannon[0]}_{shannon[1]}"
    if gc is not None:
        gc_txt = f"_gc_{gc[0]}_{gc[1]}"
    chunk_size_file_name = get_chunk_size_file_name(chunk_size)
    return f"{chunk_size_file_name}{shannon_txt}{gc_txt}".replace(".", "_")

import logging as pyLogging
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


def get_model_by_id(modelId):
    for model in MODELS:
        if model['modelId'] == modelId:
            return model
    return None

def get_pretrained_model_by_id(modelId):
    print(modelId)
    for model in PRETRAINED_MODELS:
        if model['modelId'] == modelId:
            return model
    return None

def get_task_by_id(taskId):
    for task in TASKS:
        if task['taskId'] == taskId:
            return task
    return None