from downstream_tasks import MODELS, TASKS

LOGLEVEL = 22

def get_chunk_size_folder_name(chunk_size) -> str:
    return (str(chunk_size / 1000).replace(".", "_") + "kbp").replace("_0", "")


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

def get_task_by_id(taskId):
    for task in TASKS:
        if task['taskId'] == taskId:
            return task
    return None