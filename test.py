import os
from downstream_tasks import TASKS, MODELS

task_permutation = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 5, 1, 3, 2, 6, 7, 8, 27, 19, 20, 21, 22, 23, 24, 25, 26]
model_permutation = [1.5, 4.5, 2, 1, 3, 5, 6, 4]

directory = 'data'
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

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

data = {}

for taskId in task_permutation:
    task = get_task_by_id(taskId)
    task_files = list(filter(lambda filename: task['alias'] in filename, files))
    print(task['alias'], len(task_files), task_files)

"""
for taskId in task_permutation:
    task = get_task_by_id(taskId)
    task_files = list(filter(lambda filename: task['alias'] in filename, files))
    for modelId in model_permutation:
        model = get_model_by_id(int(modelId))
        model_task_files = list(filter(lambda filename: model['name'] in filename, task_files))
        mode = ""
        try:
            if modelId == int(modelId):
                file = list(filter(lambda filename: '-with-random-weights' not in filename, model_task_files))[0]
            else:
                file = list(filter(lambda filename: '-with-random-weights' in filename, model_task_files))[0]
                mode = " with random weights"
        except:
            print("=======")
            print("ERROR ON " + task['alias'] + "   " + model['alias'])
            print(model_task_files)
        print("=======")
        print(task['alias'])
        print(model['alias'] + mode)
        print(file)
"""
