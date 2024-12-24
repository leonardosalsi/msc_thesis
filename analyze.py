import json

# Function to split a file into groups of three lines
def split_into_groups_of_three(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Group lines into chunks of three
    grouped_lines = [lines[i:i + 3] for i in range(0, len(lines), 3)]
    return grouped_lines


# Example usage
filename = "eval_mcc.log"  # Replace with your file path
groups = split_into_groups_of_three(filename)

values = {}
train_eval = {}

for idx, group in enumerate(groups, start=1):
    model_name = group[0].split(" ")[0].split("/")[1]
    task = group[0].split(" ")[-1].replace("\n","")
    mcc = float(group[2].split(" ")[-1])
    hist = json.loads(group[1].replace("'", "\""))
    if "InstaDeepAI" in task:
        task = task.split("=>")[-1]
    elif "katarinagresova" in task:
        task = task.split("/")[1].replace("=>","").lower()
    train = []
    eval = []
    for idx, group in enumerate(hist):
        if idx % 2 == 0:
            train.append(group)
        else:
            eval.append(group)

    entry = {'model': model_name, 'mcc': mcc}
    tr_ev_entry = {'model': model_name, 'train': train, 'eval': eval}
    if task not in values:
        values[task] = []
    if task not in train_eval:
        train_eval[task] = []
    values[task].append(entry)
    train_eval[task].append(tr_ev_entry)

with open("values.json", "w") as json_file:
    json.dump(values, json_file, indent=4)
with open("train_eval.json", "w") as json_file:
    json.dump(train_eval, json_file, indent=4)