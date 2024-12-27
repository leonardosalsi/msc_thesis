import json

aliases_model = {
    "nucleotide-transformer-500m-human-ref": "NT-Human-Ref. (500M)",
    "nucleotide-transformer-500m-1000g": "NT-1000g (500M)",
    'nucleotide-transformer-v2-50m-multi-species': 'NT-Multispecies V2 (50M)',
    'nucleotide-transformer-v2-100m-multi-species': 'NT-Multispecies V2 (100M)',
    'nucleotide-transformer-v2-250m-multi-species': 'NT-Multispecies V2 (250M)',
    'nucleotide-transformer-v2-500m-multi-species': 'NT-Multispecies V2 (500M)',
}

aliases_tasks = {
    "promoter_all": "Promoters",
    "promoter_tata": "Promoters (TATA)",
    "promoter_no_tata": "Promoters (non-TATA)",
    "enhancers": "Enhancers",
    "enhancers_types": "Enhancers (types)",
    "splice_sites_all": "Splicing Both",
    "splice_sites_acceptors": "Splicing Acceptors",
    "splice_sites_donors": "Splicing Donors",
    "H2AFZ": "H2AFZ",
    "H3K27ac": "H3K27ac",
    "H3K27me3": "H3K27me3",
    "H3K36me3": "H3K36me3",
    "H3K4me1": "H3K4me1",
    "H3K4me2": "H3K4me2",
    "H3K4me3": "H3K4me3",
    "H3K9ac": "H3K9ac",
    "H3K9me3": "H3K9me3",
    "H4K20me1": "H4K20me1",
    "Genomic_Benchmarks_human_ensembl_regulatory": "Human Regulatory (Enseble)" ,
    "Genomic_Benchmarks_demo_human_or_worm": "Human or Worm",
    "Genomic_Benchmarks_human_ocr_ensembl": "OCR (Human)",
    "Genomic_Benchmarks_drosophila_enhancers_stark": "Enhancers Drosophila",
    "Genomic_Benchmarks_dummy_mouse_enhancers_ensembl": "Enhancers Mouse (Ensemble)",
    "Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs": "Coding vs Intergen. Segments.",
    "Genomic_Benchmarks_human_enhancers_ensembl": "Enhancers Human (Ensemble)",
    "Genomic_Benchmarks_human_enhancers_cohn": "Enhancers Human (Cohn)",
    "Genomic_Benchmarks_human_nontata_promoters": "Promoters Human (non-TATA)",
}

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

values_tmp = {}
train_eval = {}


for idx, group in enumerate(groups, start=1):
    model_name = group[0].split(" ")[0].split("/")[1]
    task = group[0].split(" ")[-1].replace("\n","")
    mcc = float(group[2].split(" ")[-1])
    hist = json.loads(group[1].replace("'", "\""))
    if "InstaDeepAI" in task:
        task = task.split("=>")[-1]
    elif "katarinagresova" in task:
        task = task.split("/")[1].replace("=>","")
    train = []
    eval = []
    for idx, group in enumerate(hist):
        if idx % 2 == 0:
            train.append(group)
        else:
            eval.append(group)

    entry = {'model': model_name, 'mcc': mcc}
    tr_ev_entry = {'model': model_name, 'train': train, 'eval': eval}
    if task not in values_tmp:
        values_tmp[task] = []
    if task not in train_eval:
        train_eval[task] = []
    values_tmp[task].append(entry)
    train_eval[task].append(tr_ev_entry)

values = {}

for k_tasks in aliases_tasks.keys():
    val = aliases_tasks[k_tasks]
    _tsk = values_tmp[k_tasks]
    values[val] = []
    for k_models in aliases_model.keys():
        _v = aliases_model[k_models]
        entry = next((item for item in _tsk if item["model"] == k_models), None)
        entry['model'] = _v
        values[val].append(entry)


with open("values.json", "w") as json_file:
    json.dump(values, json_file, indent=4)
with open("train_eval.json", "w") as json_file:
    json.dump(train_eval, json_file, indent=4)