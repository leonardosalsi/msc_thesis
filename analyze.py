import os
import json

data = {}
data_location = "./data"

tests = ['promoter_all', 'promoter_tata', 'promoter_no_tata', 'enhancers', 'enhancers_types', 'splice_sites_all', 'splice_sites_acceptors', 'splice_sites_donors', 'H2AFZ', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K9me3', 'H4K20me1', 'Genomic_Benchmarks_human_ensembl_regulatory', 'Genomic_Benchmarks_demo_human_or_worm', 'Genomic_Benchmarks_human_ocr_ensembl', 'Genomic_Benchmarks_drosophila_enhancers_stark', 'Genomic_Benchmarks_dummy_mouse_enhancers_ensembl', 'Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs', 'Genomic_Benchmarks_human_enhancers_ensembl', 'Genomic_Benchmarks_human_enhancers_cohn', 'Genomic_Benchmarks_human_nontata_promoters']


def check_data_integrity(_content):
    keys = list(_content.keys())
    tests.sort()
    keys.sort()
    for i in range(0, len(tests)):
        if (tests[i] != keys[i]):
            return False
    return True

# Iterate through all files in the folder
for filename in os.listdir(data_location):
    file_path = os.path.join(data_location, filename)
    # Check if it's a file
    if os.path.isfile(file_path):
        model_name = filename.split(".")[0]
        with open(file_path, 'r') as file:
            content = file.read()
            content = json.loads(content)
            is_ok = check_data_integrity(content)
            if not is_ok:
                print("File '{}' is invalid".format(file_path))
            else:
                data[model_name] = content

with open('test.json', 'w') as f:
    json.dump(data, f, indent=4)