TASKS = [
    {'taskId': 1,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_all", 'alias': "promoter_all", 'len': 300, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Promoter (all)'},
    {'taskId': 2,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_tata", 'alias': "promoter_tata", 'len': 300, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Promoter (TATA)'},
    {'taskId': 3,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_no_tata", 'alias': "promoter_no_tata", 'len': 300, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Promoter (non-TATA)'},
    {'taskId': 4,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "enhancers", 'alias': "enhancers", 'len': 400, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Enhancers'},
    {'taskId': 5,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "enhancers_types", 'alias': "enhancers_types", 'len': 400, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 3, 'data_alias': 'Enhancers (types)'},
    {'taskId': 6,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_all", 'alias': "splice_sites_all", 'len': 600, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 3, 'data_alias': 'Splice Sites (all)'},
    {'taskId': 7,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_acceptors", 'alias': "splice_sites_acceptors", 'len': 600, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Splice Sites (acceptors)'},
    {'taskId': 8,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_donors", 'alias': "splice_sites_donors", 'len': 600, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Splice Sites (donors)'},
    {'taskId': 9,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H2AFZ", 'alias': "H2AFZ", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H2AFZ'},
    {'taskId': 10,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K27ac", 'alias': "H3K27ac", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H3K27ac'},
    {'taskId': 11,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K27me3", 'alias': "H3K27me3", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H3K27me3'},
    {'taskId': 12,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K36me3", 'alias': "H3K36me3", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H3K36me3'},
    {'taskId': 13,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me1", 'alias': "H3K4me1", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H3K4me1'},
    {'taskId': 14,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me2", 'alias': "H3K4me2", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H3K4me2'},
    {'taskId': 15,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me3", 'alias': "H3K4me3", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H3K4me3'},
    {'taskId': 16,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K9ac", 'alias': "H3K9ac", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H3K9ac'},
    {'taskId': 17,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K9me3", 'alias': "H3K9me3", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H3K9me3'},
    {'taskId': 18,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H4K20me1", 'alias': "H4K20me1", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'H4K20me1'},
    {'taskId': 19,'repo': "katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory", 'name': "", 'alias': "Genomic_Benchmarks_human_ensembl_regulatory", 'len': 600, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 3, 'data_alias': 'Human Ensemble (regulatory)'}, #len 80 - 800
    {'taskId': 20,'repo': "katarinagresova/Genomic_Benchmarks_demo_human_or_worm", 'name': "", 'alias': "Genomic_Benchmarks_demo_human_or_worm", 'len': 200, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Human or Worm (Demo)'},
    {'taskId': 21,'repo': "katarinagresova/Genomic_Benchmarks_human_ocr_ensembl", 'name': "", 'alias': "Genomic_Benchmarks_human_ocr_ensembl", 'len': 400, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Human Ensemble (ocr)'}, #80-600
    {'taskId': 22,'repo': "katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark", 'name': "", 'alias': "Genomic_Benchmarks_drosophila_enhancers_stark", 'len': 1000, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Enhancers (drosophila)'}, #500-2500
    {'taskId': 23,'repo': "katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", 'name': "", 'alias': "Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", 'len': 500, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Enhancers (mouse)'}, #1000-4000
    {'taskId': 24,'repo': "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", 'name': "", 'alias': "Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", 'len': 200, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Coding vs Intergenomic Seq.'},
    {'taskId': 25,'repo': "katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl", 'name': "", 'alias': "Genomic_Benchmarks_human_enhancers_ensembl", 'len': 300, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Enhancers (human ensemble)'}, #1-600
    {'taskId': 26,'repo': "katarinagresova/Genomic_Benchmarks_human_enhancers_cohn", 'name': "", 'alias': "Genomic_Benchmarks_human_enhancers_cohn", 'len': 500, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Enhancers (human cohn)'},
    {'taskId': 27,'repo': "katarinagresova/Genomic_Benchmarks_human_nontata_promoters", 'name': "", 'alias': "Genomic_Benchmarks_human_nontata_promoters", 'len': 251, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'data_alias': 'Promoter Human (no TATA)'}
]

MODELS = [
    {'modelId': 1,'repo': "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", 'name': 'nucleotide-transformer-v2-50m-multi-species', 'data_alias': 'NT-MS V2 (50M)'},
    {'modelId': 2,'repo': "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", 'name': 'nucleotide-transformer-v2-100m-multi-species', 'data_alias': 'NT-MS V2 (100M)'},
    {'modelId': 3,'repo': "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", 'name': 'nucleotide-transformer-v2-250m-multi-species', 'data_alias': 'NT-MS V2 (250M)'},
    {'modelId': 4,'repo': "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 'name': 'nucleotide-transformer-v2-500m-multi-species', 'data_alias': 'NT-MS V2 (500M)'},
    {'modelId': 5,'repo': "InstaDeepAI/nucleotide-transformer-500m-1000g", 'name': 'nucleotide-transformer-500m-1000g', 'data_alias': 'NT-1000g (500M)'},
    {'modelId': 6,'repo': "InstaDeepAI/nucleotide-transformer-500m-human-ref", 'name': 'nucleotide-transformer-500m-human-ref', 'data_alias': 'NT-Human-Ref. (500M)'},
]

def generate_file(job_name, modelId, taskId, no_lora=False, random_weights=False):
    args = ''
    if no_lora:
        args += ' --no-lora'

    if random_weights:
        args += ' --random-weights'

    content = f'''#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output=out/{job_name}.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate gpu_env

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
python /cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/msc_thesis/evaluate_model_mcc.py {modelId} {taskId}{args} '''
    with open(f'jobs/evaluate_mcc/{job_name}.sh', 'w') as rsh:
        rsh.writelines(content)

def generate_jobs():
    jobs = []
    for model in MODELS:
        for task in TASKS:
            generate_file(f"{model['name']}-{task['alias']}-lora", model['modelId'], task['taskId'])
            jobs.append(f"{model['name']}-{task['alias']}-lora")
        if (model['modelId'] == 1 or model['modelId'] == 4):
            for task in TASKS:
                generate_file(f"{model['name']}-random-weights-{task['alias']}-lora", model['modelId'], task['taskId'], random_weights=True)
                jobs.append(f"{model['name']}-random-weights-{task['alias']}-lora")
    """
    for model in MODELS:
        for task in TASKS:
            generate_file(f"{model['name']}-{task['alias']}", model['modelId'], task['taskId'], no_lora=True)
            jobs.append(f"{model['name']}-{task['alias']}-{task['alias']}")
        if (model['modelId'] == 1 or model['modelId'] == 4):
            for task in TASKS:
                generate_file(f"{model['name']}-random-weights-{task['alias']}", model['modelId'], task['taskId'], no_lora=True, random_weights=True)
                jobs.append(f"{model['name']}-random-weights-{task['alias']}")
    """
    content = '''#!/bin/bash\n\n'''

    for job in jobs:
        content += f"sbatch jobs/evaluate_mcc/{job}.sh\n"

    with open(f'evaluate_all.sh', 'w') as rsh:
        rsh.writelines(content)

generate_jobs()