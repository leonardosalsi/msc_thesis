from datasets import load_dataset

# Dataset
#load_dataset("InstaDeepAI/multi_species_genomes", cache_dir="/cluster/customapps/biomed/grlab/users/salsil/msc_thesis/datasets/multi_species_genomes", trust_remote_code=True)
#load_dataset("InstaDeepAI/multi_species_genomes", cache_dir="/cluster/customapps/biomed/grlab/users/salsil/msc_thesis/datasets/multi_species_genomes", trust_remote_code=True)
dataset = load_dataset("InstaDeepAI/multi_species_gednomes", cache_dir="/cluster/project/grlab/tmp/petagraph/dataset", trust_remote_code=True)
print(dataset[0])