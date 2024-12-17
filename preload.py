from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Tokenizer and Model nucleotide-transformer-v2-500m-multi-species
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir="/cluster/customapps/biomed/grlab/users/salsil/msc_thesis/models/nt-500m", trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir="/cluster/customapps/biomed/grlab/users/salsil/msc_thesis/datasets/nt-500m", trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-250m-multi-species
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", cache_dir="./models/nt-250m", trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", cache_dir="/cluster/customapps/biomed/grlab/users/salsil/msc_thesis/datasets/nt-250m", trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-100m-multi-species
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", cache_dir="./models/nt-100m", trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", cache_dir="/cluster/customapps/biomed/grlab/users/salsil/msc_thesis/datasets/nt-100m", trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-50m-multi-species
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", cache_dir="/cluster/customapps/biomed/grlab/users/salsil/msc_thesis/models/nt-50m", trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", cache_dir="/cluster/customapps/biomed/grlab/users/salsil/msc_thesis/datasets/nt-50m", trust_remote_code=True)

# Dataset
load_dataset("InstaDeepAI/multi_species_genomes", cache_dir="./datasets/multi-species-genomes", trust_remote_code=True)