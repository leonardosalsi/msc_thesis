from transformers import AutoTokenizer, AutoModelForMaskedLM
import config

def show_info(repo, model, tokenizer):
    print(repo)
    print(model)
    print(tokenizer)
    print("==============================================")

cache_dir = config.models_cache_dir

# Tokenizer and Model nucleotide-transformer-v2-500m-multi-species
t1 = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir=cache_dir, trust_remote_code=True)
m1 = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir=cache_dir, trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-250m-multi-species
t2 = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", cache_dir=cache_dir, trust_remote_code=True)
m2 = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", cache_dir=cache_dir, trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-100m-multi-species
t3 = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", cache_dir=cache_dir, trust_remote_code=True)
m3 = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", cache_dir=cache_dir, trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-50m-multi-species
t4 = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", cache_dir=cache_dir, trust_remote_code=True)
m4 = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", cache_dir=cache_dir, trust_remote_code=True)

show_info("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", m1, t1)
show_info("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", m2, t2)
show_info("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", m3, t3)
show_info("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", m4, t4)