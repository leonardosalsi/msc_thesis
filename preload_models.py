from transformers import AutoTokenizer, AutoModelForMaskedLM
import config

cache_dir = config.models_cache_dir
print(cache_dir)
# Tokenizer and Model nucleotide-transformer-v2-500m-multi-species
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir=cache_dir, trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir=cache_dir, trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-250m-multi-species
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", cache_dir=cache_dir, trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", cache_dir=cache_dir, trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-100m-multi-species
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", cache_dir=cache_dir, trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", cache_dir=cache_dir, trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-50m-multi-species
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", cache_dir=cache_dir, trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", cache_dir=cache_dir, trust_remote_code=True)