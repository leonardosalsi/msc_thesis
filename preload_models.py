from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

cache_dir = config.models_cache_dir
print(cache_dir)
# Tokenizer and Model nucleotide-transformer-v2-500m-multi-species
AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)


# Tokenizer and Model nucleotide-transformer-v2-250m-multi-species
AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)

# Tokenizer and Model nucleotide-transformer-v2-100m-multi-species
AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)


# Tokenizer and Model nucleotide-transformer-v2-50m-multi-species
AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", cache_dir=cache_dir, trust_remote_code=True, local_files_only=True)

# Tokenizer and Model nucleotide-transformer-v2-50m-multi-species
AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g", cache_dir=cache_dir, trust_remote_code=True)
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g", cache_dir=cache_dir, trust_remote_code=True)

# Tokenizer and Model nucleotide-transformer-v2-50m-multi-species
AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", cache_dir=cache_dir, trust_remote_code=True)
AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", cache_dir=cache_dir, trust_remote_code=True)