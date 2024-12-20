from datasets import load_dataset
import config

def show_info(repo, dataset):
    print(repo)
    print(dataset)
    print("==============================================")

# Cache directory
cache_dir = config.datasets_cache_dir

# Datasets
d1 = load_dataset("InstaDeepAI/multi_species_genomes", cache_dir=cache_dir, trust_remote_code=True)
d2 = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", cache_dir=cache_dir, trust_remote_code=True)
d3 = load_dataset("katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory", cache_dir=cache_dir, trust_remote_code=True)
d4 = load_dataset("katarinagresova/Genomic_Benchmarks_demo_human_or_worm", cache_dir=cache_dir, trust_remote_code=True)
d5 = load_dataset("katarinagresova/Genomic_Benchmarks_human_ocr_ensembl", cache_dir=cache_dir, trust_remote_code=True)
d6 = load_dataset("katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark", cache_dir=cache_dir, trust_remote_code=True)
d7 = load_dataset("katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", cache_dir=cache_dir, trust_remote_code=True)
d8 = load_dataset("katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", cache_dir=cache_dir, trust_remote_code=True)
d9 = load_dataset("katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl", cache_dir=cache_dir, trust_remote_code=True)
d10 = load_dataset("katarinagresova/Genomic_Benchmarks_human_enhancers_cohn", cache_dir=cache_dir, trust_remote_code=True)
d11 = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters", cache_dir=cache_dir, trust_remote_code=True)

show_info("InstaDeepAI/multi_species_genomes", d1)
show_info("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", d2)
show_info("katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory", d3)
show_info("katarinagresova/Genomic_Benchmarks_demo_human_or_worm", d4)
show_info("katarinagresova/Genomic_Benchmarks_human_ocr_ensembl", d5)
show_info("katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark", d6)
show_info("katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", d7)
show_info("katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", d8)
show_info("katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl", d9)
show_info("katarinagresova/Genomic_Benchmarks_human_enhancers_cohn", d10)
show_info("katarinagresova/Genomic_Benchmarks_human_nontata_promoters", d11)