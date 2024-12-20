from datasets import load_dataset

def show_data(repo, dataset):
    print(repo)
    print(dataset)
    print("==============================================")

# Cache directory
cache_dir = "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/datasets"

# Dataset
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

show_data("InstaDeepAI/multi_species_genomes", d1)
show_data("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", d2)
show_data("katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory", d3)
show_data("katarinagresova/Genomic_Benchmarks_demo_human_or_worm", d4)
show_data("katarinagresova/Genomic_Benchmarks_human_ocr_ensembl", d5)
show_data("katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark", d6)
show_data("katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", d7)
show_data("katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", d8)
show_data("katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl", d9)
show_data("katarinagresova/Genomic_Benchmarks_human_enhancers_cohn", d10)
show_data("katarinagresova/Genomic_Benchmarks_human_nontata_promoters", d11)