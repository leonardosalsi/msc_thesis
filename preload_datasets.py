from datasets import load_dataset
import config

TASKS = [
    "promoter_all",
    "promoter_tata",
    "promoter_no_tata",
    "enhancers",
    "enhancers_types",
    "splice_sites_all",
    "splice_sites_acceptors",
    "splice_sites_donors",
    "H2AFZ",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
    "H4K20me1"
]

# Cache directory
cache_dir = config.datasets_cache_dir

# Datasets
#load_dataset("InstaDeepAI/multi_species_genomes", cache_dir=cache_dir, trust_remote_code=True)
#for task in TASKS:
#    load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", name=task, cache_dir=cache_dir, trust_remote_code=True)
d1 = load_dataset("katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory", name="", cache_dir=cache_dir, trust_remote_code=True) #3
d2 = load_dataset("katarinagresova/Genomic_Benchmarks_demo_human_or_worm", cache_dir=cache_dir, trust_remote_code=True)
d3 = load_dataset("katarinagresova/Genomic_Benchmarks_human_ocr_ensembl", cache_dir=cache_dir, trust_remote_code=True)
d4 = load_dataset("katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark", cache_dir=cache_dir, trust_remote_code=True)
d5 = load_dataset("katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", cache_dir=cache_dir, trust_remote_code=True)
d6 = load_dataset("katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", cache_dir=cache_dir, trust_remote_code=True)
d7 = load_dataset("katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl", cache_dir=cache_dir, trust_remote_code=True)
d8 = load_dataset("katarinagresova/Genomic_Benchmarks_human_enhancers_cohn", cache_dir=cache_dir, trust_remote_code=True)
d9 = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters", cache_dir=cache_dir, trust_remote_code=True)

def get_info(ds):
    lengths = []
    labels = []
    test = ds['test']
    for d in test:
        lengths.append(len(d['seq']))
        labels.append(d['label'])

    lengths = list(set(lengths))
    labels = list(set(labels))

    print(lengths)
    print(labels)
    print("==============")

get_info(d1)
get_info(d2)
get_info(d3)
get_info(d4)
get_info(d5)
get_info(d6)
get_info(d7)
get_info(d8)
get_info(d9)