TASKS = [
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_all", 'len': 300, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_tata", 'len': 300, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_no_tata", 'len': 300, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "enhancers", 'len': 400, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "enhancers_types", 'len': 400, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 3},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_all", 'len': 600, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 3},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_acceptors", 'len': 600, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_donors", 'len': 600, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H2AFZ", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K27ac", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K27me3", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K36me3", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me1", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me2", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me3", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K9ac", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K9me3", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H4K20me1", 'len': 1000, 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory", 'name': "", 'len': 600, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 3}, #len 80 - 800
    {'repo': "katarinagresova/Genomic_Benchmarks_demo_human_or_worm", 'name': "", 'len': 200, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "katarinagresova/Genomic_Benchmarks_human_ocr_ensembl", 'name': "", 'len': 400, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2}, #80-600
    {'repo': "katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark", 'name': "", 'len': 1000, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2}, #500-2500
    {'repo': "katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", 'name': "", 'len': 500, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2}, #1000-4000
    {'repo': "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", 'name': "", 'len': 200, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl", 'name': "", 'len': 300, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2}, #1-600
    {'repo': "katarinagresova/Genomic_Benchmarks_human_enhancers_cohn", 'name': "", 'len': 500, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2},
    {'repo': "katarinagresova/Genomic_Benchmarks_human_nontata_promoters", 'name': "", 'len': 251, 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2}
]


MODELS = [
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    #"InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    #"InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    #"InstaDeepAI/nucleotide-transformer-500m-human-ref"
    #"InstaDeepAI/nucleotide-transformer-500m-1000g"
]