import os
from config import datasets_cache_dir

TASKS = [
    {'taskId': 1,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_all", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 2,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_tata", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 3,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "promoter_no_tata", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 4,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "enhancers", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 5,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "enhancers_types", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 3, 'grouping': 'Regulatory Elements'},
    {'taskId': 6,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_all", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 3, 'grouping': 'Splicing'},
    {'taskId': 7,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_acceptors", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Splicing'},
    {'taskId': 8,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "splice_sites_donors", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Splicing'},
    {'taskId': 9,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H2AFZ", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 10,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K27ac", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 11,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K27me3", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 12,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K36me3", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 13,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me1", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 14,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me2", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 15,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K4me3", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 16,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K9ac", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 17,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H3K9me3", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 18,'repo': "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", 'name': "H4K20me1", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Chromatin Profiles'},
    {'taskId': 19,'repo': "katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory", 'name': "gb_human_ensembl_regulatory", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 3, 'grouping': 'Regulatory Elements'},
    {'taskId': 20,'repo': "katarinagresova/Genomic_Benchmarks_demo_human_or_worm", 'name': "gb_demo_human_or_worm", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Cross-Species & Coding vs. Intergenic'},
    {'taskId': 21,'repo': "katarinagresova/Genomic_Benchmarks_human_ocr_ensembl", 'name': "gb_human_ocr_ensembl", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 22,'repo': "katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark", 'name': "gb_drosophila_enhancers_stark", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 23,'repo': "katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", 'name': "gb_dummy_mouse_enhancers_ensembl", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 24,'repo': "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", 'name': "gb_demo_coding_vs_intergenomic_seqs", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Cross-Species & Coding vs. Intergenic'},
    {'taskId': 25,'repo': "katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl", 'name': "gb_human_enhancers_ensembl", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 26,'repo': "katarinagresova/Genomic_Benchmarks_human_enhancers_cohn", 'name': "gb_human_enhancers_cohn", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 27,'repo': "katarinagresova/Genomic_Benchmarks_human_nontata_promoters", 'name': "gb_human_nontata_promoters", 'sequence_feature': 'seq', 'label_feature': 'label', 'num_labels': 2, 'grouping': 'Regulatory Elements'},
    {'taskId': 28,'repo': os.path.join(datasets_cache_dir, '5_utr_classification'), 'name': "utr5_ben_pat", 'sequence_feature': 'sequence', 'label_feature': 'label', 'num_labels': 2, 'grouping': '5UTR'},
]

TASK_GROUPS = [
    'Chromatin Profiles',
    'Cross-Species & Coding vs. Intergenic',
    'Regulatory Elements',
    'Splicing'
]

MODELS = {
    'default_multi_species_no_cont': 'NT-50M (no continual)',
    'default_multi_species_no_cont_100': 'NT-100M (no continual)',
    'default_multi_species_no_cont_250': 'NT-250M (no continual)',
    'default_multi_species_no_cont_500': 'NT-500M (no continual)',
    'default_multi_species': 'NT-50M (no overlap, multispecies)',
    'default_multi_species_2kb': 'NT-50M (no overlap, multispecies, 2k ctx.)',
    'overlap_multi_species': 'NT-50M (overlap, multispecies)',
    'overlap_multi_species_2kb': 'NT-50M (overlap, multispecies, 2k ctx.)',
    'overlap_logan_no_ewc': 'NT-50M (overlap, logan, no EWC)',
    'overlap_logan_ewc_0_5': 'NT-50M (overlap, logan, EWC 0.5)',
    'overlap_logan_ewc_1': 'NT-50M (overlap, logan, EWC 1)',
    'overlap_logan_ewc_2': 'NT-50M (overlap, logan, EWC 2)',
    'overlap_logan_ewc_5': 'NT-50M (overlap, logan, EWC 5)',
    'overlap_logan_ewc_10': 'NT-50M (overlap, logan, EWC 10)',
    'overlap_logan_ewc_25': 'NT-50M (overlap, logan, EWC 25)',
    'default_multi_species_sh_gc': 'NT-50M (no overlap, multispecies, GC & Shannon)',
    'default_multi_species_2kb_sh_gc': 'NT-50M (no overlap, multispecies, GC & Shannon, 2k ctx.)',
    'overlap_multi_species_sh_gc': 'NT-50M (overlap, multispecies, GC & Shannon)',
    'overlap_multi_species_2kb_sh_gc': 'NT-50M (overlap, multispecies, GC & Shannon, 2k ctx.)',
    'overlap_multi_species_pca_cls_256': 'NT-50M (overlap, multispecies, contrastive CLS)',
    'overlap_multi_species_pca_mean_256': 'NT-50M (overlap, multispecies, contrastive mean-pool)',
    'default_logan_no_ewc': 'NT-50M (no overlap, logan, no EWC)',
    'default_logan_ewc_0_5': 'NT-50M (no overlap, logan, EWC 0.5)',
    'default_logan_ewc_1': 'NT-50M (no overlap, logan, EWC 1)',
    'default_logan_ewc_2': 'NT-50M (no overlap, logan, EWC 2)',
    'default_logan_ewc_5': 'NT-50M (no overlap, logan, EWC 5)',
    'default_logan_ewc_10': 'NT-50M (no overlap, logan, EWC 10)',
    'default_logan_ewc_25': 'NT-50M (no overlap, logan, EWC 25)',
    'default_multi_species_pca_cls_256': 'NT-50M (no overlap, multispecies, contrastive CLS)',
    'default_multi_species_pca_mean_256': 'NT-50M (no overlap, multispecies, contrastive mean-pool)',
    'default_logan_ewc_5_2kb': 'NT-50M (no overlap, logan, EWC 5, 2k ctx.)',
    'overlap_logan_ewc_5_pca_cls_256': 'NT-50M (overlap, logan, EWC 5, contrastive CLS)',
    'overlap_logan_ewc_5_pca_mean_256': 'NT-50M (overlap, logan, EWC 5, contrastive mean-pool)',
    'default_logan_ewc_5_pca_cls_256': 'NT-50M (no overlap, logan, EWC 5, contrastive CLS)',
    'default_logan_ewc_5_pca_mean_256': 'NT-50M (no overlap, logan, EWC 5, contrastive mean-pool)',
    'default_logan_ewc_5_sh_gc':  'NT-50M (no overlap, logan, EWC 5, GC & Shannon)',
    'default_logan_ewc_5_2kb_sh_gc':  'NT-50M (no overlap, logan, EWC 5, GC & Shannon, 2k ctx.)',
    'overlap_logan_ewc_5_sh_gc':  'NT-50M (overlap, logan, EWC 5, GC & Shannon)',
    'overlap_logan_ewc_5_2kb_sh_gc':  'NT-50M (overlap, logan, EWC 5, GC & Shannon, 2k ctx.)',
    'overlap_logan_ewc_5_2kb': 'NT-50M (overlap, logan, EWC 5, 2k ctx.)'
}

TASK_DEFINITIONS = [
    {'name': 'promoter_all', 'alias': 'Promoter (all)'},
    {'name': 'promoter_tata', 'alias': 'Promoter (TATA)'},
    {'name': 'promoter_no_tata', 'alias': 'Promoter (non-TATA)'},
    {'name': 'enhancers', 'alias': 'Enhancers'},
    {'name': 'enhancers_types', 'alias': 'Enhancers (types)'},
    {'name': 'splice_sites_all', 'alias': 'Splice Sites (all)'},
    {'name': 'splice_sites_acceptors', 'alias': 'Splice Sites (acceptors)'},
    {'name': 'splice_sites_donors', 'alias': 'Splice Sites (donors)'},
    {'name': 'H2AFZ', 'alias': 'H2AFZ'},
    {'name': 'H3K27ac', 'alias': 'H3K27ac'},
    {'name': 'H3K27me3', 'alias': 'H3K27me3'},
    {'name': 'H3K36me3', 'alias': 'H3K36me3'},
    {'name': 'H3K4me1', 'alias': 'H3K4me1'},
    {'name': 'H3K4me2', 'alias': 'H3K4me2'},
    {'name': 'H3K4me3', 'alias': 'H3K4me3'},
    {'name': 'H3K9ac', 'alias': 'H3K9ac'},
    {'name': 'H3K9me3', 'alias': 'H3K9me3'},
    {'name': 'H4K20me1', 'alias': 'H4K20me1'},
    {'name': 'gb_human_ensembl_regulatory', 'alias': 'Human Ensemble (regulatory)'},
    {'name': 'gb_demo_human_or_worm', 'alias': 'Human or Worm (Demo)'},
    {'name': 'gb_human_ocr_ensembl', 'alias': 'Human Ensemble (ocr)'},
    {'name': 'gb_drosophila_enhancers_stark', 'alias': 'Enhancers (drosophila)'},
    {'name': 'gb_dummy_mouse_enhancers_ensembl', 'alias': 'Enhancers (mouse)'},
    {'name': 'gb_demo_coding_vs_intergenomic_seqs', 'alias': 'Coding vs Intergen. Seq.'},
    {'name': 'gb_human_enhancers_ensembl', 'alias': 'Enh. (human ensemble)'},
    {'name': 'gb_human_enhancers_cohn', 'alias': 'Enh. (human cohn)'},
    {'name': 'gb_human_nontata_promoters', 'alias': 'Prom. Human (no TATA)'}
]

def get_task_by_name(name):
    for task in TASKS:
        if task['name'] == name:
            return task

if __name__ == '__main__':
    for t in TASK_DEFINITIONS:
        print(t['name'])

task_permutation = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 5, 1, 3, 2, 6, 7, 8, 27, 19, 20, 21, 22, 23, 24, 25, 26]