
# Datasets and Models

Datasets are downloaded using the [Hugging Face `datasets` library](https://huggingface.co/docs/datasets/). Models and tokenizers are downloaded using
the [Hugging Face `transformers` library](https://huggingface.co/docs/transformers/).
When the relevant code is executed, the library will automatically fetch and store the required datasets in this directory.

It is important to create a file called `config.py` in the root of the project which defines the following variables

```python
models_cache_dir = "<PATH_TO_MODELS>"
datasets_cache_dir = "<PATH_TO_DATASETS>"
```
When `preload_models.py` and `preload_datasets.py` are executed for the first time, all necessary data is downloaded
to the paths specified in `config.py`. Splits are generated and stored for future use.

### Used Models

1. **[InstaDeepAI/nucleotide-transformer-v2-50m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)**
2. **[InstaDeepAI/nucleotide-transformer-v2-100m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species)**
3. **[InstaDeepAI/nucleotide-transformer-v2-250m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-250m-multi-species)**
4. **[InstaDeepAI/nucleotide-transformer-v2-500m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)**

### Used Datasets

1. **[InstaDeepAI/multi_species_genomes](https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes)**
2. **[InstaDeepAI/nucleotide_transformer_downstream_tasks_revised](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised)**
3. **[katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_ensembl_regulatory)**
4. **[katarinagresova/Genomic_Benchmarks_demo_human_or_worm](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_demo_human_or_worm)**
5. **[katarinagresova/Genomic_Benchmarks_human_ocr_ensembl](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_ocr_ensembl)**
6. **[katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark)**
7. **[katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl)**
8. **[katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs)**
9. **[katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl)**
10. **[katarinagresova/Genomic_Benchmarks_human_enhancers_cohn](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_enhancers_cohn)**
11. **[katarinagresova/Genomic_Benchmarks_human_nontata_promoters](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_nontata_promoters)**


