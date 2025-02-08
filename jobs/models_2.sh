#!/bin/bash
cd ..

sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K4me1.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K4me2.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K4me3.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K9ac.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K9me3.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H4K20me1.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_human_ensembl_regulatory.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_demo_human_or_worm.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_human_ocr_ensembl.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_drosophila_enhancers_stark.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_dummy_mouse_enhancers_ensembl.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_human_enhancers_ensembl.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_human_enhancers_cohn.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_human_nontata_promoters.sh
