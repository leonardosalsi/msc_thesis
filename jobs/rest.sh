#!/bin/bash
cd ..

sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K36me3.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K4me3.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_human_enhancers_ensembl.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_human_enhancers_cohn.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-Genomic_Benchmarks_human_nontata_promoters.sh
