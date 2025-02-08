#!/bin/bash
cd ..

sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-promoter_all.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-promoter_tata.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-promoter_no_tata.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-enhancers.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-enhancers_types.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-splice_sites_all.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-splice_sites_acceptors.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-splice_sites_donors.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H2AFZ.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K27ac.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K27me3.sh
sbatch jobs/eval-models-from-trained-1-123456789101112131415161718192021222324252627-10-lora/eval-model-nucleotide-transformer-v2-50m-multi-species-H3K36me3.sh
