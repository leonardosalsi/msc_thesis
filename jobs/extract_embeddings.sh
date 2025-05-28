#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

#sbatch --export=MODEL --job-name="exemb_5UTR_${MODEL}" jobs/extract_embeddings/extract_5_prime_UTR_embeddings.sh
sbatch --export=MODEL --job-name="exemb_genomelem_${MODEL}" jobs/extract_embeddings/extract_genomic_elements_embeddings.sh