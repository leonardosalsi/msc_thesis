#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL --job-name="extract_emb_${MODEL}" jobs/tSNE_embeddings/get_embeddings.sh
sbatch --export=MODEL --job-name="extract_emb_${MODEL}" jobs/tSNE_embeddings/get_embeddings_var.sh