#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL --job-name="extract_emb_${MODEL}" jobs/tSNE_embeddings_euler/get_embeddings.sh
