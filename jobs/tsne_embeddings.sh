#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL --job-name="tsne_mean_${MODEL}" jobs/tSNE_embeddings/get_embeddings_mean.sh
sbatch --export=MODEL --job-name="tsne_cls_${MODEL}" jobs/tSNE_embeddings/get_embeddings_cls.sh