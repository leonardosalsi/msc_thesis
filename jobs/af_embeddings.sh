#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL --job-name="af_mean_${MODEL}" jobs/af_embeddings/get_embeddings_mean.sh
sbatch --export=MODEL --job-name="af_cls_${MODEL}" jobs/af_embeddings/get_embeddings_cls.sh