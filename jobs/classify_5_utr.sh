#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL,ADDITIONAL --job-name="ben_pat_${MODEL}" jobs/classify_5_utr/classify.sh
