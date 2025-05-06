#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL,ADDITIONAL --job-name="utr5_fixed_${MODEL}" jobs/utr5_classification_euler/utr5_fixed.sh
sbatch --export=MODEL,ADDITIONAL --job-name="utr5_${MODEL}" jobs/utr5_classification_euler/utr5.sh
