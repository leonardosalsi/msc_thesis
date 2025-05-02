#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me2_${MODEL}" jobs/evaluate_trained_euler/H3K4me2.sh
