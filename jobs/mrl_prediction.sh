#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL,ADDITIONAL --job-name="mrl_${MODEL}" jobs/mrl_prediction/predict.sh
