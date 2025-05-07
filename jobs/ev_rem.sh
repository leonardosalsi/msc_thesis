#!/bin/bash
cd ..

MODEL='overlap_logan_ewc_25'

sbatch  --export=ALL,MODEL="$MODEL",ADDITIONAL="$ADDITIONAL" --job-name="H3K27me3_${MODEL}" jobs/evaluate_trained/H3K27me3.sh


MODEL='overlap_logan_no_ewc'

sbatch  --export=ALL,MODEL="$MODEL",ADDITIONAL="$ADDITIONAL" --job-name="gb_drosophila_enhancers_stark_${MODEL}" jobs/evaluate_trained/gb_drosophila_enhancers_stark.sh
