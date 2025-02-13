#!/bin/bash
cd ..

sbatch jobs/evaluate_trained_modelId_2/gb_demo_coding_vs_intergenomic_seqs.sh
sbatch jobs/evaluate_trained_modelId_2/gb_human_enhancers_ensembl.sh
sbatch jobs/evaluate_trained_modelId_2/gb_human_enhancers_cohn.sh

