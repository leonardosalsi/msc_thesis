#!/bin/bash
cd ..

sbatch jobs/evaluate_trained_modelId_0/splice_sites_all.sh
sbatch jobs/evaluate_trained_modelId_0/H3K4me2.sh
sbatch jobs/evaluate_trained_modelId_0/H3K4me3.sh
sbatch jobs/evaluate_trained_modelId_0/H3K9ac.sh
sbatch jobs/evaluate_trained_modelId_0/H3K9me3.sh
sbatch jobs/evaluate_trained_modelId_0/H4K20me1.sh
sbatch jobs/evaluate_trained_modelId_0/gb_human_ensembl_regulatory.sh
sbatch jobs/evaluate_trained_modelId_0/gb_demo_human_or_worm.sh

