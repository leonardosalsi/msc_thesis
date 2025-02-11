#!/bin/bash
cd ..

sbatch jobs/evaluate_trained_modelId_2/promoter_all.sh
sbatch jobs/evaluate_trained_modelId_2/H4K20me1.sh
