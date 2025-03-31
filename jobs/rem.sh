#!/bin/bash
cd ..


sbatch jobs/evaluate_evo2_1b/enhancers.sh
sbatch jobs/evaluate_evo2_1b/enhancers_types.sh
