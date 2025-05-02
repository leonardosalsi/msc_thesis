#!/bin/bash
cd ..

if [ -z "$MODEL" ]; then
  echo "Error: MODEL is not set"
  exit 1
fi

sbatch --export=MODEL,ADDITIONAL --job-name="promoter_all_${MODEL}" jobs/evaluate_trained_euler/promoter_all.sh
sbatch --export=MODEL,ADDITIONAL --job-name="promoter_tata_${MODEL}" jobs/evaluate_trained_euler/promoter_tata.sh
sbatch --export=MODEL,ADDITIONAL --job-name="promoter_no_tata_${MODEL}" jobs/evaluate_trained_euler/promoter_no_tata.sh
sbatch --export=MODEL,ADDITIONAL --job-name="enhancers_${MODEL}" jobs/evaluate_trained_euler/enhancers.sh
sbatch --export=MODEL,ADDITIONAL --job-name="enhancers_types_${MODEL}" jobs/evaluate_trained_euler/enhancers_types.sh
sbatch --export=MODEL,ADDITIONAL --job-name="splice_sites_all_${MODEL}" jobs/evaluate_trained_euler/splice_sites_all.sh
sbatch --export=MODEL,ADDITIONAL --job-name="splice_sites_acceptors_${MODEL}" jobs/evaluate_trained_euler/splice_sites_acceptors.sh
sbatch --export=MODEL,ADDITIONAL --job-name="splice_sites_donors_${MODEL}" jobs/evaluate_trained_euler/splice_sites_donors.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H2AFZ_${MODEL}" jobs/evaluate_trained_euler/H2AFZ.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K27ac_${MODEL}" jobs/evaluate_trained_euler/H3K27ac.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K27me3_${MODEL}" jobs/evaluate_trained_euler/H3K27me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K36me3_${MODEL}" jobs/evaluate_trained_euler/H3K36me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me1_${MODEL}" jobs/evaluate_trained_euler/H3K4me1.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me2_${MODEL}" jobs/evaluate_trained_euler/H3K4me2.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me3_${MODEL}" jobs/evaluate_trained_euler/H3K4me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9ac_${MODEL}" jobs/evaluate_trained_euler/H3K9ac.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9me3_${MODEL}" jobs/evaluate_trained_euler/H3K9me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H4K20me1_${MODEL}" jobs/evaluate_trained_euler/H4K20me1.sh
