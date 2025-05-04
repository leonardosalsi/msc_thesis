#!/bin/bash
cd ..

MODEL = 'overlap_multi_species_2kb_sh_gc'

sbatch --export=MODEL,ADDITIONAL --job-name="H4K20me1_${MODEL}" jobs/evaluate_trained/H4K20me1.sh
sbatch --export=MODEL,ADDITIONAL --job-name="gb_drosophila_enhancers_stark_${MODEL}" jobs/evaluate_trained/gb_drosophila_enhancers_stark.sh
sbatch --export=MODEL,ADDITIONAL --job-name="gb_dummy_mouse_enhancers_ensembl_${MODEL}" jobs/evaluate_trained/gb_dummy_mouse_enhancers_ensembl.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9ac_${MODEL}" jobs/evaluate_trained/H3K9ac.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9me3_${MODEL}" jobs/evaluate_trained/H3K9me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K36me3_${MODEL}" jobs/evaluate_trained/H3K36me3.sh

MODEL = 'overlap_logan_ewc_25'

sbatch --export=MODEL,ADDITIONAL --job-name="gb_dummy_mouse_enhancers_ensembl_${MODEL}" jobs/evaluate_trained/gb_dummy_mouse_enhancers_ensembl.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K36me3_${MODEL}" jobs/evaluate_trained/H3K36me3.sh

MODEL = 'overlap_logan_ewc_10'

sbatch --export=MODEL,ADDITIONAL --job-name="H3K27ac_${MODEL}" jobs/evaluate_trained/H3K27ac.sh

MODEL = 'overlap_multi_species_2kb'

sbatch --export=MODEL,ADDITIONAL --job-name="gb_drosophila_enhancers_stark_${MODEL}" jobs/evaluate_trained/gb_drosophila_enhancers_stark.sh
sbatch --export=MODEL,ADDITIONAL --job-name="gb_dummy_mouse_enhancers_ensembl_${MODEL}" jobs/evaluate_trained/gb_dummy_mouse_enhancers_ensembl.sh

MODEL = 'overlap_multi_species_sh_gc'

sbatch --export=MODEL,ADDITIONAL --job-name="gb_dummy_mouse_enhancers_ensembl_${MODEL}" jobs/evaluate_trained/gb_dummy_mouse_enhancers_ensembl.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H2AFZ_${MODEL}" jobs/evaluate_trained/H2AFZ.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me2_${MODEL}" jobs/evaluate_trained/H3K4me2.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9ac_${MODEL}" jobs/evaluate_trained/H3K9ac.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9me3_${MODEL}" jobs/evaluate_trained/H3K9me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H4K20me1_${MODEL}" jobs/evaluate_trained/H4K20me1.sh

MODEL = 'overlap_logan_no_ewc'

sbatch --export=MODEL,ADDITIONAL --job-name="gb_dummy_mouse_enhancers_ensembl_${MODEL}" jobs/evaluate_trained/gb_dummy_mouse_enhancers_ensembl.sh
sbatch --export=MODEL,ADDITIONAL --job-name="gb_demo_coding_vs_intergenomic_seqs_${MODEL}" jobs/evaluate_trained/gb_demo_coding_vs_intergenomic_seqs.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K27me3_${MODEL}" jobs/evaluate_trained/H3K27me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me2_${MODEL}" jobs/evaluate_trained/H3K4me2.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me3_${MODEL}" jobs/evaluate_trained/H3K4me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9ac_${MODEL}" jobs/evaluate_trained/H3K9ac.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9me3_${MODEL}" jobs/evaluate_trained/H3K9me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H4K20me1_${MODEL}" jobs/evaluate_trained/H4K20me1.sh

MODEL = 'overlap_logan_ewc_0_5'

sbatch --export=MODEL,ADDITIONAL --job-name="H3K27ac_${MODEL}" jobs/evaluate_trained/H3K27ac.sh

MODEL = 'overlap_logan_ewc_2'

sbatch --export=MODEL,ADDITIONAL --job-name="gb_drosophila_enhancers_stark_${MODEL}" jobs/evaluate_trained/gb_drosophila_enhancers_stark.sh
sbatch --export=MODEL,ADDITIONAL --job-name="gb_dummy_mouse_enhancers_ensembl_${MODEL}" jobs/evaluate_trained/gb_dummy_mouse_enhancers_ensembl.sh

MODEL = 'overlap_logan_ewc_5'

sbatch --export=MODEL,ADDITIONAL --job-name="H2AFZ_${MODEL}" jobs/evaluate_trained/H2AFZ.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K27me3_${MODEL}" jobs/evaluate_trained/H3K27me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me2_${MODEL}" jobs/evaluate_trained/H3K4me2.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K4me3_${MODEL}" jobs/evaluate_trained/H3K4me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H3K9me3_${MODEL}" jobs/evaluate_trained/H3K9me3.sh
sbatch --export=MODEL,ADDITIONAL --job-name="H4K20me1_${MODEL}" jobs/evaluate_trained/H4K20me1.sh
