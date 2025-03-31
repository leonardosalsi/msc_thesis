#!/bin/bash
cd ..

sbatch jobs/evaluate_evo2_1b/gb_human_ensembl_regulatory.sh
sbatch jobs/evaluate_evo2_1b/gb_drosophila_enhancers_stark.sh
sbatch jobs/evaluate_evo2_1b/gb_dummy_mouse_enhancers_ensembl.sh
