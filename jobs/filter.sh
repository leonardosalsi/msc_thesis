#!/bin/bash
cd ..

sbatch jobs/filer_downsize-chunks-multi_genome_dataset-train-1_2kbp-from_scratch/filter_downsize-chunks-multi_genome_dataset-train-1_2kbp-from_scratch.sh
sbatch jobs/filer_downsize-chunks-multi_genome_dataset-train-2_2kbp-from_scratch/filter_downsize-chunks-multi_genome_dataset-train-2_2kbp-from_scratch.sh
