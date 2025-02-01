#!/bin/bash
cd ..

sbatch jobs/filter_downsize-chunks-multi_genome_dataset-train-1_2kbp/filter_downsize-chunks-multi_genome_dataset-train-1_2kbp.sh
sbatch jobs/filter_downsize-chunks-multi_genome_dataset-train-2_2kbp/filter_downsize-chunks-multi_genome_dataset-train-2_2kbp.sh