#!/bin/bash
cd ..

sbatch jobs/segment_nt/logan.sh
sbatch jobs/segment_nt/multi_genome_dataset.sh
