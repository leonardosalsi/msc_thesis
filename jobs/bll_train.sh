#!/bin/bash_2
cd ..

sbatch jobs/train/train_overlap_2.sh
sbatch jobs/train/train_default_2.sh
sbatch jobs/train/train_overlap_sh_gc_2.sh
sbatch jobs/train/train_default_sh_gc_2.sh

