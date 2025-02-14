#!/bin/bash
cd ..

sbatch jobs/train/train_overlap.sh
sbatch jobs/train/train_default.sh
sbatch jobs/train/train_overlap_sh_gc.sh
sbatch jobs/train/train_default_sh_gc.sh

