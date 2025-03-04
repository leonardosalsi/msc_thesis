#!/bin/bash
cd ..

sbatch jobs/train/train_logan.sh
sbatch jobs/train/train_logan_unfiltered.sh

