#!/bin/bash
cd ..

sbatch jobs/logan/filter/generate_logan_31_reverse.sh
sbatch jobs/logan/filter/generate_logan_28_reverse.sh
sbatch jobs/logan/filter/generate_logan_25_reverse.sh
sbatch jobs/logan/filter/generate_logan_20_reverse.sh
