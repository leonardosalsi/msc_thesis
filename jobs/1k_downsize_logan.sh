#!/bin/bash
cd ..

sbatch jobs/logan/1k_filter/generate_logan_31.sh
sbatch jobs/logan/1k_filter/generate_logan_31_reverse.sh
sbatch jobs/logan/1k_filter/generate_logan_28.sh
sbatch jobs/logan/1k_filter/generate_logan_28_reverse.sh
sbatch jobs/logan/1k_filter/generate_logan_25.sh
sbatch jobs/logan/1k_filter/generate_logan_25_reverse.sh
sbatch jobs/logan/1k_filter/generate_logan_20.sh
sbatch jobs/logan/1k_filter/generate_logan_20_reverse.sh
