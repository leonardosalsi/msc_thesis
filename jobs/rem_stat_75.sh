#!/bin/bash
cd ..

sbatch jobs/logan/filter/mmseq75/mmseqs_7.sh
sbatch jobs/logan/filter/mmseq75/mmseqs_28.sh
sbatch jobs/logan/filter/mmseq75/mmseqs_87.sh
sbatch jobs/logan/filter/mmseq75/mmseqs_205.sh
sbatch jobs/logan/filter/mmseq75/mmseqs_263.sh
sbatch jobs/logan/filter/mmseq75/mmseqs_303.sh
sbatch jobs/logan/filter/mmseq75/mmseqs_362.sh
sbatch jobs/logan/filter/mmseq75/mmseqs_378.sh
sbatch jobs/logan/filter/mmseq75/mmseqs_447.sh
