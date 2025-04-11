#!/bin/bash
cd ..

sbatch jobs/logan/filter/mmseq95/mmseqs_1.sh
sbatch jobs/logan/filter/mmseq95/mmseqs_7.sh
sbatch jobs/logan/filter/mmseq95/mmseqs_9.sh
sbatch jobs/logan/filter/mmseq95/mmseqs_28.sh
sbatch jobs/logan/filter/mmseq95/mmseqs_432.sh
sbatch jobs/logan/filter/mmseq95/mmseqs_447.sh
sbatch jobs/logan/filter/mmseq95/mmseqs_454.sh
sbatch jobs/logan/filter/mmseq95/mmseqs_493.sh
