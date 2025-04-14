#!/bin/bash
cd ..

sbatch jobs/logan/filter/mmseq95/mmseqs_1.sh
sbatch jobs/logan/filter/mmseq95/mmseqs_135.sh
