#!/bin/bash
sbatch \
  --array=7,28,87,205,263,303,362,378,447 \
  jobs/logan/filter/mmseq75/mmseqs_%a.sh
