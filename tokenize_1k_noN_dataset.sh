#!/bin/bash

sbatch jobs/tokenize-noN/train_dataset.sh
sbatch jobs/tokenize-noN/test_dataset.sh
sbatch jobs/tokenize-noN/validation_dataset.sh