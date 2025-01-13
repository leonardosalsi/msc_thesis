#!/bin/bash

sbatch jobs/tokenize/train_dataset.sh
sbatch jobs/tokenize/test_dataset.sh
sbatch jobs/tokenize/validation_dataset.sh