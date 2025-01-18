#!/bin/bash

sbatch jobs/create_1k_noN_ds/train_dataset.sh
sbatch jobs/create_1k_noN_ds/test_dataset.sh
sbatch jobs/create_1k_noN_ds/validation_dataset.sh