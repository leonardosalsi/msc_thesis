#!/bin/bash

sbatch jobs/create_1k_ds/train_dataset.sh
sbatch jobs/create_1k_ds/test_dataset.sh
sbatch jobs/create_1k_ds/validation_dataset.sh