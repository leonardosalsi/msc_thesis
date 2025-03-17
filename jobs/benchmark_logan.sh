#!/bin/bash
cd ..

sbatch jobs/logan/benchmark/rust.sh
sbatch jobs/logan/benchmark/python.sh