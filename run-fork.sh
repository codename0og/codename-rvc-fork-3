#!/bin/sh
printf "\033]0;Applio\007"
. .venv/bin/activate

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Uncomment the one below in case of inference issues on idk.. M1, M2 or M3? chips.
#export OMP_NUM_THREADS=1

clear
python app.py --open
