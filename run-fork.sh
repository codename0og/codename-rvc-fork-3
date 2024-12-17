#!/bin/sh
printf "\033]0;Codename-RVC-Fork\007"
. .venv/bin/activate

 export PYTORCH_ENABLE_MPS_FALLBACK=1
 export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
 
clear
python app.py --open
