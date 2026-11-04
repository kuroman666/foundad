#!/usr/bin/env bash
set -e

# Layer 28
bash run_inside.sh TMQB28 "/workspace/dataset/DA_Wafer_use/W639/TMQB28/other_status"

# Layer 50
bash run_inside.sh TMRU50 "/workspace/dataset/DA_Wafer_use/W639/TMTN86/other_status"

# Layer 86
bash run_inside.sh TMTN86 "/workspace/dataset/DA_Wafer_use/W639/TMUN87/other_status"

echo "All three layers completed."
