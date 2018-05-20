#!/bin/bash

#montage "$1_scale_0_phase_$2.png" "$1_scale_1_phase_$2.png" "$1_scale_2_phase_$2.png" "$1_scale_3_phase_$2.png" "$1_scale_4_phase_$2.png" "$1_scale_5_phase_$2.png" "$1_scale_6_phase_$2.png" #"$1_scale_7_phase_$2.png" -geometry 265x265\>+2+4 "concats/$1_phase_$2.png"

montage "$1_scale_2.png" "$1_scale_3.png" "$1_scale_4.png" "$1_scale_5.png" "$1_scale_6.png" "$1_scale_7.png" -tile 1x6 "concats/$1_concatenated.png"

