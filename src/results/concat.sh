#!/bin/bash

#montage "$1_scale_0_phase_$2.png" "$1_scale_1_phase_$2.png" "$1_scale_2_phase_$2.png" "$1_scale_3_phase_$2.png" "$1_scale_4_phase_$2.png" "$1_scale_5_phase_$2.png" "$1_scale_6_phase_$2.png" #"$1_scale_7_phase_$2.png" -geometry 265x265\>+2+4 "concats/$1_phase_$2.png"

#montage "gabor_res/concats/$1_phase_0.png" "gabor_res/concats/$1_phase_90.png" "square_res/concats/$1_phase_0.png" "square_res/concats/$1_phase_90.png" "energy_res/concats/$1_concatenated.png" -geometry #128x756 -tile 5x1 "responses_collection/$1_responses_collection.png"

#montage "$1_0.png" "$1_1.png" "$1_2.png" "$1_3.png" "$1_4.png" "$1_5.png" -geometry 556x315 -tile 1x6 "responses_collection/$1_channels.png"

montage "feature_orientation_0.png" "feature_orientation_1.png" "feature_orientation_2.png" "feature_orientation_3.png" "feature_orientation_4.png" "feature_orientation_5.png" "feature_orientation_6.png" "feature_orientation_7.png" -geometry 556x315 -tile 1x8 "responses_collection/orientation_conspicuity.png"

montage "feature_intensity_0.png" "feature_intensity_1.png" -geometry 556x315 -tile 1x2 "responses_collection/intensity_conspicuity.png"

montage "feature_color_0.png" "feature_color_1.png" "feature_color_2.png" "feature_color_3.png" -geometry 556x315 -tile 1x4 "responses_collection/color_conspicuity.png"



