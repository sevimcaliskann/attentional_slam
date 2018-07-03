#!/bin/bash

#montage "$1_scale_0_phase_$2.png" "$1_scale_1_phase_$2.png" "$1_scale_2_phase_$2.png" "$1_scale_3_phase_$2.png" "$1_scale_4_phase_$2.png" "$1_scale_5_phase_$2.png" "$1_scale_6_phase_$2.png" #"$1_scale_7_phase_$2.png" -geometry 265x265\>+2+4 "concats/$1_phase_$2.png"

#montage "gabor_res/concats/$1_phase_0.png" "gabor_res/concats/$1_phase_90.png" "square_res/concats/$1_phase_0.png" "square_res/concats/$1_phase_90.png" "energy_res/concats/$1_concatenated.png" -geometry #128x756 -tile 5x1 "responses_collection/$1_responses_collection.png"

#montage "$1_0.png" "$1_1.png" "$1_2.png" "$1_3.png" "$1_4.png" "$1_5.png" -geometry 556x315 -tile 1x6 "responses_collection/$1_channels.png"

montage "feature_orientation_0.png" "feature_orientation_1.png" "feature_orientation_2.png" "feature_orientation_3.png" "feature_orientation_4.png" "feature_orientation_5.png" "feature_orientation_6.png" "feature_orientation_7.png" -geometry 556x315 -tile 1x8 "responses_collection/orientation_conspicuity.png"

montage "feature_intensity_0.png" "feature_intensity_1.png" -geometry 556x315 -tile 1x2 "responses_collection/intensity_conspicuity.png"

montage "feature_color_0.png" "feature_color_1.png" "feature_color_2.png" "feature_color_3.png" -geometry 556x315 -tile 1x4 "responses_collection/color_conspicuity.png"


montage "conspicuity_0.png" "conspicuity_1.png" "conspicuity_2.png" "conspicuity_3.png" "conspicuity_4.png" "conspicuity_5.png" "conspicuity_6.png" -geometry 199x113 -tile 1x7 "responses_collection/conspicuity.png"

montage "conspicuity_L.png" "conspicuity_a.png" "conspicuity_b.png" "conspicuity_0.png" "conspicuity_45.png" "conspicuity_90.png" "conspicuity_135.png" -geometry 199x113 -tile 1x7 "responses_collection/conspicuity.png"


montage "pyr_center_L_0_0.png" "pyr_center_L_1_0.png" "pyr_center_L_2_0.png" "pyr_center_L_3_0.png" "pyr_center_L_4_0.png" "pyr_center_L_5_0.png" "pyr_center_L_6_0.png" "pyr_center_L_7_0.png" -tile 8x1 "responses_collection/pyramid_L.png"

montage "pyr_center_a_0_0.png" "pyr_center_a_1_0.png" "pyr_center_a_2_0.png" "pyr_center_a_3_0.png" "pyr_center_a_4_0.png" "pyr_center_a_5_0.png" "pyr_center_a_6_0.png" "pyr_center_a_7_0.png" -tile 8x1 "responses_collection/pyramid_a.png"

montage "pyr_center_b_0_0.png" "pyr_center_b_1_0.png" "pyr_center_b_2_0.png" "pyr_center_b_3_0.png" "pyr_center_b_4_0.png" "pyr_center_b_5_0.png" "pyr_center_b_6_0.png" "pyr_center_b_7_0.png" -tile 8x1 "responses_collection/pyramid_b.png"

montage "off_on_L_0.png" "off_on_L_1.png" "off_on_L_2.png" "off_on_L_3.png" "off_on_L_4.png" "off_on_L_5.png" -tile 1x6 "responses_collection/off_on_L.png"

montage "on_off_L_0.png" "on_off_L_1.png" "on_off_L_2.png" "on_off_L_3.png" "on_off_L_4.png" "on_off_L_5.png" -tile 1x6 "responses_collection/on_off_L.png"

montage "off_on_a_0.png" "off_on_a_1.png" "off_on_a_2.png" "off_on_a_3.png" "off_on_a_4.png" "off_on_a_5.png" -tile 1x6 "responses_collection/off_on_a.png"

montage "on_off_a_0.png" "on_off_a_1.png" "on_off_a_2.png" "on_off_a_3.png" "on_off_a_4.png" "on_off_a_5.png" -tile 1x6 "responses_collection/on_off_a.png"

montage "off_on_b_0.png" "off_on_b_1.png" "off_on_b_2.png" "off_on_b_3.png" "off_on_b_4.png" "off_on_b_5.png" -tile 1x6 "responses_collection/off_on_b.png"

montage "on_off_b_0.png" "on_off_b_1.png" "on_off_b_2.png" "on_off_b_3.png" "on_off_b_4.png" "on_off_b_5.png" -tile 1x6 "responses_collection/on_off_b.png"



montage "gaborKernel_0phase_0.png" "gaborKernel_1phase_0.png" "gaborKernel_2phase_0.png" "gaborKernel_3phase_0.png" "gaborKernel_0phase_90.png" "gaborKernel_1phase_90.png" "gaborKernel_2phase_90.png" "gaborKernel_3phase_90.png" -geometry 50x50 -tile 4x2 "gabors_collection.png"

montage "gabor_0_ 0.png" "gabor_0_ 1.png" "gabor_0_ 2.png" "gabor_0_ 3.png" "gabor_0_ 4.png" "gabor_0_ 5.png" -geometry 128x128 -tile 1x6 "responses_collection/norm_gabors0_collection.png"

montage "gabor_45_ 0.png" "gabor_45_ 1.png" "gabor_45_ 2.png" "gabor_45_ 3.png" "gabor_45_ 4.png" "gabor_45_ 5.png" -geometry 128x128 -tile 1x6 "responses_collection/norm_gabors45_collection.png"

montage "gabor_90_ 0.png" "gabor_90_ 1.png" "gabor_90_ 2.png" "gabor_90_ 3.png" "gabor_90_ 4.png" "gabor_90_ 5.png" -geometry 128x128 -tile 1x6 "responses_collection/norm_gabors90_collection.png"

montage "gabor_135_ 0.png" "gabor_135_ 1.png" "gabor_135_ 2.png" "gabor_135_ 3.png" "gabor_135_ 4.png" "gabor_135_ 5.png" -geometry 128x128 -tile 1x6 "responses_collection/norm_gabors135_collection.png"

