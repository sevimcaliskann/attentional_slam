/*****************************************************************************
*
* VOCUS2.cpp file for the saliency program VOCUS2. 
* A detailed description of the algorithm can be found in the paper: "Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.  
* Please cite this paper if you use our method.
*
* Implementation:	  Thomas Werner   (wernert@cs.uni-bonn.de)
* Design and supervision: Simone Frintrop (frintrop@iai.uni-bonn.de)
*
* Version 1.1
*
* This code is published under the MIT License 
* (see file LICENSE.txt for details)
*
******************************************************************************/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <string>

#include "VOCUS2.h"
using namespace std;

VOCUS2::VOCUS2(){
	// set up default config
	this->cfg = VOCUS2_Cfg();

	// set flags indicating status of intermediate steps
	this->salmap_ready = false;
	this->splitted_ready = false;
	this->processed = false;
}

VOCUS2::VOCUS2(const VOCUS2_Cfg& cfg) {
	// use given config
	this->cfg = cfg;

	// set flags
	this->salmap_ready = false;
	this->splitted_ready = false;
	this->processed = false;
}

VOCUS2::~VOCUS2() {}

void VOCUS2::setCfg(const VOCUS2_Cfg& cfg) {
  this->cfg = cfg;
  this->salmap_ready = false;
  this->splitted_ready = false;
}

void VOCUS2::write_out(string dir){
	if(!salmap_ready) return;

	double mi, ma;
	
	std::cout << "Writing intermediate results to directory: " << dir <<"/"<< endl; 
	
	for(int i = 0; i < (int)pyr_center_L.size(); i++){
        minMaxLoc(pyr_center_L[i], &mi, &ma);
        imwrite(dir + "/pyr_center_L_" + to_string(i) + ".png", (pyr_center_L[i]-mi)/(ma-mi)*255.f);

        minMaxLoc(pyr_center_a[i], &mi, &ma);
        imwrite(dir + "/pyr_center_a_" + to_string(i) + ".png", (pyr_center_a[i]-mi)/(ma-mi)*255.f);

        minMaxLoc(pyr_center_b[i], &mi, &ma);
        imwrite(dir + "/pyr_center_b_" + to_string(i) + ".png", (pyr_center_b[i]-mi)/(ma-mi)*255.f);

        minMaxLoc(pyr_surround_L[i], &mi, &ma);
        imwrite(dir + "/pyr_surround_L_" + to_string(i) + ".png", (pyr_surround_L[i]-mi)/(ma-mi)*255.f);

        minMaxLoc(pyr_surround_a[i], &mi, &ma);
        imwrite(dir + "/pyr_surround_a_" + to_string(i) + ".png", (pyr_surround_a[i]-mi)/(ma-mi)*255.f);

        minMaxLoc(pyr_surround_b[i], &mi, &ma);
        imwrite(dir + "/pyr_surround_b_" + to_string(i) + ".png", (pyr_surround_b[i]-mi)/(ma-mi)*255.f);
	}

	for(int i = 0; i < (int)on_off_L.size(); i++){
		minMaxLoc(on_off_L[i], &mi, &ma);
		imwrite(dir + "/on_off_L_" + to_string(i) + ".png", (on_off_L[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(on_off_a[i], &mi, &ma);
		imwrite(dir + "/on_off_a_" + to_string(i) + ".png", (on_off_a[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(on_off_b[i], &mi, &ma);
		imwrite(dir + "/on_off_b_" + to_string(i) + ".png", (on_off_b[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(off_on_L[i], &mi, &ma);
		imwrite(dir + "/off_on_L_" + to_string(i) + ".png", (off_on_L[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(off_on_a[i], &mi, &ma);
		imwrite(dir + "/off_on_a_" + to_string(i) + ".png", (off_on_a[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(off_on_b[i], &mi, &ma);
		imwrite(dir + "/off_on_b_" + to_string(i) + ".png", (off_on_b[i]-mi)/(ma-mi)*255.f);

		
	}

	vector<Mat> tmp(6);

	tmp[0] = fuse(on_off_L, cfg.fuse_feature);
	minMaxLoc(tmp[0], &mi, &ma);
	imwrite(dir + "/feat_on_off_L.png", (tmp[0]-mi)/(ma-mi)*255.f);
	
	tmp[1] = fuse(on_off_a, cfg.fuse_feature);
	minMaxLoc(tmp[1], &mi, &ma);
	imwrite(dir + "/feat_on_off_a.png", (tmp[1]-mi)/(ma-mi)*255.f);

	tmp[2] = fuse(on_off_b, cfg.fuse_feature);
	minMaxLoc(tmp[2], &mi, &ma);
	imwrite(dir + "/feat_on_off_b.png", (tmp[2]-mi)/(ma-mi)*255.f);

	tmp[3] = fuse(off_on_L, cfg.fuse_feature);
	minMaxLoc(tmp[3], &mi, &ma);
	imwrite(dir + "/feat_off_on_L.png", (tmp[3]-mi)/(ma-mi)*255.f);
	
	tmp[4] = fuse(off_on_a, cfg.fuse_feature);
	minMaxLoc(tmp[4], &mi, &ma);
	imwrite(dir + "/feat_off_on_a.png", (tmp[4]-mi)/(ma-mi)*255.f);

	tmp[5] = fuse(off_on_b, cfg.fuse_feature);
	minMaxLoc(tmp[5], &mi, &ma);
	imwrite(dir + "/feat_off_on_b.png", (tmp[5]-mi)/(ma-mi)*255.f);
	
	for(int i = 0; i < 3; i++){
		vector<Mat> tmp2;
		tmp2.push_back(tmp[i]);
		tmp2.push_back(tmp[i+3]);

		string ch = "";

		switch(i){
		case 0: ch = "L"; break;
		case 1: ch = "a"; break;
		case 2: ch = "b"; break;
		}

		Mat tmp3 = fuse(tmp2, cfg.fuse_feature);

		minMaxLoc(tmp3, &mi, &ma);
		imwrite(dir + "/conspicuity_" + ch + ".png", (tmp3-mi)/(ma-mi)*255.f);
	}

	imwrite(dir + "/salmap.png", salmap);	
}
	
void VOCUS2::process(const Mat& img){
	// clone the input image
	input = img.clone();

	// call process for desired pyramid strcture
	if(cfg.pyr_struct == NEW) pyramid_new(img);  // default
	else if(cfg.pyr_struct == CODI) pyramid_codi(img);
	else pyramid_classic(img);
 
	// set flag indicating that the pyramids are present
	this->processed = true;

	// compute center surround contrast
	center_surround_diff();		

    if(cfg.orientation)	orientationWithCenterSurroundDiff();

}

	
void VOCUS2::pyramid_codi(const Mat& img){
	// clear previous results
	clear();

	// set flags
	salmap_ready = false;
	splitted_ready = false;

	// prepare input image (convert colorspace + split planes)
	planes = prepare_input(img);
	
	// create base pyramids
    vector<Mat> pyr_base_L, pyr_base_a, pyr_base_b, pyr_base_a2, pyr_base_b2;
#pragma omp parallel sections
	{
#pragma omp section
	pyr_base_L = build_multiscale_pyr(planes[0], 1.f);
#pragma omp section
	pyr_base_a = build_multiscale_pyr(planes[1], 1.f);
#pragma omp section
    pyr_base_a2 = build_multiscale_pyr(planes[2], 1.f);
#pragma omp section
    pyr_base_b = build_multiscale_pyr(planes[3], 1.f);
#pragma omp section
    pyr_base_b2 = build_multiscale_pyr(planes[4], 1.f);
	}

	// recompute sigmas that are needed to reach the desired
	// smoothing for center and surround
	float adapted_center_sigma = sqrt(pow(cfg.center_sigma,2)-1);
	float adapted_surround_sigma = sqrt(pow(cfg.surround_sigma,2)-1);

    std::cout << "adapted center sigma: " << adapted_center_sigma << std::endl;
    std::cout << "adapted surround sigma:" << adapted_surround_sigma << std::endl;

	// reserve space
	pyr_center_L.resize(pyr_base_L.size());
	pyr_center_a.resize(pyr_base_L.size());
	pyr_center_b.resize(pyr_base_L.size());
	pyr_surround_L.resize(pyr_base_L.size());
	pyr_surround_a.resize(pyr_base_L.size());
	pyr_surround_b.resize(pyr_base_L.size());

	// for every layer of the pyramid
	for(int o = 0; o < (int)pyr_base_L.size(); o++){

		// for all scales build the center and surround pyramids independently
#pragma omp parallel for

            float scaled_center_sigma = adapted_center_sigma*pow(2.0, (double)o/(double)pyr_base_L.size());
            float scaled_surround_sigma = adapted_surround_sigma*pow(2.0, (double)o/(double)pyr_base_L.size());

            GaussianBlur(pyr_base_L[o], pyr_center_L[o], Size(5,5), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
            //cv::normalize(pyr_center_L[o][s], pyr_center_L[o][s], 0, 255, NORM_MINMAX, CV_8UC1);
            GaussianBlur(pyr_base_L[o], pyr_surround_L[o], Size(5,5), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);
            //cv::normalize(pyr_surround_L[o][s], pyr_surround_L[o][s], 0, 255, NORM_MINMAX, CV_8UC1);

            GaussianBlur(pyr_base_a[o], pyr_center_a[o], Size(5,5), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
            //cv::normalize(pyr_center_a[o][s], pyr_center_a[o][s], 0, 255, NORM_MINMAX, CV_8UC1);
            GaussianBlur(pyr_base_a2[o], pyr_surround_a[o], Size(5,5), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);
            //cv::normalize(pyr_surround_a[o][s], pyr_surround_a[o][s], 0, 255, NORM_MINMAX, CV_8UC1);

            GaussianBlur(pyr_base_b[o], pyr_center_b[o], Size(5,5), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
            //cv::normalize(pyr_center_b[o][s], pyr_center_b[o][s], 0, 255, NORM_MINMAX, CV_8UC1);
            GaussianBlur(pyr_base_b2[o], pyr_surround_b[o], Size(5,5), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);
            //cv::normalize(pyr_surround_b[o][s], pyr_surround_b[o][s], 0, 255, NORM_MINMAX, CV_8UC1);
	}
}

void VOCUS2::pyramid_new(const Mat& img){
	// clear previous results
	clear();

	salmap_ready = false;
	splitted_ready = false;

	// prepare input image (convert colorspace + split channels)
	planes = prepare_input(img);

	// build center pyramid
#pragma omp parallel sections
	{
#pragma omp section
	pyr_center_L = build_multiscale_pyr(planes[0], (float)cfg.center_sigma);
#pragma omp section
	pyr_center_a = build_multiscale_pyr(planes[1], (float)cfg.center_sigma);
#pragma omp section
    pyr_center_b = build_multiscale_pyr(planes[3], (float)cfg.center_sigma);
    cv::namedWindow("view2");

    }

	// compute new surround sigma
	float adapted_sigma = sqrt(pow(cfg.surround_sigma,2)-pow(cfg.center_sigma,2));

	// reserve space
	pyr_surround_L.resize(pyr_center_L.size());
	pyr_surround_a.resize(pyr_center_a.size());
	pyr_surround_b.resize(pyr_center_b.size());

	// for all layers
	for(int o = 0; o < (int)pyr_center_L.size(); o++){

		// for all scales, compute surround counterpart
#pragma omp parallel for
        float scaled_sigma = adapted_sigma*pow(2.0, (double)o/(double)pyr_center_L.size());

        GaussianBlur(pyr_center_L[o], pyr_surround_L[o], Size(5,5), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
        GaussianBlur(pyr_center_a[o], pyr_surround_a[o], Size(5,5), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
        GaussianBlur(pyr_center_b[o], pyr_surround_b[o], Size(5,5), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
	}
}

void VOCUS2::pyramid_classic(const Mat& img){
	// clear previous results
	clear();

	salmap_ready = false;
	splitted_ready = false;

	// prepare input image (convert colorspace + split channels)
	planes = prepare_input(img);

	// compute center and surround pyramid directly but independent
#pragma omp parallel sections
{
#pragma omp section
	pyr_center_L = build_multiscale_pyr(planes[0], (float)cfg.center_sigma);

#pragma omp section
	pyr_center_a = build_multiscale_pyr(planes[1], (float)cfg.center_sigma);

#pragma omp section

    pyr_center_b = build_multiscale_pyr(planes[3], (float)cfg.center_sigma);


#pragma omp section
	pyr_surround_L = build_multiscale_pyr(planes[0], (float)cfg.surround_sigma);

#pragma omp section
    pyr_surround_a = build_multiscale_pyr(planes[2], (float)cfg.surround_sigma);

#pragma omp section
    pyr_surround_b = build_multiscale_pyr(planes[4], (float)cfg.surround_sigma);
}


}

vector<Mat> VOCUS2::build_multiscale_pyr(Mat& mat, float sigma){

    Mat tmp = mat.clone();
    cv::normalize(tmp, tmp, 0, 255, NORM_MINMAX);

    vector<Mat > pyr;
    int num_layer = cfg.stop_layer-cfg.start_layer+1;
    pyr.resize(num_layer);

    // compute pyramid as it is done in [Lowe2004]
    float sig = 0.0f;

    for(int o = cfg.start_layer; o <= cfg.stop_layer; o++){
        sig = pow(2.0, o)*sigma;
        GaussianBlur(tmp, pyr[o], Size(5,5), sig, sig, BORDER_REPLICATE);
    }
    return pyr;
}

void VOCUS2::center_surround_diff(){
    int on_off_size = 3*2;

	on_off_L.resize(on_off_size); off_on_L.resize(on_off_size);
	on_off_a.resize(on_off_size); off_on_a.resize(on_off_size);
	on_off_b.resize(on_off_size); off_on_b.resize(on_off_size);
    int pos = 0;
	// compute DoG by subtracting layers of two pyramids
    for(int o = 2; o <= (int)4; o++){
        Mat diff;
        for(int s = 1; s<=2; s++){
    #pragma omp parallel for

            // ========== L channel ==========
            diff = pyr_center_L[o]-pyr_surround_L[o+s+2];
            threshold(diff, on_off_L[pos], 0, 1, THRESH_TOZERO);
            diff = pyr_surround_L[o+s+2] - pyr_center_L[o];
            threshold(diff, off_on_L[pos], 0, 1, THRESH_TOZERO);

            // ========== a channel ==========
            diff = pyr_center_a[o]-pyr_surround_a[o+s+2];
            threshold(diff, on_off_a[pos], 0, 1, THRESH_TOZERO);
            diff = pyr_surround_a[o+s+2] - pyr_center_a[o];
            threshold(diff, off_on_a[pos], 0, 1, THRESH_TOZERO);

            // ========== b channel ==========
            diff = pyr_center_b[o]-pyr_surround_b[o+s+2];
            threshold(diff, off_on_b[pos], 0, 1, THRESH_TOZERO);
            diff = pyr_surround_b[o+s+2] - pyr_center_b[o];
            threshold(diff, on_off_b[pos], 0, 1, THRESH_TOZERO);

            pos++;

        }

	}
}

std::vector<cv::Mat> VOCUS2::getFeatureChannel(FeatureChannels name){
    switch(name){
    case ON_OFF_L:
        return on_off_L;
    case OFF_ON_L:
        return off_on_L;
    case ON_OFF_A:
        return on_off_a;
    case OFF_ON_A:
        return off_on_a;
    case ON_OFF_B:
        return on_off_b;
    case OFF_ON_B:
        return off_on_b;
    case GABOR:{
        return on_off_gabor0;
    }
    default:
        return on_off_L;
    }
}


void VOCUS2::orientationWithCenterSurroundDiff(){

    int on_off_size = 3*2;
    int filter_size = 11*cfg.center_sigma+1;

    on_off_gabor0.resize(on_off_size); off_on_gabor0.resize(on_off_size);
    on_off_gabor45.resize(on_off_size); off_on_gabor45.resize(on_off_size);
    on_off_gabor90.resize(on_off_size); off_on_gabor90.resize(on_off_size);
    on_off_gabor135.resize(on_off_size); off_on_gabor135.resize(on_off_size);

    Mat gaborKernel0 = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, 0, 10, .5, 2*CV_PI);
    Mat gaborKernel45 = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, M_PI/4, 10, .5, 2*CV_PI);
    Mat gaborKernel90 = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, (2*M_PI)/4, 10, .5, 2*CV_PI);
    Mat gaborKernel135 = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, (3*M_PI)/4, 10, .5, 2*CV_PI);

    float k_sum =  sum(sum(abs(gaborKernel0)))[0];
    gaborKernel0 /= k_sum;
    k_sum =  sum(sum(abs(gaborKernel45)))[0];
    gaborKernel45 /= k_sum;
    k_sum =  sum(sum(abs(gaborKernel90)))[0];
    gaborKernel90 /= k_sum;
    k_sum =  sum(sum(abs(gaborKernel135)))[0];
    gaborKernel135 /= k_sum;
    Mat tmp1, tmp2;
    int pos = 0;
    // compute DoG by subtracting layers of two pyramids


    for(int o = 2; o <=4; o++){
#pragma omp parallel for
        for(int s = 1; s <=2; s++){
            Mat diff;
            // ========== 0 channel ==========
            filter2D(pyr_center_L[o], tmp1, -1, gaborKernel0, Point(-1,-1), 0, BORDER_REPLICATE);
            filter2D(pyr_surround_L[o+s+2], tmp2, -1, gaborKernel0, Point(-1,-1), 0, BORDER_REPLICATE);

            diff = tmp1-tmp2;
            threshold(diff, on_off_gabor0[pos], 0, 1, THRESH_TOZERO);

            diff = tmp2 - tmp1;
            threshold(diff, off_on_gabor0[pos], 0, 1, THRESH_TOZERO);


            // ========== 45 channel ==========
            filter2D(pyr_center_L[o], tmp1, -1, gaborKernel45, Point(-1,-1), 0, BORDER_REPLICATE);
            filter2D(pyr_surround_L[o+s+2], tmp2, -1, gaborKernel45, Point(-1,-1), 0, BORDER_REPLICATE);

            diff = tmp1-tmp2;
            threshold(diff, on_off_gabor45[pos], 0, 1, THRESH_TOZERO);

            diff = tmp2 - tmp1;
            threshold(diff, off_on_gabor45[pos], 0, 1, THRESH_TOZERO);


            // ========== 90 channel ==========
            filter2D(pyr_center_L[o], tmp1, -1, gaborKernel90, Point(-1,-1), 0, BORDER_REPLICATE);
            filter2D(pyr_surround_L[o+s+2], tmp2, -1, gaborKernel90, Point(-1,-1), 0, BORDER_REPLICATE);

            diff = tmp1-tmp2;
            threshold(diff, on_off_gabor90[pos], 0, 1, THRESH_TOZERO);

            diff = tmp2 - tmp1;
            threshold(diff, off_on_gabor90[pos], 0, 1, THRESH_TOZERO);



            // ========== 135 channel ==========
            filter2D(pyr_center_L[o], tmp1, -1, gaborKernel135, Point(-1,-1), 0, BORDER_REPLICATE);
            filter2D(pyr_surround_L[o+s+2], tmp2, -1, gaborKernel135, Point(-1,-1), 0, BORDER_REPLICATE);
            diff = tmp1-tmp2;
            threshold(diff, on_off_gabor135[pos], 0, 1, THRESH_TOZERO);

            diff = tmp2 - tmp1;
            threshold(diff, off_on_gabor135[pos], 0, 1, THRESH_TOZERO);

            pos++;


        }

    }
}



Mat VOCUS2::get_salmap(){

	// check if center surround contrasts are computed
	if(!processed){
		cout << "Image not yet processed. Call process(Mat)." << endl;
		return Mat();
	}

	// if saliency map is already present => return it
	if(salmap_ready) return salmap;

	// intensity feature maps
	vector<Mat> feature_intensity;
	feature_intensity.push_back(fuse(on_off_L, cfg.fuse_feature));
    feature_intensity.push_back(fuse(off_on_L, cfg.fuse_feature));

	// color feature maps
	vector<Mat> feature_color1, feature_color2;

	if(cfg.combined_features){
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
        feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(on_off_b, cfg.fuse_feature)); 
        feature_color1.push_back(fuse(off_on_b, cfg.fuse_feature));

	}
	else{
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
        feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
        feature_color2.push_back(fuse(on_off_b, cfg.fuse_feature));
        feature_color2.push_back(fuse(off_on_b, cfg.fuse_feature));
	}

    vector<Mat> feature_orientation1, feature_orientation2, feature_orientation3, feature_orientation4;
	if(cfg.orientation && cfg.combined_features){
        //for(int i = 0; i < 4; i++){
            //feature_orientation.push_back(fuse(gabor[i], cfg.fuse_feature));
            feature_orientation1.push_back(fuse(on_off_gabor0, cfg.fuse_feature));
            feature_orientation1.push_back(fuse(on_off_gabor45, cfg.fuse_feature));
            feature_orientation1.push_back(fuse(on_off_gabor90, cfg.fuse_feature));
            feature_orientation1.push_back(fuse(on_off_gabor135, cfg.fuse_feature));

            feature_orientation1.push_back(fuse(off_on_gabor0, cfg.fuse_feature));
            feature_orientation1.push_back(fuse(off_on_gabor45, cfg.fuse_feature));
            feature_orientation1.push_back(fuse(off_on_gabor90, cfg.fuse_feature));
            feature_orientation1.push_back(fuse(off_on_gabor135, cfg.fuse_feature));

        //}
    }
    else if(cfg.orientation && !cfg.combined_features){
        feature_orientation1.push_back(fuse(on_off_gabor0, cfg.fuse_feature));
        feature_orientation2.push_back(fuse(on_off_gabor45, cfg.fuse_feature));
        feature_orientation3.push_back(fuse(on_off_gabor90, cfg.fuse_feature));
        feature_orientation4.push_back(fuse(on_off_gabor135, cfg.fuse_feature));

        feature_orientation1.push_back(fuse(off_on_gabor0, cfg.fuse_feature));
        feature_orientation2.push_back(fuse(off_on_gabor45, cfg.fuse_feature));
        feature_orientation3.push_back(fuse(off_on_gabor90, cfg.fuse_feature));
        feature_orientation4.push_back(fuse(off_on_gabor135, cfg.fuse_feature));
    }

	// conspicuity maps
    vector<Mat> conspicuity_maps;
    //conspicuity_maps.push_back(fuse(feature_intensity, cfg.fuse_conspicuity));

    if(cfg.combined_features){
        conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_feature));
        if(cfg.orientation) conspicuity_maps.push_back(fuse(feature_orientation1, cfg.fuse_feature));
		
	}
	else{
        conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_conspicuity));
        conspicuity_maps.push_back(fuse(feature_color2, cfg.fuse_conspicuity));
        if(cfg.orientation){
            //for(int i = 0; i < 4; i++){
                //conspicuity_maps.push_back(fuse(gabor[i], cfg.fuse_feature));
                conspicuity_maps.push_back(fuse(feature_orientation1, cfg.fuse_conspicuity));
                conspicuity_maps.push_back(fuse(feature_orientation2, cfg.fuse_conspicuity));
                conspicuity_maps.push_back(fuse(feature_orientation3, cfg.fuse_conspicuity));
                conspicuity_maps.push_back(fuse(feature_orientation4, cfg.fuse_conspicuity));
            //}
		}
    }



	// saliency map
	salmap = fuse(conspicuity_maps, cfg.fuse_conspicuity);


	// normalize output to [0,1]
	if(cfg.normalize){
        //double mi, ma;
        //minMaxLoc(salmap, &mi, &ma);
        //salmap = (salmap-mi)/(ma-mi);
        cv::normalize(salmap, salmap, 0, 255, NORM_MINMAX, CV_8UC1);
	}

    string dir = "/home/sevim/catkin_ws/src/vocus2/src/results";

    imwrite(dir + "/salmap.png", salmap);

	salmap_ready = true;

	return salmap;
}

Mat VOCUS2::add_center_bias(float lambda){
	if(!salmap_ready) get_salmap();

	// center
	int cr = salmap.rows/2;
	int cc = salmap.cols/2;

	// weight saliency by gaussian
	for(int r = 0; r < salmap.rows; r++){
		float* sal_row = salmap.ptr<float>(r);

		for(int c = 0; c < salmap.cols; c++){
			float d = sqrt((r-cr)*(r-cr)+(c-cc)*(c-cc));
			float fak = exp(-lambda*d*d);
			sal_row[c] *= fak;
		}
	}

	// normalize to [0,1]
	if(cfg.normalize){
		double mi, ma;
		minMaxLoc(salmap, &mi, &ma);
		salmap = (salmap-mi)/(ma-mi);
	}

	return salmap;
} 

vector<Mat> VOCUS2::get_splitted_salmap(){
	if(!processed){
		cout << "Image not yet processed. Call process(Mat)." << endl;
		return vector<Mat>(1, Mat());
	}
	if(splitted_ready) return salmap_splitted;

	salmap_splitted.resize(on_off_L.size());

	for(int o = 0; o < (int)on_off_L.size(); o++){
		Mat tmp = Mat::zeros(on_off_L[o].size(), CV_32F);

		tmp += on_off_L[o];
		tmp += off_on_L[o];
		tmp += on_off_a[o];
		tmp += off_on_a[o];
		tmp += on_off_b[o];
		tmp += off_on_b[o];

		tmp /= 6.f;

		if(cfg.normalize){
			double mi, ma;
			minMaxLoc(tmp, &mi, &ma);
			tmp = (tmp-mi)/(ma-mi);
		}

		salmap_splitted[o] = tmp;
	}
	
	splitted_ready = true;

	return salmap_splitted;
}



float VOCUS2::compute_uniqueness_weight(Mat& img, float t = 0.5){

	// hold maximal points
	vector<Point> point_maxima;

	// hold maximal blobs
	vector<vector<Point> > blob_maxima;

	CV_Assert(img.channels() == 1);

	// find maximum
	double ma;
	minMaxLoc(img, nullptr, &ma);

	// ignore map if global max is too small
	if(ma < 0.05) return 0.f;

	// ignore values < some portion t of the maximal value
	float thresh = ma*t;
	Mat mask;
	threshold(img, mask, thresh, 1, THRESH_BINARY_INV);
	mask.convertTo(mask, CV_8U);

	// number of maxima
	int n_max = 0;

	// for each image pixel
	for(int r = 0; r < img.rows; r++){
		for(int c = 0; c < img.cols; c++){
			
			// skip marked pixel
			if(mask.ptr<uchar>(r)[c] != 0) continue;

			float val = img.ptr<float>(r)[c];

			vector<Point> lower, greater, equal;

			// investigate neighborhood for pixel of values
			// greater, lower or equal to the current pixel
			for(int dr = -1; dr <= 1; dr++){
				for(int dc = -1; dc <= 1; dc++){
					// skip current pixel
					if(dr == 0 && dc == 0) continue;

					// skip out of bound pixels
					if(r+dr < 0 || r+dr >= img.rows) continue;
					if(c+dc < 0 || c+dc >= img.cols) continue;

					float tmp = img.ptr<float>(r+dr)[c+dc];
					Point p = Point(c+dc, r+dr);

					if(tmp < val) lower.push_back(p);
					else if(tmp > val) greater.push_back(p);
					else equal.push_back(p);
				}
			}

			// case 1: isolated point
			if(equal.size() == 0){

				// current point is done
				mask.ptr<uchar>(r)[c] = 1;
				
				// all smaller neighbours are definitive no maxima
				for(Point& p : lower) mask.ptr<uchar>(p.y)[p.x] = 1;

				// if no greater neighbours => maximum
				if(greater.size() == 0){
					// add as maximum
					point_maxima.push_back(Point(c,r));
					n_max++;
				}
			}

			// case 2: blob
			else{
				Mat considered = Mat::zeros(img.size(), CV_8U);
				
				// mark all pixel as considered
				for(Point& p : lower) considered.ptr<uchar>(p.y)[p.x] = 1;
				for(Point& p : equal) considered.ptr<uchar>(p.y)[p.x] = 1;
				for(Point& p : greater) considered.ptr<uchar>(p.y)[p.x] = 1;
				considered.ptr<uchar>(r)[c] = 1;

				// extent point to blob
				int pos = 0;
				while(pos < (int)equal.size()){
					int nr = equal[pos].y;
					int nc = equal[pos].x;

					for(int dr = -1; dr <= 1; dr++){
						for(int dc = -1; dc <= 1; dc++){
							// skip current pixel
							if(dr == 0 && dc == 0) continue;

							// skip out of bound pixels
							if(nr+dr < 0 || nr+dr >= img.rows) continue;
							if(nc+dc < 0 || nc+dc >= img.cols) continue;

							// skip considered pixels
							if(considered.ptr<uchar>(nr+dr)[nc+dc] == 1) continue;

							float tmp = img.ptr<float>(nr+dr)[nc+dc];
							Point p = Point(nc+dc, nr+dr);

							if(tmp < val) lower.push_back(p);
							else if(tmp > val) greater.push_back(p);
							else equal.push_back(p);

							considered.ptr<uchar>(p.y)[p.x] = 1;
						}
					}
					pos++;
				}

				// mark all lower neighbours (definitive no maxima)
				for(Point& p : lower) mask.ptr<uchar>(p.y)[p.x] = 1.f;

				// mark all blob pixels (maxima)
				equal.push_back(Point(c,r));
				for(Point& p : equal) mask.ptr<uchar>(p.y)[p.x] = 1.f;

				// case 2.1: all neighbours are lower
				if(greater.size() == 0){
					blob_maxima.push_back(equal);
					n_max++;
				}
			}
		}
	}

	if(n_max == 0) return 0.f;
	else return 1/sqrt(n_max);
}

//Fuse maps using operation
Mat VOCUS2::fuse(vector<Mat> maps, FusionOperation op){
	
	// resulting map that is returned
	Mat fused = Mat::zeros(maps[0].size(), CV_32F);
	int n_maps = maps.size();	// no. of maps to fuse
	vector<Mat> resized;		// temp. array to hold the resized maps
	resized.resize(n_maps);		// reserve space (needed to use openmp for parallel resizing)

	// ========== ARTIMETIC MEAN ==========

	if(op == ARITHMETIC_MEAN){
#pragma omp parallel for schedule(dynamic, 1)
		for(int i = 0; i < n_maps; i++){
			if(fused.size() != maps[i].size()){
				resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
			else{
				resized[i] = maps[i];
			}
		}

		for(int i = 0; i < n_maps; i++){
			cv::add(fused, resized[i], fused, Mat(), CV_32F);
		}

		fused /= (float)n_maps;
	}

	// ========== MAX ==========

	else if(op == MAX){

#pragma omp parallel for schedule(dynamic, 1)
		for(int i = 0; i < n_maps; i++){
			resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
		}

		for(int i = 0; i < n_maps; i++){
#pragma omp parallel for
			for(int r = 0; r < fused.rows; r++){
				float* row_tmp = resized[i].ptr<float>(r);
				float* row_fused = fused.ptr<float>(r);
				for(int c = 0; c < fused.cols; c++){
					row_fused[c] = max(row_fused[c], row_tmp[c]);
				}
			}
		}
	}

	// ========== UNIQUENESS WEIGHTING ==========

	else if(op == UNIQUENESS_WEIGHT){
		float weight[n_maps];
		for(int i = 0; i < n_maps; i++){
			weight[i] = compute_uniqueness_weight(maps[i]);
			if(weight[i] > 0){
				resize(maps[i]*weight[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
		}

		float sum_weights = 0;

		for(int i = 0; i < n_maps; i++){
			if(weight[i] > 0){
				sum_weights += weight[i];
				cv::add(fused, resized[i], fused, Mat(), CV_32F);
			}
		}

		if(sum_weights > 0) fused /= sum_weights;
	}

    cv::normalize(fused, fused, 0, 255, NORM_MINMAX);

	return fused;
}

vector<Mat> VOCUS2::prepare_input(const Mat& img){

	CV_Assert(img.channels() == 3);
	vector<Mat> planes;
    planes.resize(5);

    Mat converted;
    img.convertTo(converted, CV_32FC3);

    vector<Mat> planes_bgr;
    split(converted, planes_bgr);


    Mat luminance = (planes_bgr[0] + planes_bgr[1] + planes_bgr[2])/3*255.f;
    for(int i = 0; i<planes_bgr.size(); i++){
        planes_bgr[i] /= 255.f;
        cv::divide(planes_bgr[i], luminance, planes_bgr[i]);
        //planes_bgr[i] = planes_bgr[i] - luminance;
        //threshold(planes_bgr[i], planes_bgr[i], 0, 1, THRESH_TOZERO);

    }

    if(cfg.c_space == OPPONENT_CODI){
        planes[0] = luminance;

        planes[1] = planes_bgr[2] - planes_bgr[1];
        threshold(planes[1], planes[1], 0, 1, THRESH_TOZERO);

        planes[2] = planes_bgr[1] - planes_bgr[2];
        threshold(planes[2], planes[2], 0, 1, THRESH_TOZERO);

        planes[3] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f;
        threshold(planes[3], planes[3], 0, 1, THRESH_TOZERO);

        planes[4] = (planes_bgr[1] + planes_bgr[2])/2.f - planes_bgr[0];
        threshold(planes[4], planes[4], 0, 1, THRESH_TOZERO);

	}
	else if(cfg.c_space == OPPONENT){

        planes[0] = luminance;

        planes[1] = planes_bgr[2] - planes_bgr[1]+1.f;
        threshold(planes[1], planes[1], 0, 1, THRESH_TOZERO);
        planes[1] /= 2.f;

        planes[2] = planes_bgr[1] - planes_bgr[2]+1.f;
        threshold(planes[2], planes[2], 0, 1, THRESH_TOZERO);
        planes[2] /= 2.f;

        planes[3] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f+1.f;
        threshold(planes[3], planes[3], 0, 1, THRESH_TOZERO);
        planes[3] /= 2.f;

        planes[4] = (planes_bgr[1] + planes_bgr[2])/2.f - planes_bgr[0] +1.f;
        threshold(planes[4], planes[4], 0, 1, THRESH_TOZERO);
        planes[4] /= 2.f;

	}
	else{
		Mat converted;
		img.convertTo(converted, CV_32FC3);
		converted /= 255.f;
		split(converted, planes);
	}

	return planes;
}

void VOCUS2::clear(){
	salmap.release();
	on_off_L.clear();
	off_on_L.clear();
	on_off_a.clear();
	off_on_a.clear();
	on_off_b.clear();
	off_on_b.clear();

	pyr_center_L.clear();
	pyr_surround_L.clear();
	pyr_center_a.clear();
	pyr_surround_a.clear();
	pyr_center_b.clear();
	pyr_surround_b.clear();
}


